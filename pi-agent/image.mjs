// Image generation for canvas agents on the Pi harness: one picture from a prompt, and optionally from images to
// work from, through the ChatGPT subscription's Codex backend and its built-in `image_generation` tool.
//
// stdin: one JSON line, {prompt, images: [{data, mimeType}], model?}.
// stdout: one JSON line, {ok: true, image (base64 PNG), revised_prompt} or {ok: false, error}.
//
// The request and the stream parser are ported from pi-codex-image-gen 0.1.15 (Apache-2.0, Jose Mocito,
// https://github.com/jvm/pi-mono/tree/main/packages/pi-codex-image-gen). The extension needs Pi's coding agent
// and loads through Pi's extension API, which this sidecar doesn't have, so the parts that make the request live
// here. It signs in with the `openai-codex` login conveyor already keeps (`conveyor auth login openai`), the
// login the extension accepts as its fallback, and Pi's own provider refreshes the token.

import { pathToFileURL } from "node:url";
import { createModels } from "@earendil-works/pi-ai";
import { openaiCodexProvider } from "@earendil-works/pi-ai/providers/openai-codex";
import { JsonCredentialStore } from "./credentials.mjs";

export const CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses";
export const DEFAULT_MODEL = "gpt-6-astra"; // the routing model; the backend picks the image model
const REQUEST_TIMEOUT_MS = 5 * 60_000;
const MAX_RESPONSE_BYTES = 100 * 1024 * 1024;
const MAX_IMAGE_BYTES = 32 * 1024 * 1024;
const MAX_TEXT_CHARS = 4000;
const MAX_RETRIES = 3;
const MAX_RETRY_DELAY_MS = 30_000;
const PNG = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
const QUOTA_CODES = new Set([
	"insufficient_quota", "quota_exceeded", "usage_limit_reached", "usage_limit_exceeded",
	"billing_hard_limit_reached", "billing_not_active", "organization_usage_limit_exceeded",
	"workspace_member_usage_limit_reached",
]);

const object = (value) => (value !== null && typeof value === "object" && !Array.isArray(value) ? value : {});

// Backend text can echo the request; never let a token through.
function clean(value, secrets) {
	if (typeof value !== "string") return "";
	let out = value;
	for (const secret of secrets) if (secret) out = out.split(secret).join("[redacted]");
	return out.replace(/\beyJ[A-Za-z0-9_.-]*/g, "[redacted]").replace(/[\u0000-\u001f\u007f-\u009f]/g, " ").slice(0, MAX_TEXT_CHARS);
}

export function accountId(token) {
	try {
		const payload = JSON.parse(Buffer.from(token.split(".")[1], "base64url").toString("utf8"));
		const id = payload["https://api.openai.com/auth"].chatgpt_account_id;
		if (typeof id === "string" && /^[a-zA-Z0-9_-]{1,128}$/.test(id)) return id;
	} catch {}
	throw new Error("The OpenAI Codex login has no ChatGPT account id. Run `conveyor auth login openai` again.");
}

export function requestBody(prompt, images, model, sessionId) {
	return {
		model,
		store: false,
		stream: true,
		prompt_cache_key: sessionId,
		instructions:
			"You are generating bitmap image assets. For this request, call the image_generation tool exactly once. Do not answer with only text unless image generation is unavailable.",
		input: [{
			role: "user",
			content: [
				{ type: "input_text", text: prompt },
				...images.map((image) => ({ type: "input_image", image_url: `data:${image.mimeType};base64,${image.data}` })),
			],
		}],
		tools: [{ type: "image_generation", output_format: "png" }],
		tool_choice: "auto",
		parallel_tool_calls: false,
		text: { verbosity: "low" },
	};
}

function hint(error) {
	const { code, type } = object(error);
	if ([code, type].some((v) => typeof v === "string" && QUOTA_CODES.has(v)))
		return "The ChatGPT subscription's image quota is used up for now.";
	if (code === "moderation_blocked" || type === "image_generation_user_error")
		return "The backend refused this prompt. Describe the picture differently.";
	return "The backend could not make the image.";
}

async function failure(response, secrets) {
	let error = {};
	try {
		error = object(object(JSON.parse((await response.text()).slice(0, 16 * 1024))).error);
	} catch {}
	const terminal = [error.code, error.type].some((v) => QUOTA_CODES.has(v)) || error.code === "moderation_blocked"
		|| error.type === "image_generation_user_error";
	const why = response.status === 401
		? "The OpenAI Codex login was rejected. Run `conveyor auth login openai` again."
		: response.headers.get("cf-mitigated") === "challenge" ? "Cloudflare challenged the connection." : hint(error);
	return { message: clean(`Image request failed (HTTP ${response.status}). ${why}`, secrets),
		retry: !terminal && [429, 500, 502, 503, 504].includes(response.status) };
}

// The Responses stream: server-sent events, one finished `image_generation_call` item holding the base64 image.
export async function parseStream(body, secrets = []) {
	const parsed = { text: "" };
	let completed = false;
	const take = (value) => {
		const item = object(value);
		if (item.type !== "image_generation_call") return;
		if (parsed.image) throw new Error("The backend returned more than one image.");
		if (item.status !== "completed" || typeof item.result !== "string" || !item.result)
			throw new Error("The backend's image generation did not complete.");
		parsed.image = item.result;
		parsed.revisedPrompt = clean(item.revised_prompt, secrets) || null;
	};
	const handle = (frame) => {
		const data = frame.split(/\r?\n/).filter((l) => l.startsWith("data:")).map((l) => l.slice(5).trim()).join("\n");
		if (!data || data === "[DONE]") return;
		let event;
		try {
			event = object(JSON.parse(data));
		} catch {
			throw new Error("The backend sent an unreadable stream event.");
		}
		if (event.type === "error" || event.type === "response.failed")
			throw new Error(hint(object(event.response).error ?? event.error ?? event));
		if (event.type === "response.incomplete") throw new Error("The backend's response was incomplete.");
		if (event.type === "response.output_text.delta" && typeof event.delta === "string")
			parsed.text = (parsed.text + event.delta).slice(0, MAX_TEXT_CHARS);
		if (event.type === "response.output_item.done") take(event.item);
		if (event.type === "response.completed") {
			const output = object(event.response).output;
			if (!parsed.image && Array.isArray(output)) output.forEach(take);
			completed = true;
		}
	};
	const decoder = new TextDecoder();
	let buffer = "";
	let bytes = 0;
	for await (const chunk of body) {
		bytes += chunk.byteLength;
		if (bytes > MAX_RESPONSE_BYTES) throw new Error("The backend's response was too large.");
		buffer += decoder.decode(chunk, { stream: true });
		let match;
		while (!completed && (match = /\r?\n\r?\n/.exec(buffer))) {
			handle(buffer.slice(0, match.index));
			buffer = buffer.slice(match.index + match[0].length);
		}
		if (completed) break;
	}
	if (!completed && buffer.trim()) handle(buffer + decoder.decode());
	if (!completed) throw new Error("The backend's stream ended before the image was done.");
	parsed.text = clean(parsed.text, secrets);
	return parsed;
}

export function checkPng(base64) {
	const value = base64.trim();
	if (value.length > Math.ceil(MAX_IMAGE_BYTES / 3) * 4 || value.length % 4 || /[^A-Za-z0-9+/=]/.test(value))
		throw new Error("The backend returned bad image data.");
	const bytes = Buffer.from(value, "base64");
	if (bytes.length < 8 || !bytes.subarray(0, 8).equals(PNG)) throw new Error("The backend's image isn't a PNG.");
	return value;
}

const sleep = (ms, signal) => new Promise((resolve, reject) => {
	const timer = setTimeout(resolve, ms);
	signal.addEventListener("abort", () => { clearTimeout(timer); reject(signal.reason); }, { once: true });
});

export async function generate({ prompt, images = [], model = DEFAULT_MODEL }, { token, fetchFn = fetch, sessionId = "conveyor-image" }) {
	if (typeof prompt !== "string" || !prompt.trim() || prompt.length > 32_000) throw new Error("The prompt must be 1 to 32,000 characters.");
	if (images.length > 5) throw new Error("At most five images to work from.");
	const secrets = [token, accountId(token)];
	const body = JSON.stringify(requestBody(prompt, images, model, sessionId));
	const headers = {
		Authorization: `Bearer ${token}`, "chatgpt-account-id": secrets[1], originator: "pi", "User-Agent": "conveyor",
		"OpenAI-Beta": "responses=experimental", accept: "text/event-stream", "content-type": "application/json",
	};
	const controller = new AbortController();
	const timer = setTimeout(() => controller.abort(new Error("Image generation timed out after 5 minutes.")), REQUEST_TIMEOUT_MS);
	try {
		for (let attempt = 1; ; attempt += 1) {
			let response;
			try {
				response = await fetchFn(CODEX_RESPONSES_URL, { method: "POST", headers, body, signal: controller.signal, redirect: "error" });
			} catch (error) {
				if (controller.signal.aborted) throw controller.signal.reason;
				throw new Error(`Couldn't reach the image backend: ${clean(String(error?.message || error), secrets)}`);
			}
			if (response.ok) {
				const parsed = await parseStream(response.body, secrets);
				if (!parsed.image) throw new Error(parsed.text ? `No image came back. The backend said: ${parsed.text}` : "No image came back.");
				return { image: checkPng(parsed.image), revisedPrompt: parsed.revisedPrompt };
			}
			const { message, retry } = await failure(response, secrets);
			if (!retry || attempt > MAX_RETRIES) throw new Error(message);
			const after = Number(response.headers.get("retry-after"));
			await sleep(Math.min(Number.isFinite(after) && after > 0 ? after * 1000 : 1000 * 2 ** (attempt - 1), MAX_RETRY_DELAY_MS), controller.signal);
		}
	} finally {
		clearTimeout(timer);
	}
}

async function readStdin() {
	let text = "";
	process.stdin.setEncoding("utf8");
	for await (const chunk of process.stdin) text += chunk;
	return JSON.parse(text);
}

async function main() {
	const request = await readStdin();
	const models = createModels({ credentials: new JsonCredentialStore() });
	models.setProvider(openaiCodexProvider());
	const auth = await models.getAuth("openai-codex");
	const token = auth?.auth?.apiKey;
	if (!token) throw new Error("Not signed in to OpenAI Codex. Run `conveyor auth login openai`.");
	const out = await generate(request, { token, sessionId: request.sessionId || "conveyor-image" });
	process.stdout.write(`${JSON.stringify({ ok: true, image: out.image, revised_prompt: out.revisedPrompt })}\n`);
}

if (import.meta.url === pathToFileURL(process.argv[1] || "").href) {
	main().catch((error) => {
		process.stdout.write(`${JSON.stringify({ ok: false, error: String(error?.message || error) })}\n`);
		process.exitCode = 1;
	});
}
