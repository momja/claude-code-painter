import assert from "node:assert/strict";
import test from "node:test";
import { accountId, CODEX_RESPONSES_URL, generate, parseStream, requestBody } from "./image.mjs";

const PNG = Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), Buffer.from("rest")]).toString("base64");
const claims = { "https://api.openai.com/auth": { chatgpt_account_id: "acct_1" } };
const TOKEN = `eyJhbGciOiJub25lIn0.${Buffer.from(JSON.stringify(claims)).toString("base64url")}.sig`;

const sse = (...events) => events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join("");
const stream = (text, size = 7) => new ReadableStream({
	start(controller) {
		const bytes = new TextEncoder().encode(text);
		for (let i = 0; i < bytes.length; i += size) controller.enqueue(bytes.slice(i, i + size));
		controller.close();
	},
});
const done = (item) => [{ type: "response.created", response: { id: "resp_1" } },
	{ type: "response.image_generation_call.generating" },
	{ type: "response.output_item.done", item },
	{ type: "response.completed", response: { id: "resp_1", output: [] } }];

test("the request asks for one PNG and carries the images to work from", () => {
	const body = requestBody("a hand, three angles", [{ data: "AAAA", mimeType: "image/png" }], "gpt-6-astra", "s1");
	assert.equal(body.model, "gpt-6-astra");
	assert.deepEqual(body.tools, [{ type: "image_generation", output_format: "png" }]);
	assert.deepEqual(body.input[0].content, [{ type: "input_text", text: "a hand, three angles" },
		{ type: "input_image", image_url: "data:image/png;base64,AAAA" }]);
	assert.equal(accountId(TOKEN), "acct_1");
	assert.throws(() => accountId("not.a.jwt"), /conveyor auth login openai/);
});

test("the stream parser finds the image across chunk boundaries", async () => {
	const item = { type: "image_generation_call", status: "completed", result: PNG, revised_prompt: `study ${TOKEN}` };
	const parsed = await parseStream(stream(sse(...done(item))), [TOKEN]);
	assert.equal(parsed.image, PNG);
	assert.equal(parsed.revisedPrompt, "study [redacted]");
});

test("a failed response or a stream cut short is an error", async () => {
	await assert.rejects(parseStream(stream(sse({ type: "response.failed", response: { error: { code: "moderation_blocked" } } }))),
		/refused this prompt/);
	await assert.rejects(parseStream(stream(sse({ type: "response.created", response: {} }))), /ended before/);
});

test("generate retries a 503, then returns the image", async () => {
	const calls = [];
	const fetchFn = async (url, init) => {
		calls.push([url, init.headers["chatgpt-account-id"]]);
		if (calls.length === 1) return new Response("{}", { status: 503, headers: { "retry-after": "0.01" } });
		return new Response(stream(sse(...done({ type: "image_generation_call", status: "completed", result: PNG }))));
	};
	const out = await generate({ prompt: "a hand" }, { token: TOKEN, fetchFn });
	assert.equal(out.image, PNG);
	assert.deepEqual(calls, [[CODEX_RESPONSES_URL, "acct_1"], [CODEX_RESPONSES_URL, "acct_1"]]);
});

test("a quota error isn't retried", async () => {
	let calls = 0;
	const fetchFn = async () => {
		calls += 1;
		return new Response(JSON.stringify({ error: { code: "usage_limit_reached" } }), { status: 429 });
	};
	await assert.rejects(generate({ prompt: "a hand" }, { token: TOKEN, fetchFn }), /quota is used up/);
	assert.equal(calls, 1);
});
