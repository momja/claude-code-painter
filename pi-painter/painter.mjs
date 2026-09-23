// Pi sidecar for conveyor's conversation painting harness.
//
// One process paints one picture. Pi's Agent keeps the whole conversation and runs the tool loop; every
// tool call (a brush stroke, `look`, `finish`) is executed by the Python host, which owns the canvas.
//
// Protocol: newline-delimited JSON. stdout carries protocol messages only; diagnostics go to stderr.
//   host -> sidecar: start, tool_result, abort
//   sidecar -> host: llm_request, llm_params, llm_delta, llm_end, tool, tool_error, note, fatal, done

import { Agent } from "@earendil-works/pi-agent-core";
import {
	Type,
	createModels,
	fauxAssistantMessage,
	fauxProvider,
	fauxText,
	fauxThinking,
	fauxToolCall,
} from "@earendil-works/pi-ai";
import { opencodeGoProvider } from "@earendil-works/pi-ai/providers/opencode-go";
import { openrouterProvider } from "@earendil-works/pi-ai/providers/openrouter";

const DELTA_INTERVAL_MS = 300;
const LENGTH_NUDGE =
	"Your last reply ran out of room while thinking and made no tool calls. " +
	"Don't analyze further. Reply now with tool calls.";

const send = (msg) => process.stdout.write(`${JSON.stringify(msg)}\n`);
const log = (...parts) => process.stderr.write(`${parts.join(" ")}\n`);

// ---- stdin: split on \n only, per Pi's JSONL guidance ------------------------------------------------

const pendingTools = new Map();
let resolveConfig;
const configReady = new Promise((resolve) => {
	resolveConfig = resolve;
});
let agent = null;
let buffer = "";

process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
	buffer += chunk;
	let newline = buffer.indexOf("\n");
	while (newline >= 0) {
		const line = buffer.slice(0, newline);
		buffer = buffer.slice(newline + 1);
		if (line.trim()) {
			try {
				handleHost(JSON.parse(line));
			} catch (error) {
				log("bad message from host:", error.message);
			}
		}
		newline = buffer.indexOf("\n");
	}
});
process.stdin.on("end", () => {
	agent?.abort();
	for (const pending of pendingTools.values()) pending.reject(new Error("host closed the connection"));
});

function handleHost(msg) {
	if (msg.type === "start") {
		resolveConfig(msg);
	} else if (msg.type === "tool_result") {
		const pending = pendingTools.get(msg.id);
		if (!pending) return log("result for unknown tool call", msg.id);
		pendingTools.delete(msg.id);
		pending.resolve(msg);
	} else if (msg.type === "abort") {
		agent?.abort();
	}
}

function callHost(toolCallId, name, args) {
	return new Promise((resolve, reject) => {
		pendingTools.set(toolCallId, { resolve, reject });
		send({ type: "tool", id: toolCallId, name, args });
	});
}

// ---- tools --------------------------------------------------------------------------------------------

let stopRequested = false;

function makeTool(spec, width, height) {
	let parameters;
	if (spec.kind === "brush") {
		parameters = Type.Object({
			x: Type.Number({ description: `Start x, 0 to ${width}` }),
			y: Type.Number({ description: `Start y, 0 to ${height}` }),
			angle: Type.Number({ description: "Direction in degrees. 0 points right, 90 down." }),
			color: Type.String({ description: "Hex color, for example #8a6d4f" }),
			pressure: Type.Optional(Type.Number({ description: "0.1 to 1, default 1. Lower lays less pigment." })),
		});
	} else if (spec.kind === "finish") {
		parameters = Type.Object({ note: Type.Optional(Type.String({ description: "Optional closing note" })) });
	} else {
		parameters = Type.Object({});
	}
	return {
		name: spec.name,
		label: spec.name,
		description: spec.description,
		parameters,
		executionMode: "sequential", // strokes land on one canvas in order
		execute: async (toolCallId, args) => {
			const result = await callHost(toolCallId, spec.name, args);
			if (result.stop) stopRequested = true;
			if (result.isError) throw new Error(result.text || "tool failed");
			const content = [];
			if (result.text) content.push({ type: "text", text: result.text });
			for (const data of result.images || []) content.push({ type: "image", data, mimeType: "image/png" });
			return { content, details: result.details || {} };
		},
	};
}

// ---- helpers ------------------------------------------------------------------------------------------

// GLM answers 400 too_many_images above 8 per request, and a painting keeps every image it has sent. The host
// caps looks so this shouldn't trigger; it's here so a long conversation can't break the run if it does.
function trimImages(messages, limit) {
	if (!limit) return messages;
	// The first image is the target, which the painter needs to the end; the rest are older views of its own
	// canvas, and the newest of those is the only one worth keeping.
	const positions = [];
	messages.forEach((msg, i) => {
		if (!Array.isArray(msg.content)) return;
		msg.content.forEach((block, j) => {
			if (block.type === "image") positions.push([i, j]);
		});
	});
	if (positions.length <= limit) return messages;
	const keep = new Set([`${positions[0][0]}:${positions[0][1]}`]); // the target
	for (const [i, j] of positions.slice(-(limit - 1))) keep.add(`${i}:${j}`);

	const trimmed = messages.slice();
	const touched = new Map();
	for (const [i, j] of positions) {
		if (keep.has(`${i}:${j}`)) continue;
		const content = (touched.get(i) || trimmed[i].content).slice();
		content[j] = { type: "text", text: "[older canvas view dropped: the provider limits images per conversation]" };
		touched.set(i, content);
	}
	for (const [i, content] of touched) trimmed[i] = { ...trimmed[i], content };
	return trimmed;
}

function snapshot(message) {
	let thinking = "";
	let text = "";
	const toolCalls = [];
	for (const block of message.content || []) {
		if (block.type === "thinking") thinking += block.thinking || "";
		else if (block.type === "text") text += block.text || "";
		else if (block.type === "toolCall") toolCalls.push({ id: block.id, name: block.name, arguments: block.arguments });
	}
	return { thinking, text, toolCalls };
}

function fauxMessage(response) {
	const blocks = response.blocks.map((b) => {
		if (b.thinking !== undefined) return fauxThinking(b.thinking);
		if (b.text !== undefined) return fauxText(b.text);
		return fauxToolCall(b.tool, b.args || {});
	});
	const stopReason = response.stopReason || (response.blocks.some((b) => b.tool) ? "toolUse" : "stop");
	return fauxAssistantMessage(blocks, { stopReason });
}

// ---- main ---------------------------------------------------------------------------------------------

async function main() {
	const cfg = await configReady;
	const models = createModels();
	let model;
	if (cfg.faux) {
		// Scripted responses for tests: the real agent loop and protocol, no network.
		const faux = fauxProvider({ provider: "faux", models: [{ id: "faux-painter", reasoning: true }] });
		models.setProvider(faux.provider);
		model = faux.getModel("faux-painter");
		faux.setResponses(cfg.faux.map(fauxMessage));
	} else {
		const provider = cfg.provider || "opencode-go";
		models.setProvider(provider === "openrouter" ? openrouterProvider() : opencodeGoProvider());
		// Pi ships its own table for both providers; cfg.model covers a model this pi-ai predates.
		model = models.getModel(provider, cfg.model.id) ?? {
			...cfg.model,
			api: "openai-completions",
			provider,
			baseUrl: cfg.baseUrl,
		};
	}

	let turn = 0;
	let sent = 0;
	let lastDelta = 0;
	let lengthNudges = 0;
	let lastStop = null;

	const streamFn = (m, context, options) => {
		turn += 1;
		// What this request adds to the conversation. Earlier assistant turns are recorded as their own calls.
		const messages = context.messages || [];
		const added = messages.slice(sent).filter((msg) => msg.role !== "assistant");
		sent = messages.length;
		send({
			type: "llm_request",
			turn,
			nMessages: messages.length,
			messages: added,
			systemPrompt: turn === 1 ? context.systemPrompt : undefined,
			tools:
				turn === 1 ? (context.tools || []).map((t) => ({ name: t.name, description: t.description })) : undefined,
		});
		return models.streamSimple(m, context, {
			...options,
			maxTokens: cfg.maxTokens,
			cacheRetention: cfg.cacheRetention,
			maxRetries: 3,
			// OpenCode Go rejects a request without this (400 MissingSessionID) and routes on it. This pi-ai
			// doesn't send it, so the painting's own session id goes out by hand.
			headers: cfg.sessionHeader ? { [cfg.sessionHeader]: cfg.sessionId } : undefined,
			onPayload: (payload) => {
				send({
					type: "llm_params",
					turn,
					params: {
						max_tokens: payload.max_completion_tokens ?? payload.max_tokens,
						reasoning: payload.reasoning ?? payload.reasoning_effort,
						tool_choice: payload.tool_choice,
						prompt_cache_key: payload.prompt_cache_key,
						wire_messages: (payload.messages || []).length,
					},
				});
				return undefined;
			},
		});
	};

	agent = new Agent({
		initialState: {
			systemPrompt: cfg.systemPrompt,
			model,
			thinkingLevel: cfg.thinkingLevel,
			tools: cfg.tools.map((spec) => makeTool(spec, cfg.width, cfg.height)),
		},
		streamFn,
		transformContext: async (messages) => trimImages(messages, cfg.maxImages),
		sessionId: cfg.sessionId, // sent upstream as x-session-id, so every turn lands on the same cache
		toolExecution: "sequential",
		shouldStopAfterTurn: async () => stopRequested || turn >= cfg.maxTurns,
	});

	agent.subscribe(async (event) => {
		if (event.type === "message_update" && event.message?.role === "assistant") {
			const now = Date.now();
			if (now - lastDelta >= DELTA_INTERVAL_MS) {
				lastDelta = now;
				send({ type: "llm_delta", turn, ...snapshot(event.message) });
			}
		} else if (event.type === "message_end" && event.message?.role === "assistant") {
			const m = event.message;
			const snap = snapshot(m);
			lastStop = m.stopReason;
			lastDelta = 0;
			send({
				type: "llm_end",
				turn,
				...snap,
				stopReason: m.stopReason,
				errorMessage: m.errorMessage,
				usage: m.usage,
				model: m.model,
			});
			if (m.stopReason === "length" && snap.toolCalls.length === 0 && !stopRequested && lengthNudges < 2) {
				lengthNudges += 1;
				send({ type: "note", turn, text: "Ran out of room while thinking; nudged it to act." });
				agent.followUp({ role: "user", content: LENGTH_NUDGE, timestamp: Date.now() });
			}
		} else if (event.type === "tool_execution_end" && event.isError) {
			// Covers failures the host never saw, such as calls to tools that don't exist.
			const text = (event.result?.content || []).map((c) => c.text || "").join(" ");
			send({ type: "tool_error", id: event.toolCallId, name: event.toolName, error: text });
		}
	});

	try {
		await agent.prompt({ role: "user", content: cfg.prompt, timestamp: Date.now() });
	} catch (error) {
		send({ type: "fatal", error: String(error?.stack || error) });
	}
	send({ type: "done", turns: turn, stopReason: lastStop, error: agent.state.errorMessage ?? null });
	process.exit(0);
}

main().catch((error) => {
	send({ type: "fatal", error: String(error?.stack || error) });
	process.exit(1);
});
