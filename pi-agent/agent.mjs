// Pi agent sidecar for conveyor: a drop-in for `claude -p`.
//
// stdin: one JSON line, the job's config (model, system prompt, first message, MCP server, tools, schema).
// stdout: Claude Code's stream-json events (system/init, assistant, user tool results, result), so the host
// records a Pi session exactly as it records a Claude Code one. Diagnostics go to stderr.
//
// The sidecar launches the job's MCP server itself (the paint server or the workbench), hands its tools to Pi's
// agent loop as-is (Pi validates plain JSON Schema parameters), and forwards each call with the tool call id in
// `_meta`, where the paint server reads it to join its stroke log to the transcript. Structured output is a
// `respond` tool whose parameters are the job's JSON schema.

import { spawn } from "node:child_process";
import {
	Agent,
	BACKGROUND_CONTEXT,
	COMPACTION_SUMMARY_PREFIX,
	COMPACTION_SUMMARY_SUFFIX,
	generateSummaryWithUsage,
} from "@earendil-works/pi-agent-core";
import {
	createModels,
	fauxAssistantMessage,
	fauxProvider,
	fauxText,
	fauxThinking,
	fauxToolCall,
} from "@earendil-works/pi-ai";
import { opencodeGoProvider } from "@earendil-works/pi-ai/providers/opencode-go";
import { openrouterProvider } from "@earendil-works/pi-ai/providers/openrouter";
import { openaiCodexProvider } from "@earendil-works/pi-ai/providers/openai-codex";
import { JsonCredentialStore } from "./credentials.mjs";
import { boundedPainterContext, textChars } from "./painter-context.mjs";
import { isTransientTransportError, MAX_TRANSPORT_RETRIES, retryDelayMs } from "./transient.mjs";

const LENGTH_NUDGE =
	"Your last reply ran out of room while thinking and made no tool calls. Don't analyze further. Reply now with tool calls.";
const RESPOND_RULE =
	"\n\nWhen you have your answer, call the `respond` tool with it, exactly once. That is the only way your answer is recorded.";

const emit = (msg) => process.stdout.write(`${JSON.stringify(msg)}\n`);
const log = (...parts) => process.stderr.write(`${parts.join(" ")}\n`);

// ---- stdin: the config line -----------------------------------------------------------------------------

function readConfig() {
	return new Promise((resolve, reject) => {
		let buffer = "";
		process.stdin.setEncoding("utf8");
		process.stdin.on("data", (chunk) => {
			buffer += chunk;
			const newline = buffer.indexOf("\n");
			if (newline >= 0) {
				process.stdin.pause();
				try {
					resolve(JSON.parse(buffer.slice(0, newline)));
				} catch (error) {
					reject(error);
				}
			}
		});
		process.stdin.on("end", () => reject(new Error("stdin closed before a config line arrived")));
	});
}

// ---- a minimal MCP client over the server's stdio ---------------------------------------------------------

class McpClient {
	constructor(command, args) {
		this.proc = spawn(command, args, { stdio: ["pipe", "pipe", "inherit"] });
		this.pending = new Map();
		this.nextId = 1;
		this.buffer = "";
		this.proc.stdout.setEncoding("utf8");
		this.proc.stdout.on("data", (chunk) => {
			this.buffer += chunk;
			let newline = this.buffer.indexOf("\n");
			while (newline >= 0) {
				const line = this.buffer.slice(0, newline);
				this.buffer = this.buffer.slice(newline + 1);
				if (line.trim()) this.onLine(line);
				newline = this.buffer.indexOf("\n");
			}
		});
		this.proc.on("exit", (code) => {
			for (const pending of this.pending.values()) pending.reject(new Error(`the MCP server exited (${code})`));
			this.pending.clear();
		});
	}

	onLine(line) {
		let msg;
		try {
			msg = JSON.parse(line);
		} catch {
			return log("MCP server wrote a non-JSON line:", line.slice(0, 200));
		}
		const pending = this.pending.get(msg.id);
		if (!pending) return;
		this.pending.delete(msg.id);
		if (msg.error) pending.reject(new Error(msg.error.message || "MCP error"));
		else pending.resolve(msg.result);
	}

	request(method, params) {
		const id = this.nextId++;
		return new Promise((resolve, reject) => {
			this.pending.set(id, { resolve, reject });
			this.proc.stdin.write(`${JSON.stringify({ jsonrpc: "2.0", id, method, params })}\n`);
		});
	}

	async connect() {
		await this.request("initialize", {
			protocolVersion: "2025-06-18",
			capabilities: {},
			clientInfo: { name: "conveyor-pi", version: "1" },
		});
		this.proc.stdin.write(`${JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" })}\n`);
		return (await this.request("tools/list", {})).tools || [];
	}

	call(name, args, toolUseId) {
		return this.request("tools/call", { name, arguments: args, _meta: {
			"claudecode/toolUseId": toolUseId, "conveyor/paintingState": true,
		} });
	}

	close() {
		try {
			this.proc.stdin.end();
		} catch {}
	}
}

// ---- helpers ------------------------------------------------------------------------------------------------

// GLM answers 400 too_many_images above 8 per request, and a conversation keeps every image it has seen. Keep
// the first image (the target) and the newest ones; older canvas views are stale anyway.
function trimImages(messages, limit) {
	if (!limit) return messages;
	const positions = [];
	messages.forEach((msg, i) => {
		if (!Array.isArray(msg.content)) return;
		msg.content.forEach((block, j) => {
			if (block.type === "image") positions.push([i, j]);
		});
	});
	if (positions.length <= limit) return messages;
	const keep = new Set([`${positions[0][0]}:${positions[0][1]}`]);
	for (const [i, j] of positions.slice(-(limit - 1))) keep.add(`${i}:${j}`);
	const trimmed = messages.slice();
	const touched = new Map();
	for (const [i, j] of positions) {
		if (keep.has(`${i}:${j}`)) continue;
		const content = (touched.get(i) || trimmed[i].content).slice();
		content[j] = { type: "text", text: "[older image dropped: the provider limits images per request]" };
		touched.set(i, content);
	}
	for (const [i, content] of touched) trimmed[i] = { ...trimmed[i], content };
	return trimmed;
}

function claudeBlocks(message) {
	const blocks = [];
	for (const block of message.content || []) {
		if (block.type === "thinking" && block.thinking) blocks.push({ type: "thinking", thinking: block.thinking });
		else if (block.type === "text" && block.text) blocks.push({ type: "text", text: block.text });
		else if (block.type === "toolCall")
			blocks.push({ type: "tool_use", id: block.id, name: block.name, input: block.arguments || {} });
	}
	return blocks;
}

function claudeResultContent(content) {
	return (content || []).map((block) =>
		block.type === "image"
			? { type: "image", source: { type: "base64", media_type: block.mimeType || "image/png", data: block.data } }
			: { type: "text", text: block.text || "" },
	);
}

// A model this pi-ai doesn't list. Its API follows a listed model of the same family where there is one
// (gpt-6-luna speaks what gpt-5.6-luna speaks), and otherwise the catalog's hint from the Python side.
function unknownModel(models, cfg) {
	const family = (cfg.model.id.match(/^[a-z]+/i) || [""])[0].toLowerCase();
	const listed = (models.getModels?.(cfg.provider) || []).filter((m) => m.id.toLowerCase().startsWith(`${family}-`));
	const sibling = family ? listed[0] : undefined;
	const api = sibling?.api || cfg.model.api || "openai-completions";
	const baseUrl = sibling?.baseUrl || (api === "anthropic-messages" ? cfg.baseUrl.replace(/\/v1$/, "") : cfg.baseUrl);
	log(`${cfg.model.id} isn't in pi-ai's table; using ${api} at ${baseUrl}` + (sibling ? ` like ${sibling.id}` : ""));
	return { ...cfg.model, api, provider: cfg.provider, baseUrl };
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

// Compaction after N looks. Pi's own compaction waits for the context window to fill, which these runs rarely
// reach: a provider cuts the stream off or limits images first. So the sidecar compacts on a count instead. Once
// `everyLooks` canvas views (tool results holding an image) have piled up since the last compaction, everything
// before the most recent look is summarized to text (the summarizer drops images) and replaced by that summary.
// The first message (the task and the target) and the newest look onward stay as they are.
const COMPACT_INSTRUCTIONS =
	"This is a painter's working history. Keep what it needs to carry on: the plan, which regions are done and " +
	"which are not, the colours and sizes that worked, mistakes to avoid, and the latest score with the actions and " +
	"looks left. Leave out per-stroke detail.";

function looksIn(messages) {
	return messages.filter(
		(m) => m.role === "toolResult" && Array.isArray(m.content) && m.content.some((b) => b.type === "image"),
	).length;
}

// Where to cut: the start of the assistant turn that made the newest look, so the tail keeps that look and every
// tool call keeps its result. -1 when there is no such turn after `from`.
function cutBeforeLastLook(messages, from) {
	let look = -1;
	for (let i = messages.length - 1; i >= from; i--) {
		const m = messages[i];
		if (m.role === "toolResult" && Array.isArray(m.content) && m.content.some((b) => b.type === "image")) {
			look = i;
			break;
		}
	}
	for (let i = look; i > from; i--) if (messages[i].role === "assistant") return i;
	return -1;
}

function makeCompactor({ models, model, everyLooks, headers, reserveTokens, onSummary }) {
	let boundary = 1; // messages[1..boundary) are summarized; messages[boundary..] are live
	let summary = null;
	let broken = false;
	// The summary request is a plain completion; this wrapper adds the provider's session header to it.
	const withHeaders = {
		completeSimple: (m, aiContext, options) =>
			models.completeSimple(m, aiContext, { ...options, headers: { ...options?.headers, ...headers } }),
	};
	const view = (messages) =>
		summary === null
			? messages
			: [
					messages[0],
					{ role: "user", content: COMPACTION_SUMMARY_PREFIX + summary + COMPACTION_SUMMARY_SUFFIX, timestamp: Date.now() },
					...messages.slice(boundary),
				];
	return async (messages) => {
		if (broken || messages.length <= boundary) return view(messages);
		try {
			if (looksIn(messages.slice(boundary)) < everyLooks) return view(messages);
			const cut = cutBeforeLastLook(messages, boundary);
			if (cut <= boundary) return view(messages);
			const result = await generateSummaryWithUsage(
				messages.slice(boundary, cut),
				withHeaders,
				model,
				reserveTokens,
				COMPACT_INSTRUCTIONS,
				summary ?? undefined,
				undefined,
				undefined,
				undefined,
				BACKGROUND_CONTEXT,
			);
			if (!result.ok) throw new Error(result.error?.message || String(result.error));
			summary = result.value.text;
			onSummary({ summarized: cut - boundary, kept: messages.length - cut, chars: summary.length, usage: result.value.usage });
			boundary = cut;
		} catch (error) {
			broken = true; // one failure and we stop trying; the image trim still holds the request down
			log(`compaction failed, continuing without it: ${error?.message || error}`);
		}
		return view(messages);
	};
}

// ---- main ---------------------------------------------------------------------------------------------------

async function main() {
	const cfg = await readConfig();
	const started = Date.now();
	const models = createModels({ credentials: new JsonCredentialStore() });
	let model;
	if (cfg.faux) {
		const faux = fauxProvider({ provider: "faux", models: [{ id: cfg.model.id || "faux-model", reasoning: true }] });
		models.setProvider(faux.provider);
		model = faux.getModel(cfg.model.id || "faux-model");
		faux.setResponses(cfg.faux.map(fauxMessage));
	} else {
		const provider = cfg.provider === "openrouter"
			? openrouterProvider()
			: cfg.provider === "openai-codex"
				? openaiCodexProvider()
				: opencodeGoProvider();
		models.setProvider(provider);
		// Pi ships its own table for these providers; the catalog covers models this pi-ai predates.
		model = models.getModel(cfg.provider, cfg.model.id) ?? unknownModel(models, cfg);
	}

	let mcp = null;
	let stopRequested = false;
	let structured = null;
	let paintingState = null;
	const tools = [];
	const serverName = cfg.mcp?.name || null;
	if (cfg.mcp) {
		mcp = new McpClient(cfg.mcp.command, cfg.mcp.args || []);
		const listed = await mcp.connect();
		const allowed = new Set(cfg.tools || []);
		for (const t of listed) {
			if (allowed.size && !allowed.has(t.name)) continue;
			tools.push({
				name: t.name,
				label: t.name,
				description: t.description || "",
				parameters: t.inputSchema || { type: "object", properties: {} },
				executionMode: "sequential", // strokes land on one canvas in order
				execute: async (toolCallId, args) => {
					const result = await mcp.call(t.name, args, toolCallId);
					if (result.structuredContent?.painting_state) paintingState = result.structuredContent.painting_state;
					// A server can end the session itself: the shared canvas does once an agent's tool calls run out.
					if (result.structuredContent?.stop) stopRequested = true;
					const content = (result.content || []).map((block) =>
						block.type === "image"
							? { type: "image", data: block.data, mimeType: block.mimeType || "image/png" }
							: { type: "text", text: block.text || "" },
					);
					if (result.isError) throw new Error(content.map((c) => c.text || "").join(" ") || "tool failed");
					if ((cfg.stopTools || []).includes(t.name)) stopRequested = true;
					return { content, details: paintingState ? { paintingState } : {} };
				},
			});
		}
	}
	if (cfg.jsonSchema) {
		tools.push({
			name: "respond",
			label: "respond",
			description: "Record your answer. Call it exactly once, when you have the answer.",
			parameters: cfg.jsonSchema,
			executionMode: "sequential",
			execute: async (toolCallId, args) => {
				structured = args;
				stopRequested = true;
				return { content: [{ type: "text", text: "Recorded. You're done." }], details: {} };
			},
		});
	}

	emit({
		type: "system",
		subtype: "init",
		harness: "pi",
		provider: cfg.provider,
		model: model.id,
		tools: tools.map((t) => t.name),
		mcp_servers: serverName ? [{ name: serverName, status: "connected" }] : [],
	});

	let turn = 0;
	let nudges = 0;
	let cost = 0;
	let overBudget = false;
	let lastText = "";
	let lastError = null;
	const usage = { input_tokens: 0, output_tokens: 0, cache_read_input_tokens: 0, cache_creation_input_tokens: 0, reasoning: 0, compactions: 0, context_trims: 0 };

	// One line per request, so a stalled session shows what it was waiting on: how much it had sent, and (once the
	// request ends) how long the first byte took. A request that never ends leaves only its `request_start`.
	let requestNo = 0;
	let requestStart = 0;
	let firstEventMs = null;
	const streamFn = (m, context, options) => {
		requestNo += 1;
		requestStart = Date.now();
		firstEventMs = null;
		let images = 0;
		let imageBytes = 0;
		for (const msg of context.messages || []) {
			if (!Array.isArray(msg.content)) continue;
			for (const block of msg.content) {
				if (block.type !== "image") continue;
				images += 1;
				imageBytes += (block.data || "").length;
			}
		}
		emit({ type: "system", subtype: "request_start", request: requestNo, messages: (context.messages || []).length, images, image_bytes: imageBytes, text_chars: textChars(context.messages || []) });
		return models.streamSimple(m, context, {
			...options,
			maxTokens: cfg.maxTokens,
			maxRetries: 3,
			// OpenCode Go rejects a request without this header and routes on it; this pi-ai doesn't send it.
			headers,
		});
	};

	const headers = cfg.sessionHeader ? { [cfg.sessionHeader]: cfg.sessionId } : undefined;
	const compact = cfg.compactEveryLooks
		? makeCompactor({
				models,
				model,
				everyLooks: cfg.compactEveryLooks,
				headers,
				reserveTokens: cfg.maxTokens || 16384,
				onSummary: ({ summarized, kept, chars, usage: summaryUsage }) => {
					cost += summaryUsage?.cost?.total || 0;
					usage.input_tokens += summaryUsage?.input || 0;
					usage.output_tokens += summaryUsage?.output || 0;
					usage.cache_read_input_tokens += summaryUsage?.cacheRead || 0;
					usage.cache_creation_input_tokens += summaryUsage?.cacheWrite || 0;
					usage.reasoning += summaryUsage?.reasoning || 0;
					usage.compactions += 1;
					log(`compacted ${summarized} messages into ${chars} chars, kept ${kept}`);
					emit({ type: "system", subtype: "compact_boundary", summarized, kept, summary_chars: chars });
				},
			})
		: null;

	const agent = new Agent({
		initialState: {
			systemPrompt: cfg.systemPrompt + (cfg.jsonSchema ? RESPOND_RULE : ""),
			model,
			thinkingLevel: cfg.thinkingLevel || "off",
			tools,
		},
		streamFn,
		transformContext: async (messages) => {
			const context = compact ? await compact(messages) : boundedPainterContext(messages, {
				keepTurns: cfg.paintContextTurns,
				state: paintingState,
				onTrim: (stats) => {
					usage.context_trims += 1;
					emit({ type: "system", subtype: "painter_context", ...stats });
				},
			});
			return trimImages(context, cfg.maxImages);
		},
		sessionId: cfg.sessionId,
		toolExecution: "sequential",
		shouldStopAfterTurn: async () => stopRequested || overBudget || turn >= (cfg.maxTurns || 400),
	});

	agent.subscribe(async (event) => {
		if (event.type === "message_update" && firstEventMs === null) firstEventMs = Date.now() - requestStart;
		if (event.type === "message_end" && event.message?.role === "assistant") {
			turn += 1;
			const m = event.message;
			emit({
				type: "system",
				subtype: "request_end",
				request: requestNo,
				first_event_ms: firstEventMs,
				total_ms: Date.now() - requestStart,
				stop_reason: m.stopReason,
				error: m.errorMessage || null,
				input_tokens: m.usage?.input || 0,
			});
			const u = m.usage || {};
			usage.input_tokens += u.input || 0;
			usage.output_tokens += u.output || 0;
			usage.cache_read_input_tokens += u.cacheRead || 0;
			usage.cache_creation_input_tokens += u.cacheWrite || 0;
			usage.reasoning += u.reasoning || 0;
			cost += u.cost?.total || 0;
			if (cfg.maxBudgetUsd && cost >= cfg.maxBudgetUsd) overBudget = true;
			const blocks = claudeBlocks(m);
			const text = blocks.filter((b) => b.type === "text").map((b) => b.text).join("\n");
			if (text) lastText = text;
			if (m.stopReason === "error" || m.stopReason === "aborted") lastError = m.errorMessage || m.stopReason;
			emit({
				type: "assistant",
				message: { id: `msg_${turn}`, model: m.model || model.id, content: blocks, stop_reason: m.stopReason },
				...(lastError && (m.stopReason === "error" || m.stopReason === "aborted") ? { error: lastError } : {}),
			});
			const calledTools = blocks.some((b) => b.type === "tool_use");
			if (m.stopReason === "length" && !calledTools && !stopRequested && nudges < 2) {
				nudges += 1;
				agent.followUp({ role: "user", content: LENGTH_NUDGE, timestamp: Date.now() });
			}
		} else if (event.type === "tool_execution_end") {
			emit({
				type: "user",
				message: {
					role: "user",
					content: [
						{
							type: "tool_result",
							tool_use_id: event.toolCallId,
							content: claudeResultContent(event.result?.content),
							is_error: Boolean(event.isError),
						},
					],
				},
			});
		}
	});

	let fatal = null;
	try {
		await agent.prompt({ role: "user", content: cfg.content, timestamp: Date.now() });
		// A dropped connection ends the run with an errored assistant message. Remove it and take the turn again
		// from the last user or tool-result message; the work already done in the session stays.
		for (let attempt = 1; attempt <= MAX_TRANSPORT_RETRIES; attempt += 1) {
			const messages = agent.state.messages;
			const last = messages[messages.length - 1];
			if (stopRequested || overBudget || last?.role !== "assistant" || last.stopReason !== "error") break;
			if (!isTransientTransportError(last.errorMessage)) break;
			log(`transport error, retry ${attempt}/${MAX_TRANSPORT_RETRIES}: ${last.errorMessage}`);
			emit({ type: "system", subtype: "transport_retry", attempt, error: last.errorMessage });
			agent.state.messages = messages.slice(0, -1);
			lastError = null;
			await new Promise((resolve) => setTimeout(resolve, retryDelayMs(attempt)));
			await agent.continue();
		}
	} catch (error) {
		fatal = String(error?.stack || error);
		log(fatal);
	}
	mcp?.close();
	const errorText = fatal || agent.state.errorMessage || lastError;
	const subtype = overBudget ? "error_max_budget_usd" : errorText && !structured && !stopRequested ? "error_during_execution" : "success";
	emit({
		type: "result",
		subtype,
		is_error: subtype !== "success",
		num_turns: turn,
		total_cost_usd: cost,
		duration_ms: Date.now() - started,
		usage: {
			input_tokens: usage.input_tokens,
			output_tokens: usage.output_tokens,
			cache_read_input_tokens: usage.cache_read_input_tokens,
			cache_creation_input_tokens: usage.cache_creation_input_tokens,
			output_tokens_details: { thinking_tokens: usage.reasoning },
			compactions: usage.compactions,
			context_trims: usage.context_trims,
		},
		result: lastText,
		structured_output: structured,
		errors: subtype === "success" ? [] : [errorText || subtype],
		terminal_reason: stopRequested ? "completed" : overBudget ? "max_budget" : turn >= (cfg.maxTurns || 400) ? "max_turns" : "completed",
	});
	process.exit(0);
}

main().catch((error) => {
	emit({ type: "result", subtype: "error_during_execution", is_error: true, num_turns: 0, total_cost_usd: 0, errors: [String(error?.stack || error)] });
	process.exit(1);
});
