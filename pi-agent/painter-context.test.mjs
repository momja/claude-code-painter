import assert from "node:assert/strict";
import test from "node:test";
import { boundedPainterContext, textChars, trimImages } from "./painter-context.mjs";

const image = (data) => ({ type: "image", data, mimeType: "image/png" });
const task = () => ({ role: "user", content: [{ type: "text", text: "Target and demo" }, image("target"), image("demo")], timestamp: 1 });
const state = { actions_used: 50, actions_left: 150, looks_left: 4, score: 0.6, pen: { down: true, color: "#123456", x: 10 },
	scope: [10, 10, 30, 30], plan: "Background done. Face next.", finished: false };

function turn(n, view = false) {
	return [
		{ role: "assistant", content: [{ type: "thinking", thinking: `plan ${n}`, thinkingSignature: "opaque" },
			{ type: "toolCall", name: "paint_batch", id: `paint-${n}`, arguments: { calls: [{ x: n }], plan: "x".repeat(1000) } },
			...(view ? [{ type: "toolCall", name: "look", id: `view-${n}`, arguments: {} }] : [])], timestamp: n * 10 },
		{ role: "toolResult", toolCallId: `paint-${n}`, toolName: "paint_batch", content: [{ type: "text", text: `painted ${n}` }], timestamp: n * 10 + 1 },
		...(view ? [{ role: "toolResult", toolCallId: `view-${n}`, toolName: "look", content: [image(`canvas-${n}`)],
			details: { paintingState: { actions_used: n } }, timestamp: n * 10 + 2 }] : []),
	];
}

function assertPaired(messages) {
	const calls = messages.flatMap((m) => m.role === "assistant" ? m.content.filter((b) => b.type === "toolCall").map((b) => b.id) : []);
	const results = messages.filter((m) => m.role === "toolResult").map((m) => m.toolCallId);
	assert.deepEqual(results, calls);
}

const bound = (messages, options = {}) => boundedPainterContext(messages, { keepTurns: 2, state, ...options });

test("short histories, disabled mode and absent state are unchanged", () => {
	const short = [task(), ...turn(1), ...turn(2)];
	assert.equal(bound(short), short);
	const long = [...short, ...turn(3)];
	assert.equal(bound(long, { keepTurns: 0 }), long);
	assert.equal(bound(long, { state: null }), long);
});

test("keep last two complete turns and authoritative state, without mutating history", () => {
	const messages = [task(), ...turn(1), ...turn(2), ...turn(3), ...turn(4)];
	const before = JSON.stringify(messages);
	let stats;
	const result = bound(messages, { onTrim: (value) => { stats = value; } });
	assert.equal(result[0], messages[0]);
	assert.deepEqual(result.filter((m) => m.role === "assistant").map((m) => m.timestamp), [30, 40]);
	assertPaired(result);
	assert.match(result.at(-1).content, /AFTER the retained history/);
	assert.match(result.at(-1).content, /Background done\. Face next/);
	assert.match(result.at(-1).content, /"actions_left":150/);
	assert.match(result.at(-1).content, /"scope":\[10,10,30,30\]/);
	assert.equal(JSON.stringify(messages), before);
	assert.equal(stats.dropped, 4);
	assert.ok(stats.after_chars < stats.before_chars);
	assert.equal(result[1].content[0].thinkingSignature, "opaque");
});

test("latest older view retains its entire tool-call turn", () => {
	const messages = [task(), ...turn(1, true), ...turn(2, true), ...turn(3), ...turn(4), ...turn(5)];
	const result = bound(messages);
	assertPaired(result);
	assert.deepEqual(result.filter((m) => m.role === "assistant").map((m) => m.timestamp), [20, 40, 50]);
	assert.equal(result.find((m) => m.toolCallId === "view-2").content[0].data, "canvas-2");
	assert.match(result.at(-1).content, /"view_actions_used":2/);
});

test("only the newest canvas view's images remain in retained history", () => {
	const messages = [task(), ...turn(1), ...turn(2, true), ...turn(3, true)];
	const result = bound(messages);
	assert.deepEqual(result.flatMap((m) => Array.isArray(m.content) ? m.content.filter((b) => b.type === "image").map((b) => b.data) : []),
		["target", "demo", "canvas-3"]);
	assert.equal(messages.find((m) => m.toolCallId === "view-2").content[0].type, "image");
	assertPaired(result);
});

test("history remains bounded over 100 batches even when the newest view is old", () => {
	const messages = [task(), ...turn(1, true)];
	let sizeAtTen;
	for (let i = 2; i <= 100; i++) {
		messages.push(...turn(i));
		const result = bound(messages);
		assertPaired(result);
		assert.ok(result.filter((m) => m.role === "assistant").length <= 3);
		assert.ok(result.length <= 9);
		if (i === 10) sizeAtTen = textChars(result);
		if (i > 10) assert.ok(textChars(result) < sizeAtTen + 50);
	}
	assert.ok(textChars(bound(messages)) < textChars(messages) / 20);
});

test("a recovery nudge immediately before retained turns survives", () => {
	const nudge = { role: "user", content: "Reply with tool calls now.", timestamp: 29 };
	const messages = [task(), ...turn(1), ...turn(2), nudge, ...turn(3), ...turn(4)];
	assert.ok(bound(messages).includes(nudge));
	assertPaired(bound(messages));
});

test("text metrics count drawing arguments, not image bytes", () => {
	assert.equal(textChars([{ role: "user", content: [image("x".repeat(10000)), { type: "text", text: "hello" }] }]), 5);
	assert.equal(textChars([{ role: "assistant", content: [{ type: "toolCall", arguments: { x: 1 } }] }]), 7);
});

test("an image trim keeps the first image, the newest reference picture and the newest views", () => {
	const img = (n) => ({ type: "image", data: String(n), mimeType: "image/png" });
	const messages = [
		{ role: "user", content: [img(0)] },
		{ role: "toolResult", toolName: "generate_reference", content: [{ type: "text", text: "ref" }, img(1)] },
		...[2, 3, 4, 5].map((n) => ({ role: "toolResult", toolName: "look", content: [img(n)] })),
	];
	const kept = (out) => out.flatMap((m) => m.content.filter((b) => b.type === "image").map((b) => b.data));
	assert.deepEqual(kept(trimImages(messages, 4)), ["0", "1", "4", "5"]);
	assert.deepEqual(kept(trimImages(messages.map((m) => ({ ...m, toolName: "look" })), 4)), ["0", "3", "4", "5"]);
	assert.equal(trimImages(messages, 10), messages);
});
