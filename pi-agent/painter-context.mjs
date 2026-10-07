// Model-facing history only. The agent and the database retain the full transcript.

export function textChars(messages) {
	return messages.reduce((total, message) => {
		if (typeof message.content === "string") return total + message.content.length;
		return total + (message.content || []).reduce((n, block) => {
			if (block.type === "image") return n;
			return n + (block.text || block.thinking || "").length +
				(block.type === "toolCall" ? JSON.stringify(block.arguments || {}).length : 0);
		}, 0);
	}, 0);
}

export function boundedPainterContext(messages, { keepTurns, state, onTrim }) {
	// Without authoritative state it is safer to keep history than to guess the pen or scope.
	if (!keepTurns || !state) return messages;
	const starts = [];
	for (let i = 1; i < messages.length; i++) if (messages[i].role === "assistant") starts.push(i);
	if (starts.length <= keepTurns) return messages;

	let tail = starts[starts.length - keepTurns];
	// A length-recovery nudge immediately before the retained turn belongs with that turn.
	while (tail > 1 && messages[tail - 1].role === "user") tail -= 1;
	let latestView = -1;
	for (let i = messages.length - 1; i > 0; i--) {
		if (messages[i].role === "toolResult" && messages[i].content?.some((b) => b.type === "image")) {
			latestView = i;
			break;
		}
	}
	// Keep the entire turn containing the newest view. Never orphan a result or split a tool-call group.
	const viewStart = starts.findLast((i) => i <= latestView) ?? -1;
	const viewEnd = starts.find((i) => i > latestView) ?? messages.length;
	const retained = [];
	let dropped = 0;
	for (let i = 0; i < messages.length; i++) {
		const message = messages[i];
		if (i !== 0 && i < tail && !(i >= viewStart && viewStart >= 0 && i < viewEnd)) {
			dropped += 1;
			continue;
		}
		// Older images in the retained turns are stale. Keep the target/demo and the newest view result.
		if (i !== 0 && i !== latestView && Array.isArray(message.content)) {
			const content = message.content.map((block) => block.type === "image"
				? { type: "text", text: "[older canvas image omitted]" } : block);
			retained.push({ ...message, content });
		} else retained.push(message);
	}
	retained.push({
		role: "user",
		timestamp: messages.at(-1).timestamp,
		content: "Current painting state AFTER the retained history. Do not replay old calls. " +
			"The plan is your saved working note; counters, pen and scope come from the server. " +
			"The latest canvas view may predate later paint calls.\n" + JSON.stringify({
				...state,
				view_actions_used: latestView < 0 ? null : messages[latestView].details?.paintingState?.actions_used ?? null,
			}),
	});
	onTrim?.({ dropped, kept: retained.length, before_chars: textChars(messages), after_chars: textChars(retained) });
	return retained;
}

// GLM answers 400 too_many_images above 8 per request, and a conversation keeps every image it has seen. Keep
// the first image (the target, or a canvas agent's first view), the newest reference picture the agent made on
// the canvas, which it is painting from, and the newest of the rest; older canvas views are stale anyway.
export function trimImages(messages, limit) {
	if (!limit) return messages;
	const positions = [];
	messages.forEach((msg, i) => {
		if (!Array.isArray(msg.content)) return;
		const reference = msg.role === "toolResult" && msg.toolName === "generate_reference";
		msg.content.forEach((block, j) => {
			if (block.type === "image") positions.push({ key: `${i}:${j}`, i, j, reference });
		});
	});
	if (positions.length <= limit) return messages;
	const keep = new Set([positions[0].key]);
	const reference = positions.findLast((p) => p.reference);
	if (reference) keep.add(reference.key);
	for (let n = positions.length - 1; n >= 0 && keep.size < limit; n--) keep.add(positions[n].key);
	const trimmed = messages.slice();
	const touched = new Map();
	for (const { key, i, j } of positions) {
		if (keep.has(key)) continue;
		const content = (touched.get(i) || trimmed[i].content).slice();
		content[j] = { type: "text", text: "[older image dropped: the provider limits images per request]" };
		touched.set(i, content);
	}
	for (const [i, content] of touched) trimmed[i] = { ...trimmed[i], content };
	return trimmed;
}
