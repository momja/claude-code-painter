import assert from "node:assert/strict";
import test from "node:test";
import { isTransientTransportError, retryDelayMs } from "./transient.mjs";

test("connection drops are transient", () => {
	for (const m of ["WebSocket closed 1006", "WebSocket closed 1012 service restart", "WebSocket idle timeout after 60000ms", "read ECONNRESET", "fetch failed"]) {
		assert.equal(isTransientTransportError(m), true, m);
	}
});

test("request and quota errors are not", () => {
	for (const m of ["400: bad request", "You have hit your ChatGPT usage limit (plus plan).", "Invalid Codex WebSocket JSON: x", "aborted", "", null, undefined]) {
		assert.equal(isTransientTransportError(m), false, String(m));
	}
});

test("backoff doubles and caps", () => {
	assert.deepEqual([1, 2, 3, 4, 5, 6].map(retryDelayMs), [2000, 4000, 8000, 16000, 30000, 30000]);
});
