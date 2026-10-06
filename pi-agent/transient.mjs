// Errors that mean the connection died, not that the model or the request was wrong. pi-ai retries these only
// before a Codex WebSocket delivers its first event; after that it rethrows, so the sidecar retries the turn.
const TRANSIENT = /WebSocket (closed|error|idle timeout|connect timeout)|ECONNRESET|ETIMEDOUT|EPIPE|socket hang up|fetch failed|terminated|servers are currently overloaded/i;

export const MAX_TRANSPORT_RETRIES = 5;

export function isTransientTransportError(message) {
	return typeof message === "string" && TRANSIENT.test(message);
}

// Backoff before retry number `attempt` (1-based): 2s, 4s, 8s, 16s, capped at 30s.
export function retryDelayMs(attempt) {
	return Math.min(30_000, 1000 * 2 ** attempt);
}
