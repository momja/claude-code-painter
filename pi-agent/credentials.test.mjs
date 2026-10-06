import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { createModels } from "@earendil-works/pi-ai";
import { openaiCodexProvider } from "@earendil-works/pi-ai/providers/openai-codex";
import { JsonCredentialStore } from "./credentials.mjs";

test("credential writes are atomic, private, and serialized", async () => {
	const dir = await mkdtemp(join(tmpdir(), "conveyor-pi-auth-"));
	const file = join(dir, "pi-auth.json");
	try {
		const store = new JsonCredentialStore(file);
		assert.equal(await store.read("openai-codex"), undefined);
		await store.modify("openai-codex", async () => ({ type: "oauth", access: "a", refresh: "r", expires: 1 }));
		await Promise.all(Array.from({ length: 8 }, () => store.modify("counter", async (current) => ({
			type: "api_key",
			key: String(Number(current?.key || 0) + 1),
		}))));

		assert.equal((await store.read("counter")).key, "8");
		assert.equal((await store.read("openai-codex")).refresh, "r");
		assert.deepEqual(await store.list(), [
			{ providerId: "openai-codex", type: "oauth" },
			{ providerId: "counter", type: "api_key" },
		]);
		assert.equal((await stat(file)).mode & 0o777, 0o600);
		assert.equal(JSON.parse(await readFile(file, "utf8"))["openai-codex"].access, "a");

		await store.delete("openai-codex");
		assert.equal(await store.read("openai-codex"), undefined);
	} finally {
		await rm(dir, { recursive: true, force: true });
	}
});

test("OpenAI Codex resolves the saved ChatGPT OAuth credential", async () => {
	const dir = await mkdtemp(join(tmpdir(), "conveyor-codex-provider-"));
	try {
		const credentials = new JsonCredentialStore(join(dir, "pi-auth.json"));
		await credentials.modify("openai-codex", async () => ({
			type: "oauth",
			access: "access-token",
			refresh: "refresh-token",
			expires: Date.now() + 60 * 60 * 1000,
			accountId: "account-id",
		}));
		const models = createModels({ credentials });
		models.setProvider(openaiCodexProvider());
		const model = models.getModel("openai-codex", "gpt-5.4");
		assert.ok(model);
		assert.equal(model.api, "openai-codex-responses");
		assert.equal((await models.checkAuth("openai-codex")).type, "oauth");
		assert.equal((await models.getAuth(model)).auth.apiKey, "access-token");
	} finally {
		await rm(dir, { recursive: true, force: true });
	}
});
