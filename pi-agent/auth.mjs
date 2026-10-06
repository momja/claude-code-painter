import { createModels } from "@earendil-works/pi-ai";
import { openaiCodexProvider } from "@earendil-works/pi-ai/providers/openai-codex";
import { JsonCredentialStore, credentialFile } from "./credentials.mjs";

const providerId = "openai-codex";
const action = process.argv[2] || "login";
const credentials = new JsonCredentialStore();

async function main() {
	if (action === "logout") {
		await credentials.delete(providerId);
		console.log("OpenAI Codex login removed.");
		return;
	}
	if (action !== "login") throw new Error(`Unknown auth action: ${action}`);

	const models = createModels({ credentials });
	models.setProvider(openaiCodexProvider());
	const controller = new AbortController();
	const onInterrupt = () => controller.abort(new Error("Login cancelled"));
	process.once("SIGINT", onInterrupt);
	try {
		await models.login(providerId, "oauth", {
			signal: controller.signal,
			prompt: async (prompt) => {
				if (prompt.type === "select") return "device_code";
				throw new Error(`Unexpected OpenAI sign-in prompt: ${prompt.type}`);
			},
			notify: (event) => {
				if (event.type === "device_code") {
					console.log(`\nOpen ${event.verificationUri} and enter this code:\n\n  ${event.userCode}\n`);
					console.log("Waiting for OpenAI to confirm sign-in. Press Ctrl+C to cancel.");
				} else if (event.type === "auth_url") {
					console.log(`Open this URL to sign in:\n${event.url}`);
				} else if (event.message) {
					console.log(event.message);
				}
			},
		});
		console.log(`OpenAI Codex subscription linked. Credentials saved to ${credentialFile()}`);
	} finally {
		process.off("SIGINT", onInterrupt);
	}
}

main().catch((error) => {
	console.error(`OpenAI login failed: ${error?.message || error}`);
	process.exitCode = 1;
});
