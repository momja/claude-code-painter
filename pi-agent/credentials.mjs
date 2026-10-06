import { chmod, mkdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { homedir } from "node:os";
import { randomUUID } from "node:crypto";

export function credentialFile() {
	return process.env.CONVEYOR_PI_AUTH_FILE || join(homedir(), ".config", "conveyor", "pi-auth.json");
}

async function readAll(file) {
	try {
		const value = JSON.parse(await readFile(file, "utf8"));
		if (!value || typeof value !== "object" || Array.isArray(value)) {
			throw new Error("Pi credential file must contain a JSON object");
		}
		return value;
	} catch (error) {
		if (error?.code === "ENOENT") return {};
		throw error;
	}
}

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export class JsonCredentialStore {
	constructor(file = credentialFile()) {
		this.file = file;
		this.lock = `${file}.lock`;
	}

	async read(providerId, options = {}) {
		options.signal?.throwIfAborted();
		return (await readAll(this.file))[providerId];
	}

	async list(options = {}) {
		options.signal?.throwIfAborted();
		return Object.entries(await readAll(this.file)).map(([providerId, credential]) => ({
			providerId,
			type: credential?.type,
		}));
	}

	async withLock(fn, signal) {
		await mkdir(dirname(this.file), { recursive: true });
		const started = Date.now();
		while (true) {
			signal?.throwIfAborted();
			try {
				await mkdir(this.lock, { mode: 0o700 });
				break;
			} catch (error) {
				if (error?.code !== "EEXIST") throw error;
				const lockStat = await stat(this.lock).catch(() => undefined);
				if (lockStat && Date.now() - lockStat.mtimeMs > 120_000) {
					await rm(this.lock, { recursive: true, force: true });
					continue;
				}
				if (Date.now() - started > 30_000) throw new Error("Timed out waiting for the Pi credential file lock");
				await delay(50);
			}
		}
		try {
			signal?.throwIfAborted();
			return await fn();
		} finally {
			await rm(this.lock, { recursive: true, force: true });
		}
	}

	async save(values) {
		await mkdir(dirname(this.file), { recursive: true });
		const temporary = `${this.file}.${process.pid}.${randomUUID()}.tmp`;
		try {
			await writeFile(temporary, `${JSON.stringify(values, null, 2)}\n`, { mode: 0o600 });
			await rename(temporary, this.file);
			await chmodPrivate(this.file);
		} finally {
			await rm(temporary, { force: true });
		}
	}

	modify(providerId, fn, options = {}) {
		return this.withLock(async () => {
			const values = await readAll(this.file);
			const current = values[providerId];
			const next = await fn(current);
			options.signal?.throwIfAborted();
			if (next !== undefined) {
				values[providerId] = next;
				await this.save(values);
			}
			return next ?? current;
		}, options.signal);
	}

	delete(providerId, options = {}) {
		return this.withLock(async () => {
			const values = await readAll(this.file);
			if (Object.hasOwn(values, providerId)) {
				delete values[providerId];
				await this.save(values);
			}
		}, options.signal);
	}
}

async function chmodPrivate(file) {
	await chmod(file, 0o600);
}
