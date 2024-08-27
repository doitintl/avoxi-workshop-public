import {
  __async
} from "../chunk-XEFTB2OF.mjs";
import { Mutex } from "async-mutex";
import crypto from "crypto";
import fs from "fs";
import os from "os";
import path from "path";
import { logger } from "../logging.js";
import {
  TraceDataSchema
} from "./types.js";
const _LocalFileTraceStore = class _LocalFileTraceStore {
  constructor(filters = _LocalFileTraceStore.defaultFilters) {
    this.mutexes = {};
    var _a;
    const rootHash = crypto.createHash("md5").update(((_a = require == null ? void 0 : require.main) == null ? void 0 : _a.filename) || "unknown").digest("hex");
    this.storeRoot = path.resolve(os.tmpdir(), `.genkit/${rootHash}/traces`);
    fs.mkdirSync(this.storeRoot, { recursive: true });
    logger.info(
      `Initialized local file trace store at root: ${this.storeRoot}`
    );
    this.filters = filters;
  }
  load(id) {
    return __async(this, null, function* () {
      const filePath = path.resolve(this.storeRoot, `${id}`);
      if (!fs.existsSync(filePath)) {
        return void 0;
      }
      const data = fs.readFileSync(filePath, "utf8");
      const parsed = JSON.parse(data);
      if (!parsed.traceId) {
        parsed.traceId = id;
      }
      return TraceDataSchema.parse(parsed);
    });
  }
  getMutex(id) {
    if (!this.mutexes[id]) {
      this.mutexes[id] = new Mutex();
    }
    return this.mutexes[id];
  }
  save(id, rawTrace) {
    return __async(this, null, function* () {
      let trace = this.filter(rawTrace);
      if (Object.keys(trace.spans).length === 0) {
        return;
      }
      const mutex = this.getMutex(id);
      yield mutex.waitForUnlock();
      const release = yield mutex.acquire();
      try {
        const existing = yield this.load(id);
        if (existing) {
          Object.keys(trace.spans).forEach(
            (spanId) => existing.spans[spanId] = trace.spans[spanId]
          );
          existing.displayName = trace.displayName;
          existing.startTime = trace.startTime;
          existing.endTime = trace.endTime;
          trace = existing;
        }
        fs.writeFileSync(
          path.resolve(this.storeRoot, `${id}`),
          JSON.stringify(trace)
        );
      } finally {
        release();
      }
    });
  }
  list(query) {
    return __async(this, null, function* () {
      const files = fs.readdirSync(this.storeRoot);
      files.sort((a, b) => {
        return fs.statSync(path.resolve(this.storeRoot, `${b}`)).mtime.getTime() - fs.statSync(path.resolve(this.storeRoot, `${a}`)).mtime.getTime();
      });
      const startFrom = (query == null ? void 0 : query.continuationToken) ? parseInt(query == null ? void 0 : query.continuationToken) : 0;
      const stopAt = startFrom + ((query == null ? void 0 : query.limit) || 10);
      const traces = files.slice(startFrom, stopAt).map((id) => {
        const filePath = path.resolve(this.storeRoot, `${id}`);
        const data = fs.readFileSync(filePath, "utf8");
        const parsed = JSON.parse(data);
        if (!parsed.traceId) {
          parsed.traceId = id;
        }
        return TraceDataSchema.parse(parsed);
      });
      return {
        traces,
        continuationToken: files.length > stopAt ? stopAt.toString() : void 0
      };
    });
  }
  filter(trace) {
    Object.keys(trace.spans).forEach((spanId) => {
      const span = trace.spans[spanId];
      Object.keys(this.filters).forEach((f) => {
        if (span.attributes[f] === this.filters[f]) {
          delete trace.spans[spanId];
        }
      });
    });
    if (Object.keys(trace.spans).length === 1) {
      Object.keys(trace.spans).forEach((spanId) => {
        const span = trace.spans[spanId];
        if (span.attributes["genkit:name"] === "dev-run-action-wrapper") {
          delete trace.spans[spanId];
        }
      });
    }
    return trace;
  }
};
_LocalFileTraceStore.defaultFilters = {
  // Prevent prompt rendering from spamming local trace store
  "genkit:metadata:subtype": "prompt"
};
let LocalFileTraceStore = _LocalFileTraceStore;
export {
  LocalFileTraceStore
};
//# sourceMappingURL=localFileTraceStore.mjs.map