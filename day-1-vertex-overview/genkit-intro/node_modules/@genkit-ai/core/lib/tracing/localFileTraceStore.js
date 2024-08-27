"use strict";
var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
  // If the importer is in node compatibility mode or this is not an ESM
  // file that has been converted to a CommonJS file using a Babel-
  // compatible transform (i.e. "__esModule" has not been set), then set
  // "default" to the CommonJS "module.exports" for node compatibility.
  isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
  mod
));
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);
var __async = (__this, __arguments, generator) => {
  return new Promise((resolve, reject) => {
    var fulfilled = (value) => {
      try {
        step(generator.next(value));
      } catch (e) {
        reject(e);
      }
    };
    var rejected = (value) => {
      try {
        step(generator.throw(value));
      } catch (e) {
        reject(e);
      }
    };
    var step = (x) => x.done ? resolve(x.value) : Promise.resolve(x.value).then(fulfilled, rejected);
    step((generator = generator.apply(__this, __arguments)).next());
  });
};
var localFileTraceStore_exports = {};
__export(localFileTraceStore_exports, {
  LocalFileTraceStore: () => LocalFileTraceStore
});
module.exports = __toCommonJS(localFileTraceStore_exports);
var import_async_mutex = require("async-mutex");
var import_crypto = __toESM(require("crypto"));
var import_fs = __toESM(require("fs"));
var import_os = __toESM(require("os"));
var import_path = __toESM(require("path"));
var import_logging = require("../logging.js");
var import_types = require("./types.js");
const _LocalFileTraceStore = class _LocalFileTraceStore {
  constructor(filters = _LocalFileTraceStore.defaultFilters) {
    this.mutexes = {};
    var _a;
    const rootHash = import_crypto.default.createHash("md5").update(((_a = require == null ? void 0 : require.main) == null ? void 0 : _a.filename) || "unknown").digest("hex");
    this.storeRoot = import_path.default.resolve(import_os.default.tmpdir(), `.genkit/${rootHash}/traces`);
    import_fs.default.mkdirSync(this.storeRoot, { recursive: true });
    import_logging.logger.info(
      `Initialized local file trace store at root: ${this.storeRoot}`
    );
    this.filters = filters;
  }
  load(id) {
    return __async(this, null, function* () {
      const filePath = import_path.default.resolve(this.storeRoot, `${id}`);
      if (!import_fs.default.existsSync(filePath)) {
        return void 0;
      }
      const data = import_fs.default.readFileSync(filePath, "utf8");
      const parsed = JSON.parse(data);
      if (!parsed.traceId) {
        parsed.traceId = id;
      }
      return import_types.TraceDataSchema.parse(parsed);
    });
  }
  getMutex(id) {
    if (!this.mutexes[id]) {
      this.mutexes[id] = new import_async_mutex.Mutex();
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
        import_fs.default.writeFileSync(
          import_path.default.resolve(this.storeRoot, `${id}`),
          JSON.stringify(trace)
        );
      } finally {
        release();
      }
    });
  }
  list(query) {
    return __async(this, null, function* () {
      const files = import_fs.default.readdirSync(this.storeRoot);
      files.sort((a, b) => {
        return import_fs.default.statSync(import_path.default.resolve(this.storeRoot, `${b}`)).mtime.getTime() - import_fs.default.statSync(import_path.default.resolve(this.storeRoot, `${a}`)).mtime.getTime();
      });
      const startFrom = (query == null ? void 0 : query.continuationToken) ? parseInt(query == null ? void 0 : query.continuationToken) : 0;
      const stopAt = startFrom + ((query == null ? void 0 : query.limit) || 10);
      const traces = files.slice(startFrom, stopAt).map((id) => {
        const filePath = import_path.default.resolve(this.storeRoot, `${id}`);
        const data = import_fs.default.readFileSync(filePath, "utf8");
        const parsed = JSON.parse(data);
        if (!parsed.traceId) {
          parsed.traceId = id;
        }
        return import_types.TraceDataSchema.parse(parsed);
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
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  LocalFileTraceStore
});
//# sourceMappingURL=localFileTraceStore.js.map