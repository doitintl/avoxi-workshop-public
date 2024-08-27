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
var localFileFlowStateStore_exports = {};
__export(localFileFlowStateStore_exports, {
  LocalFileFlowStateStore: () => LocalFileFlowStateStore
});
module.exports = __toCommonJS(localFileFlowStateStore_exports);
var import_crypto = __toESM(require("crypto"));
var import_fs = __toESM(require("fs"));
var import_os = __toESM(require("os"));
var import_path = __toESM(require("path"));
var import_flowTypes = require("./flowTypes.js");
var import_logging = require("./logging.js");
class LocalFileFlowStateStore {
  constructor() {
    var _a;
    const rootHash = import_crypto.default.createHash("md5").update(((_a = require == null ? void 0 : require.main) == null ? void 0 : _a.filename) || "unknown").digest("hex");
    this.storeRoot = import_path.default.resolve(import_os.default.tmpdir(), `.genkit/${rootHash}/flows`);
    import_fs.default.mkdirSync(this.storeRoot, { recursive: true });
    import_logging.logger.info("Using DevFlowStateStore. Root: " + this.storeRoot);
  }
  load(id) {
    return __async(this, null, function* () {
      const filePath = import_path.default.resolve(this.storeRoot, `${id}`);
      if (!import_fs.default.existsSync(filePath)) {
        return void 0;
      }
      const data = import_fs.default.readFileSync(filePath, "utf8");
      return import_flowTypes.FlowStateSchema.parse(JSON.parse(data));
    });
  }
  save(id, state) {
    return __async(this, null, function* () {
      import_logging.logger.debug("save flow state " + id);
      import_fs.default.writeFileSync(
        import_path.default.resolve(this.storeRoot, `${id}`),
        JSON.stringify(state)
      );
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
      const flowStates = files.slice(startFrom, stopAt).map((id) => {
        const filePath = import_path.default.resolve(this.storeRoot, `${id}`);
        const data = import_fs.default.readFileSync(filePath, "utf8");
        return import_flowTypes.FlowStateSchema.parse(JSON.parse(data));
      });
      return {
        flowStates,
        continuationToken: files.length > stopAt ? stopAt.toString() : void 0
      };
    });
  }
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  LocalFileFlowStateStore
});
//# sourceMappingURL=localFileFlowStateStore.js.map