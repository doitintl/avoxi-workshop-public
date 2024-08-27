"use strict";
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
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
var __reExport = (target, mod, secondTarget) => (__copyProps(target, mod, "default"), secondTarget && __copyProps(secondTarget, mod, "default"));
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);
var src_exports = {};
__export(src_exports, {
  GENKIT_CLIENT_HEADER: () => GENKIT_CLIENT_HEADER,
  GENKIT_VERSION: () => GENKIT_VERSION,
  GenkitError: () => import_error.GenkitError,
  defineJsonSchema: () => import_schema.defineJsonSchema,
  defineSchema: () => import_schema.defineSchema
});
module.exports = __toCommonJS(src_exports);
var import_version = require("./__codegen/version.js");
__reExport(src_exports, require("./action.js"), module.exports);
__reExport(src_exports, require("./config.js"), module.exports);
var import_error = require("./error.js");
__reExport(src_exports, require("./flowTypes.js"), module.exports);
var import_schema = require("./schema.js");
__reExport(src_exports, require("./telemetryTypes.js"), module.exports);
const GENKIT_VERSION = import_version.version;
const GENKIT_CLIENT_HEADER = `genkit-node/${GENKIT_VERSION} gl-node/${process.versions.node}`;
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  GENKIT_CLIENT_HEADER,
  GENKIT_VERSION,
  GenkitError,
  defineJsonSchema,
  defineSchema,
  ...require("./action.js"),
  ...require("./config.js"),
  ...require("./flowTypes.js"),
  ...require("./telemetryTypes.js")
});
//# sourceMappingURL=index.js.map