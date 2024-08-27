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
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);
var utils_exports = {};
__export(utils_exports, {
  generateFlowId: () => generateFlowId,
  getActiveContext: () => getActiveContext,
  getFlowAuth: () => getFlowAuth,
  metadataPrefix: () => metadataPrefix,
  runWithActiveContext: () => runWithActiveContext
});
module.exports = __toCommonJS(utils_exports);
var import_node_async_hooks = require("node:async_hooks");
var import_uuid = require("uuid");
function metadataPrefix(name) {
  return `flow:${name}`;
}
const ctxAsyncLocalStorage = new import_node_async_hooks.AsyncLocalStorage();
function getActiveContext() {
  return ctxAsyncLocalStorage.getStore();
}
function runWithActiveContext(ctx, fn) {
  return ctxAsyncLocalStorage.run(ctx, fn);
}
function generateFlowId() {
  return (0, import_uuid.v4)();
}
function getFlowAuth() {
  const ctx = getActiveContext();
  if (!ctx) {
    throw new Error("Can only be run from a flow");
  }
  return ctx.auth;
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  generateFlowId,
  getActiveContext,
  getFlowAuth,
  metadataPrefix,
  runWithActiveContext
});
//# sourceMappingURL=utils.js.map