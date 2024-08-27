import "./chunk-7OAPEGJQ.mjs";
import { AsyncLocalStorage } from "node:async_hooks";
import { v4 as uuidv4 } from "uuid";
function metadataPrefix(name) {
  return `flow:${name}`;
}
const ctxAsyncLocalStorage = new AsyncLocalStorage();
function getActiveContext() {
  return ctxAsyncLocalStorage.getStore();
}
function runWithActiveContext(ctx, fn) {
  return ctxAsyncLocalStorage.run(ctx, fn);
}
function generateFlowId() {
  return uuidv4();
}
function getFlowAuth() {
  const ctx = getActiveContext();
  if (!ctx) {
    throw new Error("Can only be run from a flow");
  }
  return ctx.auth;
}
export {
  generateFlowId,
  getActiveContext,
  getFlowAuth,
  metadataPrefix,
  runWithActiveContext
};
//# sourceMappingURL=utils.mjs.map