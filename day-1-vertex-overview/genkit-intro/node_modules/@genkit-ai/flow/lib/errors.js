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
var errors_exports = {};
__export(errors_exports, {
  FlowExecutionError: () => FlowExecutionError,
  FlowNotFoundError: () => FlowNotFoundError,
  FlowStillRunningError: () => FlowStillRunningError,
  InterruptError: () => InterruptError,
  getErrorMessage: () => getErrorMessage,
  getErrorStack: () => getErrorStack
});
module.exports = __toCommonJS(errors_exports);
class InterruptError extends Error {
}
function getErrorMessage(e) {
  if (e instanceof Error) {
    return e.message;
  }
  return `${e}`;
}
function getErrorStack(e) {
  if (e instanceof Error) {
    return e.stack;
  }
  return void 0;
}
class FlowNotFoundError extends Error {
  constructor(msg) {
    super(msg);
  }
}
class FlowStillRunningError extends Error {
  constructor(flowId) {
    super(
      `flow ${flowId} is not done execution. Consider using waitForFlowToComplete to wait for completion before calling getOutput.`
    );
    this.flowId = flowId;
  }
}
class FlowExecutionError extends Error {
  constructor(flowId, originalMessage, originalStacktrace) {
    super(originalMessage);
    this.flowId = flowId;
    this.originalMessage = originalMessage;
    this.originalStacktrace = originalStacktrace;
    this.stack = originalStacktrace;
  }
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  FlowExecutionError,
  FlowNotFoundError,
  FlowStillRunningError,
  InterruptError,
  getErrorMessage,
  getErrorStack
});
//# sourceMappingURL=errors.js.map