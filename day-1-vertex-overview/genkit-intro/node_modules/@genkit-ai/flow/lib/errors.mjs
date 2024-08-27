import "./chunk-7OAPEGJQ.mjs";
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
export {
  FlowExecutionError,
  FlowNotFoundError,
  FlowStillRunningError,
  InterruptError,
  getErrorMessage,
  getErrorStack
};
//# sourceMappingURL=errors.mjs.map