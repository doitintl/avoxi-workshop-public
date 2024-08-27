"use strict";
var __defProp = Object.defineProperty;
var __defProps = Object.defineProperties;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropDescs = Object.getOwnPropertyDescriptors;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getOwnPropSymbols = Object.getOwnPropertySymbols;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __propIsEnum = Object.prototype.propertyIsEnumerable;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __spreadValues = (a, b) => {
  for (var prop in b || (b = {}))
    if (__hasOwnProp.call(b, prop))
      __defNormalProp(a, prop, b[prop]);
  if (__getOwnPropSymbols)
    for (var prop of __getOwnPropSymbols(b)) {
      if (__propIsEnum.call(b, prop))
        __defNormalProp(a, prop, b[prop]);
    }
  return a;
};
var __spreadProps = (a, b) => __defProps(a, __getOwnPropDescs(b));
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
var experimental_exports = {};
__export(experimental_exports, {
  asyncSleep: () => asyncSleep,
  durableFlow: () => durableFlow,
  getFlowState: () => getFlowState,
  interrupt: () => interrupt,
  resumeFlow: () => resumeFlow,
  run: () => run,
  runAction: () => runAction,
  scheduleFlow: () => scheduleFlow,
  sleep: () => sleep,
  waitFlowToComplete: () => waitFlowToComplete,
  waitFor: () => waitFor
});
module.exports = __toCommonJS(experimental_exports);
var import_logging = require("@genkit-ai/core/logging");
var import_errors = require("./errors.js");
var import_flow = require("./flow.js");
var import_utils = require("./utils.js");
function durableFlow(config, steps) {
  return (0, import_flow.defineFlow)(
    {
      name: config.name,
      inputSchema: config.inputSchema,
      outputSchema: config.outputSchema,
      streamSchema: config.streamSchema,
      invoker: config.invoker,
      experimentalScheduler: config.scheduler,
      experimentalDurable: true
    },
    steps
  );
}
function scheduleFlow(flow, payload, delaySeconds) {
  return __async(this, null, function* () {
    if (!(flow instanceof import_flow.Flow)) {
      flow = flow.flow;
    }
    const state = yield flow.invoker(flow, {
      schedule: {
        input: flow.inputSchema ? flow.inputSchema.parse(payload) : payload,
        delay: delaySeconds
      }
    });
    return state;
  });
}
function resumeFlow(flow, flowId, payload) {
  return __async(this, null, function* () {
    if (!(flow instanceof import_flow.Flow)) {
      flow = flow.flow;
    }
    return yield flow.invoker(flow, {
      resume: {
        flowId,
        payload
      }
    });
  });
}
function getFlowState(flow, flowId) {
  return __async(this, null, function* () {
    if (!(flow instanceof import_flow.Flow)) {
      flow = flow.flow;
    }
    if (!flow.stateStore) {
      throw new Error("Flow state must be configured.");
    }
    const state = yield (yield flow.stateStore()).load(flowId);
    if (!state) {
      throw new import_errors.FlowNotFoundError(`flow state ${flowId} not found`);
    }
    const op = __spreadValues({}, state.operation);
    if (state.blockedOnStep) {
      op.blockedOnStep = state.blockedOnStep;
    }
    return op;
  });
}
function runAction(action, input, actionConfig) {
  const config = __spreadProps(__spreadValues({}, actionConfig), {
    name: (actionConfig == null ? void 0 : actionConfig.name) || action.__action.name
  });
  return run(config, input, () => action(input));
}
function waitFlowToComplete(flow, flowId) {
  return __async(this, null, function* () {
    if (!(flow instanceof import_flow.Flow)) {
      flow = flow.flow;
    }
    let state = void 0;
    try {
      state = yield getFlowState(flow, flowId);
    } catch (e) {
      import_logging.logger.error(e);
      if (!(e instanceof import_errors.FlowNotFoundError)) {
        throw e;
      }
    }
    if (state && (state == null ? void 0 : state.done)) {
      return parseOutput(flowId, state);
    } else {
      yield asyncSleep(1e3);
      return yield waitFlowToComplete(flow, flowId);
    }
  });
}
function parseOutput(flowId, state) {
  var _a, _b;
  if (!state.done) {
    throw new import_errors.FlowStillRunningError(flowId);
  }
  if ((_a = state.result) == null ? void 0 : _a.error) {
    throw new import_errors.FlowExecutionError(
      flowId,
      state.result.error,
      state.result.stacktrace
    );
  }
  return (_b = state.result) == null ? void 0 : _b.response;
}
function run(nameOrConfig, funcOrInput, fn) {
  let config;
  if (typeof nameOrConfig === "string") {
    config = {
      name: nameOrConfig
    };
  } else {
    config = nameOrConfig;
  }
  const func = arguments.length === 3 ? fn : funcOrInput;
  const input = arguments.length === 3 ? funcOrInput : void 0;
  if (!func) {
    throw new Error("unable to resolve run function");
  }
  const ctx = (0, import_utils.getActiveContext)();
  if (!ctx)
    throw new Error("can only be run from a flow");
  return ctx.run(config, input, func);
}
function interrupt(stepName, responseSchema, func) {
  const ctx = (0, import_utils.getActiveContext)();
  if (!ctx)
    throw new Error("interrupt can only be run from a flow");
  return ctx.interrupt(
    stepName,
    func || ((input) => input),
    responseSchema
  );
}
function sleep(actionId, durationMs) {
  const ctx = (0, import_utils.getActiveContext)();
  if (!ctx)
    throw new Error("sleep can only be run from a flow");
  return ctx.sleep(actionId, durationMs);
}
function waitFor(stepName, flow, flowIds, pollingConfig) {
  const ctx = (0, import_utils.getActiveContext)();
  if (!ctx)
    throw new Error("waitFor can only be run from a flow");
  return ctx.waitFor({ flow, stepName, flowIds, pollingConfig });
}
function asyncSleep(duration) {
  return __async(this, null, function* () {
    return new Promise((resolve) => {
      setTimeout(resolve, duration);
    });
  });
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  asyncSleep,
  durableFlow,
  getFlowState,
  interrupt,
  resumeFlow,
  run,
  runAction,
  scheduleFlow,
  sleep,
  waitFlowToComplete,
  waitFor
});
//# sourceMappingURL=experimental.js.map