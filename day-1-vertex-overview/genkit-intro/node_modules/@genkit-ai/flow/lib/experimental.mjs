import {
  __async,
  __spreadProps,
  __spreadValues
} from "./chunk-7OAPEGJQ.mjs";
import { logger } from "@genkit-ai/core/logging";
import {
  FlowExecutionError,
  FlowNotFoundError,
  FlowStillRunningError
} from "./errors.js";
import {
  Flow,
  defineFlow
} from "./flow.js";
import { getActiveContext } from "./utils.js";
function durableFlow(config, steps) {
  return defineFlow(
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
    if (!(flow instanceof Flow)) {
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
    if (!(flow instanceof Flow)) {
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
    if (!(flow instanceof Flow)) {
      flow = flow.flow;
    }
    if (!flow.stateStore) {
      throw new Error("Flow state must be configured.");
    }
    const state = yield (yield flow.stateStore()).load(flowId);
    if (!state) {
      throw new FlowNotFoundError(`flow state ${flowId} not found`);
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
    if (!(flow instanceof Flow)) {
      flow = flow.flow;
    }
    let state = void 0;
    try {
      state = yield getFlowState(flow, flowId);
    } catch (e) {
      logger.error(e);
      if (!(e instanceof FlowNotFoundError)) {
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
    throw new FlowStillRunningError(flowId);
  }
  if ((_a = state.result) == null ? void 0 : _a.error) {
    throw new FlowExecutionError(
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
  const ctx = getActiveContext();
  if (!ctx)
    throw new Error("can only be run from a flow");
  return ctx.run(config, input, func);
}
function interrupt(stepName, responseSchema, func) {
  const ctx = getActiveContext();
  if (!ctx)
    throw new Error("interrupt can only be run from a flow");
  return ctx.interrupt(
    stepName,
    func || ((input) => input),
    responseSchema
  );
}
function sleep(actionId, durationMs) {
  const ctx = getActiveContext();
  if (!ctx)
    throw new Error("sleep can only be run from a flow");
  return ctx.sleep(actionId, durationMs);
}
function waitFor(stepName, flow, flowIds, pollingConfig) {
  const ctx = getActiveContext();
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
export {
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
};
//# sourceMappingURL=experimental.mjs.map