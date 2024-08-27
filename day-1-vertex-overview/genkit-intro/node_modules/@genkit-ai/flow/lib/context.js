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
var context_exports = {};
__export(context_exports, {
  Context: () => Context
});
module.exports = __toCommonJS(context_exports);
var import_schema = require("@genkit-ai/core/schema");
var import_tracing = require("@genkit-ai/core/tracing");
var import_v1 = require("firebase-functions/v1");
var import_errors = require("./errors.js");
var import_utils = require("./utils.js");
class Context {
  constructor(flow, flowId, state, auth) {
    this.flow = flow;
    this.flowId = flowId;
    this.state = state;
    this.auth = auth;
    this.seenSteps = {};
  }
  isCached(stepName) {
    return this.state.cache.hasOwnProperty(stepName);
  }
  getCached(stepName) {
    return this.state.cache[stepName].value;
  }
  updateCachedValue(stepName, value) {
    this.state.cache[stepName] = value ? { value } : { empty: true };
  }
  memoize(stepName, func) {
    return __async(this, null, function* () {
      if (this.isCached(stepName)) {
        return [this.getCached(stepName), true];
      }
      const value = yield func();
      this.updateCachedValue(stepName, value);
      return [value, false];
    });
  }
  saveState() {
    return __async(this, null, function* () {
      if (this.flow.stateStore) {
        yield (yield this.flow.stateStore()).save(this.flowId, this.state);
      }
    });
  }
  // Runs provided function in the current context. The config can specify retry and other behaviors.
  run(config, input, func) {
    return __async(this, null, function* () {
      return yield (0, import_tracing.runInNewSpan)(
        {
          metadata: {
            name: config.name
          },
          labels: {
            [import_tracing.SPAN_TYPE_ATTR]: "flowStep"
          }
        },
        (metadata, _, isRoot) => __async(this, null, function* () {
          const stepName = this.resolveStepName(config.name);
          (0, import_tracing.setCustomMetadataAttributes)({
            [(0, import_utils.metadataPrefix)("stepType")]: "run",
            [(0, import_utils.metadataPrefix)("stepName")]: config.name,
            [(0, import_utils.metadataPrefix)("resolvedStepName")]: stepName
          });
          if (input !== void 0) {
            metadata.input = input;
          }
          const [value, wasCached] = isRoot ? yield this.memoize(stepName, func) : [yield func(), false];
          if (wasCached) {
            (0, import_tracing.setCustomMetadataAttribute)((0, import_utils.metadataPrefix)("state"), "cached");
          } else {
            (0, import_tracing.setCustomMetadataAttribute)((0, import_utils.metadataPrefix)("state"), "run");
            if (value !== void 0) {
              metadata.output = JSON.stringify(value);
            }
          }
          return value;
        })
      );
    });
  }
  resolveStepName(name) {
    if (this.seenSteps[name] !== void 0) {
      this.seenSteps[name]++;
      name += `-${this.seenSteps[name]}`;
    } else {
      this.seenSteps[name] = 0;
    }
    return name;
  }
  // Executes interrupt step in the current context.
  interrupt(stepName, func, responseSchema, skipCache) {
    return __async(this, null, function* () {
      return yield (0, import_tracing.runInNewSpan)(
        {
          metadata: {
            name: stepName
          },
          labels: {
            [import_tracing.SPAN_TYPE_ATTR]: "flowStep"
          }
        },
        (metadata) => __async(this, null, function* () {
          const resolvedStepName = this.resolveStepName(stepName);
          (0, import_tracing.setCustomMetadataAttributes)({
            [(0, import_utils.metadataPrefix)("stepType")]: "interrupt",
            [(0, import_utils.metadataPrefix)("stepName")]: stepName,
            [(0, import_utils.metadataPrefix)("resolvedStepName")]: resolvedStepName
          });
          if (!skipCache && this.isCached(resolvedStepName)) {
            (0, import_tracing.setCustomMetadataAttribute)((0, import_utils.metadataPrefix)("state"), "skipped");
            return this.getCached(resolvedStepName);
          }
          if (this.state.eventsTriggered.hasOwnProperty(resolvedStepName)) {
            let value;
            try {
              value = yield func(
                this.state.eventsTriggered[resolvedStepName]
              );
            } catch (e) {
              if (e instanceof import_errors.InterruptError) {
                (0, import_tracing.setCustomMetadataAttribute)((0, import_utils.metadataPrefix)("state"), "interrupt");
              } else {
                (0, import_tracing.setCustomMetadataAttribute)((0, import_utils.metadataPrefix)("state"), "error");
              }
              throw e;
            }
            this.state.blockedOnStep = null;
            if (!skipCache) {
              this.updateCachedValue(resolvedStepName, value);
            }
            (0, import_tracing.setCustomMetadataAttribute)((0, import_utils.metadataPrefix)("state"), "dispatch");
            if (value !== void 0) {
              metadata.output = JSON.stringify(value);
            }
            return value;
          }
          import_v1.logger.debug("blockedOnStep", resolvedStepName);
          this.state.blockedOnStep = { name: resolvedStepName };
          if (responseSchema) {
            this.state.blockedOnStep.schema = JSON.stringify(
              (0, import_schema.toJsonSchema)({ schema: responseSchema })
            );
          }
          (0, import_tracing.setCustomMetadataAttribute)((0, import_utils.metadataPrefix)("state"), "interrupted");
          throw new import_errors.InterruptError();
        })
      );
    });
  }
  // Sleep for the specified number of seconds.
  sleep(stepName, seconds) {
    return __async(this, null, function* () {
      const resolvedStepName = this.resolveStepName(stepName);
      if (this.isCached(resolvedStepName)) {
        (0, import_tracing.setCustomMetadataAttribute)((0, import_utils.metadataPrefix)("state"), "skipped");
        return this.getCached(resolvedStepName);
      }
      yield this.flow.scheduler(
        this.flow,
        {
          runScheduled: {
            flowId: this.flowId
          }
        },
        seconds
      );
      this.updateCachedValue(resolvedStepName, void 0);
      return this.interrupt(
        stepName,
        (input) => input,
        null
      );
    });
  }
  /**
   * Wait for the provided flow to complete execution. This will do a poll.
   * Poll will be done with an exponential backoff (configurable).
   */
  waitFor(opts) {
    return __async(this, null, function* () {
      var _a;
      const resolvedStepName = this.resolveStepName(opts.stepName);
      if (this.isCached(resolvedStepName)) {
        return this.getCached(resolvedStepName);
      }
      const states = yield this.getFlowsOperations(opts.flow, opts.flowIds);
      if (states.includes(void 0)) {
        throw new Error(
          "Unable to resolve flow state for " + opts.flowIds[states.indexOf(void 0)]
        );
      }
      const ops = states.map((s) => s.operation);
      if (ops.map((op) => op.done).reduce((a, b) => a && b)) {
        this.updateCachedValue(resolvedStepName, states);
        return ops;
      }
      yield this.flow.scheduler(
        this.flow,
        {
          runScheduled: {
            flowId: this.flowId
          }
        },
        ((_a = opts.pollingConfig) == null ? void 0 : _a.interval) || 5
      );
      throw new import_errors.InterruptError();
    });
  }
  getFlowsOperations(flow, flowIds) {
    return __async(this, null, function* () {
      return yield Promise.all(
        flowIds.map((id) => __async(this, null, function* () {
          if (!flow.stateStore) {
            throw new Error("Flow state store must be configured");
          }
          return (yield flow.stateStore()).load(id);
        }))
      );
    });
  }
  /**
   * Returns current active execution state.
   */
  getCurrentExecution() {
    return this.state.executions[this.state.executions.length - 1];
  }
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  Context
});
//# sourceMappingURL=context.js.map