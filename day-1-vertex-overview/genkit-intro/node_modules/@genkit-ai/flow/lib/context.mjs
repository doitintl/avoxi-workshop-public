import {
  __async
} from "./chunk-7OAPEGJQ.mjs";
import { toJsonSchema } from "@genkit-ai/core/schema";
import {
  SPAN_TYPE_ATTR,
  runInNewSpan,
  setCustomMetadataAttribute,
  setCustomMetadataAttributes
} from "@genkit-ai/core/tracing";
import { logger } from "firebase-functions/v1";
import { InterruptError } from "./errors.js";
import { metadataPrefix } from "./utils.js";
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
      return yield runInNewSpan(
        {
          metadata: {
            name: config.name
          },
          labels: {
            [SPAN_TYPE_ATTR]: "flowStep"
          }
        },
        (metadata, _, isRoot) => __async(this, null, function* () {
          const stepName = this.resolveStepName(config.name);
          setCustomMetadataAttributes({
            [metadataPrefix("stepType")]: "run",
            [metadataPrefix("stepName")]: config.name,
            [metadataPrefix("resolvedStepName")]: stepName
          });
          if (input !== void 0) {
            metadata.input = input;
          }
          const [value, wasCached] = isRoot ? yield this.memoize(stepName, func) : [yield func(), false];
          if (wasCached) {
            setCustomMetadataAttribute(metadataPrefix("state"), "cached");
          } else {
            setCustomMetadataAttribute(metadataPrefix("state"), "run");
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
      return yield runInNewSpan(
        {
          metadata: {
            name: stepName
          },
          labels: {
            [SPAN_TYPE_ATTR]: "flowStep"
          }
        },
        (metadata) => __async(this, null, function* () {
          const resolvedStepName = this.resolveStepName(stepName);
          setCustomMetadataAttributes({
            [metadataPrefix("stepType")]: "interrupt",
            [metadataPrefix("stepName")]: stepName,
            [metadataPrefix("resolvedStepName")]: resolvedStepName
          });
          if (!skipCache && this.isCached(resolvedStepName)) {
            setCustomMetadataAttribute(metadataPrefix("state"), "skipped");
            return this.getCached(resolvedStepName);
          }
          if (this.state.eventsTriggered.hasOwnProperty(resolvedStepName)) {
            let value;
            try {
              value = yield func(
                this.state.eventsTriggered[resolvedStepName]
              );
            } catch (e) {
              if (e instanceof InterruptError) {
                setCustomMetadataAttribute(metadataPrefix("state"), "interrupt");
              } else {
                setCustomMetadataAttribute(metadataPrefix("state"), "error");
              }
              throw e;
            }
            this.state.blockedOnStep = null;
            if (!skipCache) {
              this.updateCachedValue(resolvedStepName, value);
            }
            setCustomMetadataAttribute(metadataPrefix("state"), "dispatch");
            if (value !== void 0) {
              metadata.output = JSON.stringify(value);
            }
            return value;
          }
          logger.debug("blockedOnStep", resolvedStepName);
          this.state.blockedOnStep = { name: resolvedStepName };
          if (responseSchema) {
            this.state.blockedOnStep.schema = JSON.stringify(
              toJsonSchema({ schema: responseSchema })
            );
          }
          setCustomMetadataAttribute(metadataPrefix("state"), "interrupted");
          throw new InterruptError();
        })
      );
    });
  }
  // Sleep for the specified number of seconds.
  sleep(stepName, seconds) {
    return __async(this, null, function* () {
      const resolvedStepName = this.resolveStepName(stepName);
      if (this.isCached(resolvedStepName)) {
        setCustomMetadataAttribute(metadataPrefix("state"), "skipped");
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
      throw new InterruptError();
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
export {
  Context
};
//# sourceMappingURL=context.mjs.map