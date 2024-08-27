"use strict";
var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getOwnPropSymbols = Object.getOwnPropertySymbols;
var __getProtoOf = Object.getPrototypeOf;
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
var evaluator_exports = {};
__export(evaluator_exports, {
  ATTR_PREFIX: () => ATTR_PREFIX,
  BaseDataPointSchema: () => BaseDataPointSchema,
  EVALUATOR_METADATA_KEY_DEFINITION: () => EVALUATOR_METADATA_KEY_DEFINITION,
  EVALUATOR_METADATA_KEY_DISPLAY_NAME: () => EVALUATOR_METADATA_KEY_DISPLAY_NAME,
  EVALUATOR_METADATA_KEY_IS_BILLED: () => EVALUATOR_METADATA_KEY_IS_BILLED,
  EvalResponseSchema: () => EvalResponseSchema,
  EvalResponsesSchema: () => EvalResponsesSchema,
  EvaluatorInfoSchema: () => EvaluatorInfoSchema,
  SPAN_STATE_ATTR: () => SPAN_STATE_ATTR,
  ScoreSchema: () => ScoreSchema,
  defineEvaluator: () => defineEvaluator,
  evaluate: () => evaluate,
  evaluatorRef: () => evaluatorRef
});
module.exports = __toCommonJS(evaluator_exports);
var import_core = require("@genkit-ai/core");
var import_logging = require("@genkit-ai/core/logging");
var import_registry = require("@genkit-ai/core/registry");
var import_tracing = require("@genkit-ai/core/tracing");
var z = __toESM(require("zod"));
const ATTR_PREFIX = "genkit";
const SPAN_STATE_ATTR = ATTR_PREFIX + ":state";
const BaseDataPointSchema = z.object({
  input: z.unknown(),
  output: z.unknown().optional(),
  context: z.array(z.unknown()).optional(),
  reference: z.unknown().optional(),
  testCaseId: z.string().optional(),
  traceIds: z.array(z.string()).optional()
});
const ScoreSchema = z.object({
  score: z.union([z.number(), z.string(), z.boolean()]).optional(),
  // TODO: use StatusSchema
  error: z.string().optional(),
  details: z.object({
    reasoning: z.string().optional()
  }).passthrough().optional()
});
const EVALUATOR_METADATA_KEY_DISPLAY_NAME = "evaluatorDisplayName";
const EVALUATOR_METADATA_KEY_DEFINITION = "evaluatorDefinition";
const EVALUATOR_METADATA_KEY_IS_BILLED = "evaluatorIsBilled";
const EvalResponseSchema = z.object({
  sampleIndex: z.number().optional(),
  testCaseId: z.string().optional(),
  traceId: z.string().optional(),
  spanId: z.string().optional(),
  evaluation: ScoreSchema
});
const EvalResponsesSchema = z.array(EvalResponseSchema);
function withMetadata(evaluator, dataPointType, configSchema) {
  const withMeta = evaluator;
  withMeta.__dataPointType = dataPointType;
  withMeta.__configSchema = configSchema;
  return withMeta;
}
const EvalRequestSchema = z.object({
  dataset: z.array(BaseDataPointSchema),
  options: z.unknown()
});
function defineEvaluator(options, runner) {
  var _a;
  const metadata = {};
  metadata[EVALUATOR_METADATA_KEY_IS_BILLED] = options.isBilled == void 0 ? true : options.isBilled;
  metadata[EVALUATOR_METADATA_KEY_DISPLAY_NAME] = options.displayName;
  metadata[EVALUATOR_METADATA_KEY_DEFINITION] = options.definition;
  const evaluator = (0, import_core.defineAction)(
    {
      actionType: "evaluator",
      name: options.name,
      inputSchema: EvalRequestSchema.extend({
        dataset: options.dataPointType ? z.array(options.dataPointType) : z.array(BaseDataPointSchema),
        options: (_a = options.configSchema) != null ? _a : z.unknown(),
        evalRunId: z.string()
      }),
      outputSchema: EvalResponsesSchema,
      metadata
    },
    (i) => __async(this, null, function* () {
      let evalResponses = [];
      for (let index = 0; index < i.dataset.length; index++) {
        const datapoint = i.dataset[index];
        try {
          yield (0, import_tracing.runInNewSpan)(
            {
              metadata: {
                name: `Test Case ${datapoint.testCaseId}`,
                metadata: { "evaluator:evalRunId": i.evalRunId }
              },
              labels: {
                [import_tracing.SPAN_TYPE_ATTR]: "evaluator"
              }
            },
            (metadata2, otSpan) => __async(this, null, function* () {
              const spanId = otSpan.spanContext().spanId;
              const traceId = otSpan.spanContext().traceId;
              try {
                metadata2.input = {
                  input: datapoint.input,
                  output: datapoint.output,
                  context: datapoint.context
                };
                const testCaseOutput = yield runner(datapoint, i.options);
                testCaseOutput.sampleIndex = index;
                testCaseOutput.spanId = spanId;
                testCaseOutput.traceId = traceId;
                metadata2.output = testCaseOutput;
                evalResponses.push(testCaseOutput);
                return testCaseOutput;
              } catch (e) {
                evalResponses.push({
                  sampleIndex: index,
                  spanId,
                  traceId,
                  testCaseId: datapoint.testCaseId,
                  evaluation: {
                    error: `Evaluation of test case ${datapoint.testCaseId} failed: 
${e.stack}`
                  }
                });
                throw e;
              }
            })
          );
        } catch (e) {
          import_logging.logger.error(
            `Evaluation of test case ${datapoint.testCaseId} failed: 
${e.stack}`
          );
          continue;
        }
      }
      return evalResponses;
    })
  );
  const ewm = withMetadata(
    evaluator,
    options.dataPointType,
    options.configSchema
  );
  return ewm;
}
function evaluate(params) {
  return __async(this, null, function* () {
    let evaluator;
    if (typeof params.evaluator === "string") {
      evaluator = yield (0, import_registry.lookupAction)(`/evaluator/${params.evaluator}`);
    } else if (Object.hasOwnProperty.call(params.evaluator, "info")) {
      evaluator = yield (0, import_registry.lookupAction)(`/evaluator/${params.evaluator.name}`);
    } else {
      evaluator = params.evaluator;
    }
    if (!evaluator) {
      throw new Error("Unable to utilize the provided evaluator");
    }
    return yield evaluator({
      dataset: params.dataset,
      options: params.options
    });
  });
}
const EvaluatorInfoSchema = z.object({
  /** Friendly label for this evaluator */
  label: z.string().optional(),
  metrics: z.array(z.string())
});
function evaluatorRef(options) {
  return __spreadValues({}, options);
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  ATTR_PREFIX,
  BaseDataPointSchema,
  EVALUATOR_METADATA_KEY_DEFINITION,
  EVALUATOR_METADATA_KEY_DISPLAY_NAME,
  EVALUATOR_METADATA_KEY_IS_BILLED,
  EvalResponseSchema,
  EvalResponsesSchema,
  EvaluatorInfoSchema,
  SPAN_STATE_ATTR,
  ScoreSchema,
  defineEvaluator,
  evaluate,
  evaluatorRef
});
//# sourceMappingURL=evaluator.js.map