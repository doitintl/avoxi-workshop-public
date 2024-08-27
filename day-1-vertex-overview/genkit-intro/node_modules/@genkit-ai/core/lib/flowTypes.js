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
var flowTypes_exports = {};
__export(flowTypes_exports, {
  FlowErrorSchema: () => FlowErrorSchema,
  FlowResponseSchema: () => FlowResponseSchema,
  FlowResultSchema: () => FlowResultSchema,
  FlowStateExecutionSchema: () => FlowStateExecutionSchema,
  FlowStateSchema: () => FlowStateSchema,
  OperationSchema: () => OperationSchema
});
module.exports = __toCommonJS(flowTypes_exports);
var import_zod = require("zod");
const FlowStateExecutionSchema = import_zod.z.object({
  startTime: import_zod.z.number().optional().describe("start time in milliseconds since the epoch"),
  endTime: import_zod.z.number().optional().describe("end time in milliseconds since the epoch"),
  traceIds: import_zod.z.array(import_zod.z.string())
});
const FlowResponseSchema = import_zod.z.object({
  response: import_zod.z.unknown().nullable()
});
const FlowErrorSchema = import_zod.z.object({
  error: import_zod.z.string().optional(),
  stacktrace: import_zod.z.string().optional()
});
const FlowResultSchema = FlowResponseSchema.and(FlowErrorSchema);
const OperationSchema = import_zod.z.object({
  name: import_zod.z.string().describe(
    "server-assigned name, which is only unique within the same service that originally returns it."
  ),
  metadata: import_zod.z.any().optional().describe(
    "Service-specific metadata associated with the operation. It typically contains progress information and common metadata such as create time."
  ),
  done: import_zod.z.boolean().optional().default(false).describe(
    "If the value is false, it means the operation is still in progress. If true, the operation is completed, and either error or response is available."
  ),
  result: FlowResultSchema.optional(),
  blockedOnStep: import_zod.z.object({
    name: import_zod.z.string(),
    schema: import_zod.z.string().optional()
  }).optional()
});
const FlowStateSchema = import_zod.z.object({
  name: import_zod.z.string().optional(),
  flowId: import_zod.z.string(),
  input: import_zod.z.unknown(),
  startTime: import_zod.z.number().describe("start time in milliseconds since the epoch"),
  cache: import_zod.z.record(
    import_zod.z.string(),
    import_zod.z.object({
      value: import_zod.z.any().optional(),
      empty: import_zod.z.literal(true).optional()
    })
  ),
  eventsTriggered: import_zod.z.record(import_zod.z.string(), import_zod.z.any()),
  blockedOnStep: import_zod.z.object({
    name: import_zod.z.string(),
    schema: import_zod.z.string().optional()
  }).nullable(),
  operation: OperationSchema,
  traceContext: import_zod.z.string().optional(),
  executions: import_zod.z.array(FlowStateExecutionSchema)
});
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  FlowErrorSchema,
  FlowResponseSchema,
  FlowResultSchema,
  FlowStateExecutionSchema,
  FlowStateSchema,
  OperationSchema
});
//# sourceMappingURL=flowTypes.js.map