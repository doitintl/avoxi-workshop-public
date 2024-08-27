import "./chunk-XEFTB2OF.mjs";
import { z } from "zod";
const FlowStateExecutionSchema = z.object({
  startTime: z.number().optional().describe("start time in milliseconds since the epoch"),
  endTime: z.number().optional().describe("end time in milliseconds since the epoch"),
  traceIds: z.array(z.string())
});
const FlowResponseSchema = z.object({
  response: z.unknown().nullable()
});
const FlowErrorSchema = z.object({
  error: z.string().optional(),
  stacktrace: z.string().optional()
});
const FlowResultSchema = FlowResponseSchema.and(FlowErrorSchema);
const OperationSchema = z.object({
  name: z.string().describe(
    "server-assigned name, which is only unique within the same service that originally returns it."
  ),
  metadata: z.any().optional().describe(
    "Service-specific metadata associated with the operation. It typically contains progress information and common metadata such as create time."
  ),
  done: z.boolean().optional().default(false).describe(
    "If the value is false, it means the operation is still in progress. If true, the operation is completed, and either error or response is available."
  ),
  result: FlowResultSchema.optional(),
  blockedOnStep: z.object({
    name: z.string(),
    schema: z.string().optional()
  }).optional()
});
const FlowStateSchema = z.object({
  name: z.string().optional(),
  flowId: z.string(),
  input: z.unknown(),
  startTime: z.number().describe("start time in milliseconds since the epoch"),
  cache: z.record(
    z.string(),
    z.object({
      value: z.any().optional(),
      empty: z.literal(true).optional()
    })
  ),
  eventsTriggered: z.record(z.string(), z.any()),
  blockedOnStep: z.object({
    name: z.string(),
    schema: z.string().optional()
  }).nullable(),
  operation: OperationSchema,
  traceContext: z.string().optional(),
  executions: z.array(FlowStateExecutionSchema)
});
export {
  FlowErrorSchema,
  FlowResponseSchema,
  FlowResultSchema,
  FlowStateExecutionSchema,
  FlowStateSchema,
  OperationSchema
};
//# sourceMappingURL=flowTypes.mjs.map