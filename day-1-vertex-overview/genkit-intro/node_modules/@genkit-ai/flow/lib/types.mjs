import "./chunk-7OAPEGJQ.mjs";
import { z } from "zod";
const FlowInvokeEnvelopeMessageSchema = z.object({
  // Start new flow.
  start: z.object({
    input: z.unknown().optional(),
    labels: z.record(z.string(), z.string()).optional()
  }).optional(),
  // Schedule new flow.
  schedule: z.object({
    input: z.unknown().optional(),
    delay: z.number().optional()
  }).optional(),
  // Run previously scheduled flow.
  runScheduled: z.object({
    flowId: z.string()
  }).optional(),
  // Retry failed step (only if step is setup for retry)
  retry: z.object({
    flowId: z.string()
  }).optional(),
  // Resume an interrupted flow.
  resume: z.object({
    flowId: z.string(),
    payload: z.unknown().optional()
  }).optional(),
  // State check for a given flow ID. No side effects, can be used to check flow state.
  state: z.object({
    flowId: z.string()
  }).optional()
});
const FlowActionInputSchema = FlowInvokeEnvelopeMessageSchema.extend({
  auth: z.unknown().optional()
});
export {
  FlowActionInputSchema,
  FlowInvokeEnvelopeMessageSchema
};
//# sourceMappingURL=types.mjs.map