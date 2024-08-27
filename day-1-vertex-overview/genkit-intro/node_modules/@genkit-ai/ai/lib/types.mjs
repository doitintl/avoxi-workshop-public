import "./chunk-7OAPEGJQ.mjs";
import { toJsonSchema } from "@genkit-ai/core/schema";
import { z } from "zod";
const ModelIdSchema = z.object({
  modelProvider: z.string().readonly(),
  modelName: z.string().readonly()
});
const LlmStatsSchema = z.object({
  latencyMs: z.number().optional(),
  inputTokenCount: z.number().optional(),
  outputTokenCount: z.number().optional()
});
const ToolSchema = z.object({
  name: z.string(),
  description: z.string().optional(),
  schema: z.any()
});
const ToolCallSchema = z.object({
  toolName: z.string(),
  arguments: z.any()
});
const LlmResponseSchema = z.object({
  completion: z.string(),
  toolCalls: z.array(ToolCallSchema).optional(),
  stats: LlmStatsSchema
});
function toToolWireFormat(actions) {
  if (!actions)
    return void 0;
  return actions.map((a) => {
    return {
      name: a.__action.name,
      description: a.__action.description,
      schema: {
        input: toJsonSchema({
          schema: a.__action.inputSchema,
          jsonSchema: a.__action.inputJsonSchema
        }),
        output: toJsonSchema({
          schema: a.__action.outputSchema,
          jsonSchema: a.__action.outputJsonSchema
        })
      }
    };
  });
}
const CommonLlmOptions = z.object({
  temperature: z.number().optional(),
  topK: z.number().optional(),
  topP: z.number().optional()
});
export {
  CommonLlmOptions,
  LlmResponseSchema,
  LlmStatsSchema,
  ModelIdSchema,
  ToolCallSchema,
  ToolSchema,
  toToolWireFormat
};
//# sourceMappingURL=types.mjs.map