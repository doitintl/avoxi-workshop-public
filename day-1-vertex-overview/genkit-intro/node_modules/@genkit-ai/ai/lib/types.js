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
var types_exports = {};
__export(types_exports, {
  CommonLlmOptions: () => CommonLlmOptions,
  LlmResponseSchema: () => LlmResponseSchema,
  LlmStatsSchema: () => LlmStatsSchema,
  ModelIdSchema: () => ModelIdSchema,
  ToolCallSchema: () => ToolCallSchema,
  ToolSchema: () => ToolSchema,
  toToolWireFormat: () => toToolWireFormat
});
module.exports = __toCommonJS(types_exports);
var import_schema = require("@genkit-ai/core/schema");
var import_zod = require("zod");
const ModelIdSchema = import_zod.z.object({
  modelProvider: import_zod.z.string().readonly(),
  modelName: import_zod.z.string().readonly()
});
const LlmStatsSchema = import_zod.z.object({
  latencyMs: import_zod.z.number().optional(),
  inputTokenCount: import_zod.z.number().optional(),
  outputTokenCount: import_zod.z.number().optional()
});
const ToolSchema = import_zod.z.object({
  name: import_zod.z.string(),
  description: import_zod.z.string().optional(),
  schema: import_zod.z.any()
});
const ToolCallSchema = import_zod.z.object({
  toolName: import_zod.z.string(),
  arguments: import_zod.z.any()
});
const LlmResponseSchema = import_zod.z.object({
  completion: import_zod.z.string(),
  toolCalls: import_zod.z.array(ToolCallSchema).optional(),
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
        input: (0, import_schema.toJsonSchema)({
          schema: a.__action.inputSchema,
          jsonSchema: a.__action.inputJsonSchema
        }),
        output: (0, import_schema.toJsonSchema)({
          schema: a.__action.outputSchema,
          jsonSchema: a.__action.outputJsonSchema
        })
      }
    };
  });
}
const CommonLlmOptions = import_zod.z.object({
  temperature: import_zod.z.number().optional(),
  topK: import_zod.z.number().optional(),
  topP: import_zod.z.number().optional()
});
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  CommonLlmOptions,
  LlmResponseSchema,
  LlmStatsSchema,
  ModelIdSchema,
  ToolCallSchema,
  ToolSchema,
  toToolWireFormat
});
//# sourceMappingURL=types.js.map