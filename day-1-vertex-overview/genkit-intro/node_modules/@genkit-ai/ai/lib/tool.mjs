import {
  __async,
  __spreadProps,
  __spreadValues
} from "./chunk-7OAPEGJQ.mjs";
import { defineAction } from "@genkit-ai/core";
import { lookupAction } from "@genkit-ai/core/registry";
import { toJsonSchema } from "@genkit-ai/core/schema";
import { setCustomMetadataAttributes } from "@genkit-ai/core/tracing";
function asTool(action) {
  var _a, _b;
  if (((_b = (_a = action.__action) == null ? void 0 : _a.metadata) == null ? void 0 : _b.type) === "tool") {
    return action;
  }
  const fn = (input) => {
    setCustomMetadataAttributes({ subtype: "tool" });
    return action(input);
  };
  fn.__action = __spreadProps(__spreadValues({}, action.__action), {
    metadata: __spreadProps(__spreadValues({}, action.__action.metadata), { type: "tool" })
  });
  return fn;
}
function resolveTools() {
  return __async(this, arguments, function* (tools = []) {
    return yield Promise.all(
      tools.map((ref) => __async(this, null, function* () {
        if (typeof ref === "string") {
          const tool = yield lookupAction(`/tool/${ref}`);
          if (!tool) {
            throw new Error(`Tool ${ref} not found`);
          }
          return tool;
        } else if (ref.__action) {
          return asTool(ref);
        } else if (ref.name) {
          const tool = yield lookupAction(`/tool/${ref.name}`);
          if (!tool) {
            throw new Error(`Tool ${ref} not found`);
          }
        }
        throw new Error("Tools must be strings, tool definitions, or actions.");
      }))
    );
  });
}
function toToolDefinition(tool) {
  return {
    name: tool.__action.name,
    description: tool.__action.description || "",
    outputSchema: toJsonSchema({
      schema: tool.__action.outputSchema,
      jsonSchema: tool.__action.outputJsonSchema
    }),
    inputSchema: toJsonSchema({
      schema: tool.__action.inputSchema,
      jsonSchema: tool.__action.inputJsonSchema
    })
  };
}
function defineTool({
  name,
  description,
  inputSchema,
  inputJsonSchema,
  outputSchema,
  outputJsonSchema,
  metadata
}, fn) {
  const a = defineAction(
    {
      actionType: "tool",
      name,
      description,
      inputSchema,
      inputJsonSchema,
      outputSchema,
      outputJsonSchema,
      metadata: __spreadProps(__spreadValues({}, metadata || {}), { type: "tool" })
    },
    (i) => fn(i)
  );
  return a;
}
export {
  asTool,
  defineTool,
  resolveTools,
  toToolDefinition
};
//# sourceMappingURL=tool.mjs.map