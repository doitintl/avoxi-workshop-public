"use strict";
var __defProp = Object.defineProperty;
var __defProps = Object.defineProperties;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropDescs = Object.getOwnPropertyDescriptors;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getOwnPropSymbols = Object.getOwnPropertySymbols;
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
var __spreadProps = (a, b) => __defProps(a, __getOwnPropDescs(b));
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
var tool_exports = {};
__export(tool_exports, {
  asTool: () => asTool,
  defineTool: () => defineTool,
  resolveTools: () => resolveTools,
  toToolDefinition: () => toToolDefinition
});
module.exports = __toCommonJS(tool_exports);
var import_core = require("@genkit-ai/core");
var import_registry = require("@genkit-ai/core/registry");
var import_schema = require("@genkit-ai/core/schema");
var import_tracing = require("@genkit-ai/core/tracing");
function asTool(action) {
  var _a, _b;
  if (((_b = (_a = action.__action) == null ? void 0 : _a.metadata) == null ? void 0 : _b.type) === "tool") {
    return action;
  }
  const fn = (input) => {
    (0, import_tracing.setCustomMetadataAttributes)({ subtype: "tool" });
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
          const tool = yield (0, import_registry.lookupAction)(`/tool/${ref}`);
          if (!tool) {
            throw new Error(`Tool ${ref} not found`);
          }
          return tool;
        } else if (ref.__action) {
          return asTool(ref);
        } else if (ref.name) {
          const tool = yield (0, import_registry.lookupAction)(`/tool/${ref.name}`);
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
    outputSchema: (0, import_schema.toJsonSchema)({
      schema: tool.__action.outputSchema,
      jsonSchema: tool.__action.outputJsonSchema
    }),
    inputSchema: (0, import_schema.toJsonSchema)({
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
  const a = (0, import_core.defineAction)(
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
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  asTool,
  defineTool,
  resolveTools,
  toToolDefinition
});
//# sourceMappingURL=tool.js.map