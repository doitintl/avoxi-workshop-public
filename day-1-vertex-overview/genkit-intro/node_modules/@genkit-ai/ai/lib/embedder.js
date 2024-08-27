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
var embedder_exports = {};
__export(embedder_exports, {
  EmbedderInfoSchema: () => EmbedderInfoSchema,
  EmbeddingSchema: () => EmbeddingSchema,
  defineEmbedder: () => defineEmbedder,
  embed: () => embed,
  embedMany: () => embedMany,
  embedderRef: () => embedderRef
});
module.exports = __toCommonJS(embedder_exports);
var import_core = require("@genkit-ai/core");
var import_registry = require("@genkit-ai/core/registry");
var z = __toESM(require("zod"));
var import_document = require("./document.js");
const EmbeddingSchema = z.array(z.number());
const EmbedRequestSchema = z.object({
  input: z.array(import_document.DocumentDataSchema),
  options: z.any().optional()
});
const EmbedResponseSchema = z.object({
  embeddings: z.array(z.object({ embedding: EmbeddingSchema }))
  // TODO: stats, etc.
});
function withMetadata(embedder, configSchema) {
  const withMeta = embedder;
  withMeta.__configSchema = configSchema;
  return withMeta;
}
function defineEmbedder(options, runner) {
  const embedder = (0, import_core.defineAction)(
    {
      actionType: "embedder",
      name: options.name,
      inputSchema: options.configSchema ? EmbedRequestSchema.extend({
        options: options.configSchema.optional()
      }) : EmbedRequestSchema,
      outputSchema: EmbedResponseSchema,
      metadata: {
        type: "embedder",
        info: options.info
      }
    },
    (i) => runner(
      i.input.map((dd) => new import_document.Document(dd)),
      i.options
    )
  );
  const ewm = withMetadata(
    embedder,
    options.configSchema
  );
  return ewm;
}
function embed(params) {
  return __async(this, null, function* () {
    let embedder;
    if (typeof params.embedder === "string") {
      embedder = yield (0, import_registry.lookupAction)(`/embedder/${params.embedder}`);
    } else if (Object.hasOwnProperty.call(params.embedder, "info")) {
      embedder = yield (0, import_registry.lookupAction)(
        `/embedder/${params.embedder.name}`
      );
    } else {
      embedder = params.embedder;
    }
    if (!embedder) {
      throw new Error("Unable to utilize the provided embedder");
    }
    const response = yield embedder({
      input: typeof params.content === "string" ? [import_document.Document.fromText(params.content, params.metadata)] : [params.content],
      options: params.options
    });
    return response.embeddings[0].embedding;
  });
}
function embedMany(params) {
  return __async(this, null, function* () {
    let embedder;
    if (typeof params.embedder === "string") {
      embedder = yield (0, import_registry.lookupAction)(`/embedder/${params.embedder}`);
    } else if (Object.hasOwnProperty.call(params.embedder, "info")) {
      embedder = yield (0, import_registry.lookupAction)(
        `/embedder/${params.embedder.name}`
      );
    } else {
      embedder = params.embedder;
    }
    if (!embedder) {
      throw new Error("Unable to utilize the provided embedder");
    }
    const response = yield embedder({
      input: params.content.map(
        (i) => typeof i === "string" ? import_document.Document.fromText(i, params.metadata) : i
      ),
      options: params.options
    });
    return response.embeddings;
  });
}
const EmbedderInfoSchema = z.object({
  /** Friendly label for this model (e.g. "Google AI - Gemini Pro") */
  label: z.string().optional(),
  /** Supported model capabilities. */
  supports: z.object({
    /** Model can input this type of data. */
    input: z.array(z.enum(["text", "image"])).optional(),
    /** Model can support multiple languages */
    multilingual: z.boolean().optional()
  }).optional(),
  /** Embedding dimension */
  dimensions: z.number().optional()
});
function embedderRef(options) {
  return __spreadValues({}, options);
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  EmbedderInfoSchema,
  EmbeddingSchema,
  defineEmbedder,
  embed,
  embedMany,
  embedderRef
});
//# sourceMappingURL=embedder.js.map