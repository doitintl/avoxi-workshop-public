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
var retriever_exports = {};
__export(retriever_exports, {
  CommonRetrieverOptionsSchema: () => CommonRetrieverOptionsSchema,
  Document: () => import_document2.Document,
  DocumentDataSchema: () => import_document2.DocumentDataSchema,
  IndexerInfoSchema: () => IndexerInfoSchema,
  RetrieverInfoSchema: () => RetrieverInfoSchema,
  defineIndexer: () => defineIndexer,
  defineRetriever: () => defineRetriever,
  defineSimpleRetriever: () => defineSimpleRetriever,
  index: () => index,
  indexerRef: () => indexerRef,
  retrieve: () => retrieve,
  retrieverRef: () => retrieverRef
});
module.exports = __toCommonJS(retriever_exports);
var import_core = require("@genkit-ai/core");
var import_registry = require("@genkit-ai/core/registry");
var z = __toESM(require("zod"));
var import_document = require("./document.js");
var import_document2 = require("./document.js");
const RetrieverRequestSchema = z.object({
  query: import_document.DocumentDataSchema,
  options: z.any().optional()
});
const RetrieverResponseSchema = z.object({
  documents: z.array(import_document.DocumentDataSchema)
  // TODO: stats, etc.
});
const IndexerRequestSchema = z.object({
  documents: z.array(import_document.DocumentDataSchema),
  options: z.any().optional()
});
const RetrieverInfoSchema = z.object({
  label: z.string().optional(),
  /** Supported model capabilities. */
  supports: z.object({
    /** Model can process media as part of the prompt (multimodal input). */
    media: z.boolean().optional()
  }).optional()
});
function retrieverWithMetadata(retriever, configSchema) {
  const withMeta = retriever;
  withMeta.__configSchema = configSchema;
  return withMeta;
}
function indexerWithMetadata(indexer, configSchema) {
  const withMeta = indexer;
  withMeta.__configSchema = configSchema;
  return withMeta;
}
function defineRetriever(options, runner) {
  const retriever = (0, import_core.defineAction)(
    {
      actionType: "retriever",
      name: options.name,
      inputSchema: options.configSchema ? RetrieverRequestSchema.extend({
        options: options.configSchema.optional()
      }) : RetrieverRequestSchema,
      outputSchema: RetrieverResponseSchema,
      metadata: {
        type: "retriever",
        info: options.info
      }
    },
    (i) => runner(new import_document.Document(i.query), i.options)
  );
  const rwm = retrieverWithMetadata(
    retriever,
    options.configSchema
  );
  return rwm;
}
function defineIndexer(options, runner) {
  const indexer = (0, import_core.defineAction)(
    {
      actionType: "indexer",
      name: options.name,
      inputSchema: options.configSchema ? IndexerRequestSchema.extend({
        options: options.configSchema.optional()
      }) : IndexerRequestSchema,
      outputSchema: z.void(),
      metadata: {
        type: "indexer",
        embedderInfo: options.embedderInfo
      }
    },
    (i) => runner(
      i.documents.map((dd) => new import_document.Document(dd)),
      i.options
    )
  );
  const iwm = indexerWithMetadata(
    indexer,
    options.configSchema
  );
  return iwm;
}
function retrieve(params) {
  return __async(this, null, function* () {
    let retriever;
    if (typeof params.retriever === "string") {
      retriever = yield (0, import_registry.lookupAction)(`/retriever/${params.retriever}`);
    } else if (Object.hasOwnProperty.call(params.retriever, "info")) {
      retriever = yield (0, import_registry.lookupAction)(`/retriever/${params.retriever.name}`);
    } else {
      retriever = params.retriever;
    }
    if (!retriever) {
      throw new Error("Unable to resolve the retriever");
    }
    const response = yield retriever({
      query: typeof params.query === "string" ? import_document.Document.fromText(params.query) : params.query,
      options: params.options
    });
    return response.documents.map((d) => new import_document.Document(d));
  });
}
function index(params) {
  return __async(this, null, function* () {
    let indexer;
    if (typeof params.indexer === "string") {
      indexer = yield (0, import_registry.lookupAction)(`/indexer/${params.indexer}`);
    } else if (Object.hasOwnProperty.call(params.indexer, "info")) {
      indexer = yield (0, import_registry.lookupAction)(`/indexer/${params.indexer.name}`);
    } else {
      indexer = params.indexer;
    }
    if (!indexer) {
      throw new Error("Unable to utilize the provided indexer");
    }
    return yield indexer({
      documents: params.documents,
      options: params.options
    });
  });
}
const CommonRetrieverOptionsSchema = z.object({
  k: z.number().describe("Number of documents to retrieve").optional()
});
function retrieverRef(options) {
  return __spreadValues({}, options);
}
const IndexerInfoSchema = RetrieverInfoSchema;
function indexerRef(options) {
  return __spreadValues({}, options);
}
function itemToDocument(item, options) {
  if (!item)
    throw new import_core.GenkitError({
      status: "INVALID_ARGUMENT",
      message: `Items returned from simple retriever must be non-null.`
    });
  if (typeof item === "string")
    return import_document.Document.fromText(item);
  if (typeof options.content === "function") {
    const transformed = options.content(item);
    return typeof transformed === "string" ? import_document.Document.fromText(transformed) : new import_document.Document({ content: transformed });
  }
  if (typeof options.content === "string" && typeof item === "object")
    return import_document.Document.fromText(item[options.content]);
  throw new import_core.GenkitError({
    status: "INVALID_ARGUMENT",
    message: `Cannot convert item to document without content option. Item: ${JSON.stringify(item)}`
  });
}
function itemToMetadata(item, options) {
  if (typeof item === "string")
    return void 0;
  if (Array.isArray(options.metadata) && typeof item === "object") {
    const out = {};
    options.metadata.forEach((key) => out[key] = item[key]);
  }
  if (typeof options.metadata === "function")
    return options.metadata(item);
  if (!options.metadata && typeof item === "object") {
    const out = __spreadValues({}, item);
    if (typeof options.content === "string")
      delete out[options.content];
    return out;
  }
  throw new import_core.GenkitError({
    status: "INVALID_ARGUMENT",
    message: `Unable to extract metadata from item with supplied options. Item: ${JSON.stringify(item)}`
  });
}
function defineSimpleRetriever(options, handler) {
  return defineRetriever(
    {
      name: options.name,
      configSchema: options.configSchema
    },
    (query, config) => __async(this, null, function* () {
      const result = yield handler(query, config);
      return {
        documents: result.map((item) => {
          const doc = itemToDocument(item, options);
          if (typeof item !== "string")
            doc.metadata = itemToMetadata(item, options);
          return doc;
        })
      };
    })
  );
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  CommonRetrieverOptionsSchema,
  Document,
  DocumentDataSchema,
  IndexerInfoSchema,
  RetrieverInfoSchema,
  defineIndexer,
  defineRetriever,
  defineSimpleRetriever,
  index,
  indexerRef,
  retrieve,
  retrieverRef
});
//# sourceMappingURL=retriever.js.map