import {
  __async,
  __spreadValues
} from "./chunk-7OAPEGJQ.mjs";
import { defineAction } from "@genkit-ai/core";
import { lookupAction } from "@genkit-ai/core/registry";
import * as z from "zod";
import { Document, DocumentDataSchema } from "./document.js";
const EmbeddingSchema = z.array(z.number());
const EmbedRequestSchema = z.object({
  input: z.array(DocumentDataSchema),
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
  const embedder = defineAction(
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
      i.input.map((dd) => new Document(dd)),
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
      embedder = yield lookupAction(`/embedder/${params.embedder}`);
    } else if (Object.hasOwnProperty.call(params.embedder, "info")) {
      embedder = yield lookupAction(
        `/embedder/${params.embedder.name}`
      );
    } else {
      embedder = params.embedder;
    }
    if (!embedder) {
      throw new Error("Unable to utilize the provided embedder");
    }
    const response = yield embedder({
      input: typeof params.content === "string" ? [Document.fromText(params.content, params.metadata)] : [params.content],
      options: params.options
    });
    return response.embeddings[0].embedding;
  });
}
function embedMany(params) {
  return __async(this, null, function* () {
    let embedder;
    if (typeof params.embedder === "string") {
      embedder = yield lookupAction(`/embedder/${params.embedder}`);
    } else if (Object.hasOwnProperty.call(params.embedder, "info")) {
      embedder = yield lookupAction(
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
        (i) => typeof i === "string" ? Document.fromText(i, params.metadata) : i
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
export {
  EmbedderInfoSchema,
  EmbeddingSchema,
  defineEmbedder,
  embed,
  embedMany,
  embedderRef
};
//# sourceMappingURL=embedder.mjs.map