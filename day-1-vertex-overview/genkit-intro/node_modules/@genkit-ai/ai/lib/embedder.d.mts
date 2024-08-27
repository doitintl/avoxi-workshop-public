import { Action } from '@genkit-ai/core';
import * as z from 'zod';
import { DocumentData, Document } from './document.mjs';

/**
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

type EmbeddingBatch = {
    embedding: number[];
}[];
declare const EmbeddingSchema: z.ZodArray<z.ZodNumber, "many">;
type Embedding = z.infer<typeof EmbeddingSchema>;
type EmbedderFn<EmbedderOptions extends z.ZodTypeAny> = (input: Document[], embedderOpts?: z.infer<EmbedderOptions>) => Promise<EmbedResponse>;
declare const EmbedRequestSchema: z.ZodObject<{
    input: z.ZodArray<z.ZodObject<{
        content: z.ZodArray<z.ZodUnion<[z.ZodObject<{
            media: z.ZodOptional<z.ZodNever>;
            text: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            text: string;
            media?: undefined;
        }, {
            text: string;
            media?: undefined;
        }>, z.ZodObject<{
            text: z.ZodOptional<z.ZodNever>;
            media: z.ZodObject<{
                contentType: z.ZodOptional<z.ZodString>;
                url: z.ZodString;
            }, "strip", z.ZodTypeAny, {
                url: string;
                contentType?: string | undefined;
            }, {
                url: string;
                contentType?: string | undefined;
            }>;
        }, "strip", z.ZodTypeAny, {
            media: {
                url: string;
                contentType?: string | undefined;
            };
            text?: undefined;
        }, {
            media: {
                url: string;
                contentType?: string | undefined;
            };
            text?: undefined;
        }>]>, "many">;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        content: ({
            text: string;
            media?: undefined;
        } | {
            media: {
                url: string;
                contentType?: string | undefined;
            };
            text?: undefined;
        })[];
        metadata?: Record<string, any> | undefined;
    }, {
        content: ({
            text: string;
            media?: undefined;
        } | {
            media: {
                url: string;
                contentType?: string | undefined;
            };
            text?: undefined;
        })[];
        metadata?: Record<string, any> | undefined;
    }>, "many">;
    options: z.ZodOptional<z.ZodAny>;
}, "strip", z.ZodTypeAny, {
    input: {
        content: ({
            text: string;
            media?: undefined;
        } | {
            media: {
                url: string;
                contentType?: string | undefined;
            };
            text?: undefined;
        })[];
        metadata?: Record<string, any> | undefined;
    }[];
    options?: any;
}, {
    input: {
        content: ({
            text: string;
            media?: undefined;
        } | {
            media: {
                url: string;
                contentType?: string | undefined;
            };
            text?: undefined;
        })[];
        metadata?: Record<string, any> | undefined;
    }[];
    options?: any;
}>;
declare const EmbedResponseSchema: z.ZodObject<{
    embeddings: z.ZodArray<z.ZodObject<{
        embedding: z.ZodArray<z.ZodNumber, "many">;
    }, "strip", z.ZodTypeAny, {
        embedding: number[];
    }, {
        embedding: number[];
    }>, "many">;
}, "strip", z.ZodTypeAny, {
    embeddings: {
        embedding: number[];
    }[];
}, {
    embeddings: {
        embedding: number[];
    }[];
}>;
type EmbedResponse = z.infer<typeof EmbedResponseSchema>;
type EmbedderAction<CustomOptions extends z.ZodTypeAny = z.ZodTypeAny> = Action<typeof EmbedRequestSchema, typeof EmbedResponseSchema> & {
    __configSchema?: CustomOptions;
};
/**
 * Creates embedder model for the provided {@link EmbedderFn} model implementation.
 */
declare function defineEmbedder<ConfigSchema extends z.ZodTypeAny = z.ZodTypeAny>(options: {
    name: string;
    configSchema?: ConfigSchema;
    info?: EmbedderInfo;
}, runner: EmbedderFn<ConfigSchema>): EmbedderAction<ConfigSchema>;
type EmbedderArgument<CustomOptions extends z.ZodTypeAny = z.ZodTypeAny> = string | EmbedderAction<CustomOptions> | EmbedderReference<CustomOptions>;
/**
 * A veneer for interacting with embedder models.
 */
declare function embed<ConfigSchema extends z.ZodTypeAny = z.ZodTypeAny>(params: {
    embedder: EmbedderArgument<ConfigSchema>;
    content: string | DocumentData;
    metadata?: Record<string, unknown>;
    options?: z.infer<ConfigSchema>;
}): Promise<Embedding>;
/**
 * A veneer for interacting with embedder models in bulk.
 */
declare function embedMany<ConfigSchema extends z.ZodTypeAny = z.ZodTypeAny>(params: {
    embedder: EmbedderArgument<ConfigSchema>;
    content: string[] | DocumentData[];
    metadata?: Record<string, unknown>;
    options?: z.infer<ConfigSchema>;
}): Promise<EmbeddingBatch>;
declare const EmbedderInfoSchema: z.ZodObject<{
    /** Friendly label for this model (e.g. "Google AI - Gemini Pro") */
    label: z.ZodOptional<z.ZodString>;
    /** Supported model capabilities. */
    supports: z.ZodOptional<z.ZodObject<{
        /** Model can input this type of data. */
        input: z.ZodOptional<z.ZodArray<z.ZodEnum<["text", "image"]>, "many">>;
        /** Model can support multiple languages */
        multilingual: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        input?: ("text" | "image")[] | undefined;
        multilingual?: boolean | undefined;
    }, {
        input?: ("text" | "image")[] | undefined;
        multilingual?: boolean | undefined;
    }>>;
    /** Embedding dimension */
    dimensions: z.ZodOptional<z.ZodNumber>;
}, "strip", z.ZodTypeAny, {
    label?: string | undefined;
    supports?: {
        input?: ("text" | "image")[] | undefined;
        multilingual?: boolean | undefined;
    } | undefined;
    dimensions?: number | undefined;
}, {
    label?: string | undefined;
    supports?: {
        input?: ("text" | "image")[] | undefined;
        multilingual?: boolean | undefined;
    } | undefined;
    dimensions?: number | undefined;
}>;
type EmbedderInfo = z.infer<typeof EmbedderInfoSchema>;
interface EmbedderReference<CustomOptions extends z.ZodTypeAny = z.ZodTypeAny> {
    name: string;
    configSchema?: CustomOptions;
    info?: EmbedderInfo;
}
/**
 * Helper method to configure a {@link EmbedderReference} to a plugin.
 */
declare function embedderRef<CustomOptionsSchema extends z.ZodTypeAny = z.ZodTypeAny>(options: EmbedderReference<CustomOptionsSchema>): EmbedderReference<CustomOptionsSchema>;

export { type EmbedderAction, type EmbedderArgument, type EmbedderInfo, EmbedderInfoSchema, type EmbedderReference, type EmbeddingBatch, EmbeddingSchema, defineEmbedder, embed, embedMany, embedderRef };
