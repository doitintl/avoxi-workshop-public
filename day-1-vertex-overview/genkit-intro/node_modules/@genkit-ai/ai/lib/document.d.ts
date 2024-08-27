import z__default from 'zod';

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

declare const TextPartSchema: z__default.ZodObject<{
    media: z__default.ZodOptional<z__default.ZodNever>;
    text: z__default.ZodString;
}, "strip", z__default.ZodTypeAny, {
    text: string;
    media?: undefined;
}, {
    text: string;
    media?: undefined;
}>;
type TextPart = z__default.infer<typeof TextPartSchema>;
declare const MediaPartSchema: z__default.ZodObject<{
    text: z__default.ZodOptional<z__default.ZodNever>;
    media: z__default.ZodObject<{
        /** The media content type. Inferred from data uri if not provided. */
        contentType: z__default.ZodOptional<z__default.ZodString>;
        /** A `data:` or `https:` uri containing the media content.  */
        url: z__default.ZodString;
    }, "strip", z__default.ZodTypeAny, {
        url: string;
        contentType?: string | undefined;
    }, {
        url: string;
        contentType?: string | undefined;
    }>;
}, "strip", z__default.ZodTypeAny, {
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
}>;
type MediaPart = z__default.infer<typeof MediaPartSchema>;
declare const PartSchema: z__default.ZodUnion<[z__default.ZodObject<{
    media: z__default.ZodOptional<z__default.ZodNever>;
    text: z__default.ZodString;
}, "strip", z__default.ZodTypeAny, {
    text: string;
    media?: undefined;
}, {
    text: string;
    media?: undefined;
}>, z__default.ZodObject<{
    text: z__default.ZodOptional<z__default.ZodNever>;
    media: z__default.ZodObject<{
        /** The media content type. Inferred from data uri if not provided. */
        contentType: z__default.ZodOptional<z__default.ZodString>;
        /** A `data:` or `https:` uri containing the media content.  */
        url: z__default.ZodString;
    }, "strip", z__default.ZodTypeAny, {
        url: string;
        contentType?: string | undefined;
    }, {
        url: string;
        contentType?: string | undefined;
    }>;
}, "strip", z__default.ZodTypeAny, {
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
}>]>;
type Part = z__default.infer<typeof PartSchema>;
declare const DocumentDataSchema: z__default.ZodObject<{
    content: z__default.ZodArray<z__default.ZodUnion<[z__default.ZodObject<{
        media: z__default.ZodOptional<z__default.ZodNever>;
        text: z__default.ZodString;
    }, "strip", z__default.ZodTypeAny, {
        text: string;
        media?: undefined;
    }, {
        text: string;
        media?: undefined;
    }>, z__default.ZodObject<{
        text: z__default.ZodOptional<z__default.ZodNever>;
        media: z__default.ZodObject<{
            /** The media content type. Inferred from data uri if not provided. */
            contentType: z__default.ZodOptional<z__default.ZodString>;
            /** A `data:` or `https:` uri containing the media content.  */
            url: z__default.ZodString;
        }, "strip", z__default.ZodTypeAny, {
            url: string;
            contentType?: string | undefined;
        }, {
            url: string;
            contentType?: string | undefined;
        }>;
    }, "strip", z__default.ZodTypeAny, {
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
    metadata: z__default.ZodOptional<z__default.ZodRecord<z__default.ZodString, z__default.ZodAny>>;
}, "strip", z__default.ZodTypeAny, {
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
}>;
type DocumentData = z__default.infer<typeof DocumentDataSchema>;
/**
 * Document represents document content along with its metadata that can be embedded, indexed or
 * retrieved. Each document can contain multiple parts (for example text and an image)
 */
declare class Document implements DocumentData {
    content: Part[];
    metadata?: Record<string, any>;
    constructor(data: DocumentData);
    static fromText(text: string, metadata?: Record<string, any>): Document;
    /**
     * Concatenates all `text` parts present in the document with no delimiter.
     * @returns A string of all concatenated text parts.
     */
    text(): string;
    /**
     * Returns the first media part detected in the document. Useful for extracting
     * (for example) an image.
     * @returns The first detected `media` part in the document.
     */
    media(): {
        url: string;
        contentType?: string;
    } | null;
    toJSON(): DocumentData;
}

export { Document, type DocumentData, DocumentDataSchema, type MediaPart, MediaPartSchema, type Part, PartSchema, type TextPart, TextPartSchema };
