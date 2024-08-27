import { Document } from '../document.mjs';
import { MediaPart, ModelMiddleware, ModelInfo } from '../model.mjs';
import 'zod';
import '@genkit-ai/core';

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

/**
 * Preprocess a GenerateRequest to download referenced http(s) media URLs and
 * inline them as data URIs.
 */
declare function downloadRequestMedia(options?: {
    maxBytes?: number;
    filter?: (part: MediaPart) => boolean;
}): ModelMiddleware;
/**
 * Validates that a GenerateRequest does not include unsupported features.
 */
declare function validateSupport(options: {
    name: string;
    supports?: ModelInfo['supports'];
}): ModelMiddleware;
declare function conformOutput(): ModelMiddleware;
/**
 * Provide a simulated system prompt for models that don't support it natively.
 */
declare function simulateSystemPrompt(options?: {
    preface: string;
    acknowledgement: string;
}): ModelMiddleware;
interface AugmentWithContextOptions {
    /** Preceding text to place before the rendered context documents. */
    preface?: string | null;
    /** A function to render a document into a text part to be included in the message. */
    itemTemplate?: (d: Document, options?: AugmentWithContextOptions) => string;
    /** The metadata key to use for citation reference. Pass `null` to provide no citations. */
    citationKey?: string | null;
}
declare const CONTEXT_PREFACE = "\n\nUse the following information to complete your task:\n\n";
declare function augmentWithContext(options?: AugmentWithContextOptions): ModelMiddleware;

export { type AugmentWithContextOptions, CONTEXT_PREFACE, augmentWithContext, conformOutput, downloadRequestMedia, simulateSystemPrompt, validateSupport };
