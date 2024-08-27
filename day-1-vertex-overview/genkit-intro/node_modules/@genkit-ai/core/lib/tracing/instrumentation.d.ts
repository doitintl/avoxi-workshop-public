import { Link, Span } from '@opentelemetry/api';
import { AsyncLocalStorage } from 'node:async_hooks';
import { SpanMetadata } from './types.js';
import 'zod';

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

declare const spanMetadataAls: AsyncLocalStorage<{
    name: string;
    state?: "success" | "error" | undefined;
    input?: any;
    output?: any;
    isRoot?: boolean | undefined;
    metadata?: Record<string, string> | undefined;
    path?: string | undefined;
}>;
declare const traceMetadataAls: AsyncLocalStorage<{
    timestamp: number;
    flowName?: string | undefined;
    paths?: Set<{
        path: string;
        status: string;
        latency: number;
        error?: string | undefined;
    }> | undefined;
}>;
declare const ATTR_PREFIX = "genkit";
declare const SPAN_TYPE_ATTR: string;
/**
 *
 */
declare function newTrace<T>(opts: {
    name: string;
    labels?: Record<string, string>;
    links?: Link[];
}, fn: (metadata: SpanMetadata, rootSpan: Span) => Promise<T>): Promise<T>;
/**
 *
 */
declare function runInNewSpan<T>(opts: {
    metadata: SpanMetadata;
    labels?: Record<string, string>;
    links?: Link[];
}, fn: (metadata: SpanMetadata, otSpan: Span, isRoot: boolean) => Promise<T>): Promise<T>;
/**
 * Sets provided attribute value in the current span.
 */
declare function setCustomMetadataAttribute(key: string, value: string): void;
/**
 * Sets provided attribute values in the current span.
 */
declare function setCustomMetadataAttributes(values: Record<string, string>): void;
/** Converts a fully annotated path to a friendly display version for logs */
declare function toDisplayPath(path: string): string;

export { ATTR_PREFIX, SPAN_TYPE_ATTR, newTrace, runInNewSpan, setCustomMetadataAttribute, setCustomMetadataAttributes, spanMetadataAls, toDisplayPath, traceMetadataAls };
