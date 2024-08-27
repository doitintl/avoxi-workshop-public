import { TelemetryConfig } from './telemetryTypes.js';
export { TraceStoreExporter } from './tracing/exporter.js';
export { ATTR_PREFIX, SPAN_TYPE_ATTR, newTrace, runInNewSpan, setCustomMetadataAttribute, setCustomMetadataAttributes, spanMetadataAls, toDisplayPath, traceMetadataAls } from './tracing/instrumentation.js';
export { LocalFileTraceStore } from './tracing/localFileTraceStore.js';
export { GenkitSpanProcessorWrapper } from './tracing/processor.js';
import { TraceStore } from './tracing/types.js';
export { InstrumentationLibrarySchema, LinkSchema, PathMetadata, PathMetadataSchema, SpanContextSchema, SpanData, SpanDataSchema, SpanMetadata, SpanMetadataSchema, SpanStatusSchema, TimeEventSchema, TraceData, TraceDataSchema, TraceMetadata, TraceMetadataSchema, TraceQuery, TraceQueryResponse } from './tracing/types.js';
import '@opentelemetry/sdk-node';
import '@opentelemetry/core';
import '@opentelemetry/sdk-trace-base';
import '@opentelemetry/api';
import 'node:async_hooks';
import 'async-mutex';
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

/**
 * Enables trace spans to be written to the trace store.
 */
declare function enableTracingAndMetrics(telemetryConfig: TelemetryConfig, traceStore?: TraceStore, traceStoreOptions?: {
    processor?: 'batch' | 'simple';
}): void;
declare function cleanUpTracing(): Promise<void>;
/**
 * Flushes all configured span processors
 */
declare function flushTracing(): Promise<void>;

export { TraceStore, cleanUpTracing, enableTracingAndMetrics, flushTracing };
