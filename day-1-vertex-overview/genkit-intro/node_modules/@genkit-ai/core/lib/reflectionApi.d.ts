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

declare const RunActionResponseSchema: z__default.ZodObject<{
    result: z__default.ZodOptional<z__default.ZodUnknown>;
    error: z__default.ZodOptional<z__default.ZodUnknown>;
    telemetry: z__default.ZodOptional<z__default.ZodObject<{
        traceId: z__default.ZodOptional<z__default.ZodString>;
    }, "strip", z__default.ZodTypeAny, {
        traceId?: string | undefined;
    }, {
        traceId?: string | undefined;
    }>>;
}, "strip", z__default.ZodTypeAny, {
    result?: unknown;
    error?: unknown;
    telemetry?: {
        traceId?: string | undefined;
    } | undefined;
}, {
    result?: unknown;
    error?: unknown;
    telemetry?: {
        traceId?: string | undefined;
    } | undefined;
}>;
type RunActionResponse = z__default.infer<typeof RunActionResponseSchema>;
/**
 * Starts a Reflection API that will be used by the Runner to call and control actions and flows.
 * @param port port on which to listen
 */
declare function startReflectionApi(port?: number | undefined): Promise<void>;

export { type RunActionResponse, RunActionResponseSchema, startReflectionApi };
