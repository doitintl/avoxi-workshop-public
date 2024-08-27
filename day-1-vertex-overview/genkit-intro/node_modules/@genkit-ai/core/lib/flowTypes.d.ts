import { z } from 'zod';

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

interface FlowStateQuery {
    limit?: number;
    continuationToken?: string;
}
interface FlowStateQueryResponse {
    flowStates: FlowState[];
    continuationToken?: string;
}
/**
 * Flow state store persistence interface.
 */
interface FlowStateStore {
    save(id: string, state: FlowState): Promise<void>;
    load(id: string): Promise<FlowState | undefined>;
    list(query?: FlowStateQuery): Promise<FlowStateQueryResponse>;
}
declare const FlowStateExecutionSchema: z.ZodObject<{
    startTime: z.ZodOptional<z.ZodNumber>;
    endTime: z.ZodOptional<z.ZodNumber>;
    traceIds: z.ZodArray<z.ZodString, "many">;
}, "strip", z.ZodTypeAny, {
    traceIds: string[];
    startTime?: number | undefined;
    endTime?: number | undefined;
}, {
    traceIds: string[];
    startTime?: number | undefined;
    endTime?: number | undefined;
}>;
type FlowStateExecution = z.infer<typeof FlowStateExecutionSchema>;
declare const FlowResponseSchema: z.ZodObject<{
    response: z.ZodNullable<z.ZodUnknown>;
}, "strip", z.ZodTypeAny, {
    response?: unknown;
}, {
    response?: unknown;
}>;
declare const FlowErrorSchema: z.ZodObject<{
    error: z.ZodOptional<z.ZodString>;
    stacktrace: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    error?: string | undefined;
    stacktrace?: string | undefined;
}, {
    error?: string | undefined;
    stacktrace?: string | undefined;
}>;
type FlowError = z.infer<typeof FlowErrorSchema>;
declare const FlowResultSchema: z.ZodIntersection<z.ZodObject<{
    response: z.ZodNullable<z.ZodUnknown>;
}, "strip", z.ZodTypeAny, {
    response?: unknown;
}, {
    response?: unknown;
}>, z.ZodObject<{
    error: z.ZodOptional<z.ZodString>;
    stacktrace: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    error?: string | undefined;
    stacktrace?: string | undefined;
}, {
    error?: string | undefined;
    stacktrace?: string | undefined;
}>>;
/**
 * Flow Operation, modelled after:
 * https://cloud.google.com/service-infrastructure/docs/service-management/reference/rpc/google.longrunning#google.longrunning.Operation
 */
declare const OperationSchema: z.ZodObject<{
    name: z.ZodString;
    metadata: z.ZodOptional<z.ZodAny>;
    done: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    result: z.ZodOptional<z.ZodIntersection<z.ZodObject<{
        response: z.ZodNullable<z.ZodUnknown>;
    }, "strip", z.ZodTypeAny, {
        response?: unknown;
    }, {
        response?: unknown;
    }>, z.ZodObject<{
        error: z.ZodOptional<z.ZodString>;
        stacktrace: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        error?: string | undefined;
        stacktrace?: string | undefined;
    }, {
        error?: string | undefined;
        stacktrace?: string | undefined;
    }>>>;
    blockedOnStep: z.ZodOptional<z.ZodObject<{
        name: z.ZodString;
        schema: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        schema?: string | undefined;
    }, {
        name: string;
        schema?: string | undefined;
    }>>;
}, "strip", z.ZodTypeAny, {
    name: string;
    done: boolean;
    metadata?: any;
    result?: ({
        response?: unknown;
    } & {
        error?: string | undefined;
        stacktrace?: string | undefined;
    }) | undefined;
    blockedOnStep?: {
        name: string;
        schema?: string | undefined;
    } | undefined;
}, {
    name: string;
    metadata?: any;
    done?: boolean | undefined;
    result?: ({
        response?: unknown;
    } & {
        error?: string | undefined;
        stacktrace?: string | undefined;
    }) | undefined;
    blockedOnStep?: {
        name: string;
        schema?: string | undefined;
    } | undefined;
}>;
type Operation = z.infer<typeof OperationSchema>;
/**
 * Defines the format for flow state. This is the format used for persisting the state in
 * the {@link FlowStateStore}.
 */
declare const FlowStateSchema: z.ZodObject<{
    name: z.ZodOptional<z.ZodString>;
    flowId: z.ZodString;
    input: z.ZodUnknown;
    startTime: z.ZodNumber;
    cache: z.ZodRecord<z.ZodString, z.ZodObject<{
        value: z.ZodOptional<z.ZodAny>;
        empty: z.ZodOptional<z.ZodLiteral<true>>;
    }, "strip", z.ZodTypeAny, {
        value?: any;
        empty?: true | undefined;
    }, {
        value?: any;
        empty?: true | undefined;
    }>>;
    eventsTriggered: z.ZodRecord<z.ZodString, z.ZodAny>;
    blockedOnStep: z.ZodNullable<z.ZodObject<{
        name: z.ZodString;
        schema: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        schema?: string | undefined;
    }, {
        name: string;
        schema?: string | undefined;
    }>>;
    operation: z.ZodObject<{
        name: z.ZodString;
        metadata: z.ZodOptional<z.ZodAny>;
        done: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        result: z.ZodOptional<z.ZodIntersection<z.ZodObject<{
            response: z.ZodNullable<z.ZodUnknown>;
        }, "strip", z.ZodTypeAny, {
            response?: unknown;
        }, {
            response?: unknown;
        }>, z.ZodObject<{
            error: z.ZodOptional<z.ZodString>;
            stacktrace: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            error?: string | undefined;
            stacktrace?: string | undefined;
        }, {
            error?: string | undefined;
            stacktrace?: string | undefined;
        }>>>;
        blockedOnStep: z.ZodOptional<z.ZodObject<{
            name: z.ZodString;
            schema: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            schema?: string | undefined;
        }, {
            name: string;
            schema?: string | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        done: boolean;
        metadata?: any;
        result?: ({
            response?: unknown;
        } & {
            error?: string | undefined;
            stacktrace?: string | undefined;
        }) | undefined;
        blockedOnStep?: {
            name: string;
            schema?: string | undefined;
        } | undefined;
    }, {
        name: string;
        metadata?: any;
        done?: boolean | undefined;
        result?: ({
            response?: unknown;
        } & {
            error?: string | undefined;
            stacktrace?: string | undefined;
        }) | undefined;
        blockedOnStep?: {
            name: string;
            schema?: string | undefined;
        } | undefined;
    }>;
    traceContext: z.ZodOptional<z.ZodString>;
    executions: z.ZodArray<z.ZodObject<{
        startTime: z.ZodOptional<z.ZodNumber>;
        endTime: z.ZodOptional<z.ZodNumber>;
        traceIds: z.ZodArray<z.ZodString, "many">;
    }, "strip", z.ZodTypeAny, {
        traceIds: string[];
        startTime?: number | undefined;
        endTime?: number | undefined;
    }, {
        traceIds: string[];
        startTime?: number | undefined;
        endTime?: number | undefined;
    }>, "many">;
}, "strip", z.ZodTypeAny, {
    flowId: string;
    startTime: number;
    cache: Record<string, {
        value?: any;
        empty?: true | undefined;
    }>;
    eventsTriggered: Record<string, any>;
    blockedOnStep: {
        name: string;
        schema?: string | undefined;
    } | null;
    operation: {
        name: string;
        done: boolean;
        metadata?: any;
        result?: ({
            response?: unknown;
        } & {
            error?: string | undefined;
            stacktrace?: string | undefined;
        }) | undefined;
        blockedOnStep?: {
            name: string;
            schema?: string | undefined;
        } | undefined;
    };
    executions: {
        traceIds: string[];
        startTime?: number | undefined;
        endTime?: number | undefined;
    }[];
    name?: string | undefined;
    input?: unknown;
    traceContext?: string | undefined;
}, {
    flowId: string;
    startTime: number;
    cache: Record<string, {
        value?: any;
        empty?: true | undefined;
    }>;
    eventsTriggered: Record<string, any>;
    blockedOnStep: {
        name: string;
        schema?: string | undefined;
    } | null;
    operation: {
        name: string;
        metadata?: any;
        done?: boolean | undefined;
        result?: ({
            response?: unknown;
        } & {
            error?: string | undefined;
            stacktrace?: string | undefined;
        }) | undefined;
        blockedOnStep?: {
            name: string;
            schema?: string | undefined;
        } | undefined;
    };
    executions: {
        traceIds: string[];
        startTime?: number | undefined;
        endTime?: number | undefined;
    }[];
    name?: string | undefined;
    input?: unknown;
    traceContext?: string | undefined;
}>;
type FlowState = z.infer<typeof FlowStateSchema>;

export { type FlowError, FlowErrorSchema, FlowResponseSchema, FlowResultSchema, type FlowState, type FlowStateExecution, FlowStateExecutionSchema, type FlowStateQuery, type FlowStateQueryResponse, FlowStateSchema, type FlowStateStore, type Operation, OperationSchema };
