import { Action } from '@genkit-ai/core';
import * as z from 'zod';

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

declare const ATTR_PREFIX = "genkit";
declare const SPAN_STATE_ATTR: string;
declare const BaseDataPointSchema: z.ZodObject<{
    input: z.ZodUnknown;
    output: z.ZodOptional<z.ZodUnknown>;
    context: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
    reference: z.ZodOptional<z.ZodUnknown>;
    testCaseId: z.ZodOptional<z.ZodString>;
    traceIds: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
}, "strip", z.ZodTypeAny, {
    input?: unknown;
    output?: unknown;
    context?: unknown[] | undefined;
    reference?: unknown;
    testCaseId?: string | undefined;
    traceIds?: string[] | undefined;
}, {
    input?: unknown;
    output?: unknown;
    context?: unknown[] | undefined;
    reference?: unknown;
    testCaseId?: string | undefined;
    traceIds?: string[] | undefined;
}>;
declare const ScoreSchema: z.ZodObject<{
    score: z.ZodOptional<z.ZodUnion<[z.ZodNumber, z.ZodString, z.ZodBoolean]>>;
    error: z.ZodOptional<z.ZodString>;
    details: z.ZodOptional<z.ZodObject<{
        reasoning: z.ZodOptional<z.ZodString>;
    }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
        reasoning: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
        reasoning: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough">>>;
}, "strip", z.ZodTypeAny, {
    score?: string | number | boolean | undefined;
    error?: string | undefined;
    details?: z.objectOutputType<{
        reasoning: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}, {
    score?: string | number | boolean | undefined;
    error?: string | undefined;
    details?: z.objectInputType<{
        reasoning: z.ZodOptional<z.ZodString>;
    }, z.ZodTypeAny, "passthrough"> | undefined;
}>;
declare const EVALUATOR_METADATA_KEY_DISPLAY_NAME = "evaluatorDisplayName";
declare const EVALUATOR_METADATA_KEY_DEFINITION = "evaluatorDefinition";
declare const EVALUATOR_METADATA_KEY_IS_BILLED = "evaluatorIsBilled";
type Score = z.infer<typeof ScoreSchema>;
type BaseDataPoint = z.infer<typeof BaseDataPointSchema>;
type Dataset<DataPoint extends typeof BaseDataPointSchema = typeof BaseDataPointSchema> = Array<z.infer<DataPoint>>;
declare const EvalResponseSchema: z.ZodObject<{
    sampleIndex: z.ZodOptional<z.ZodNumber>;
    testCaseId: z.ZodOptional<z.ZodString>;
    traceId: z.ZodOptional<z.ZodString>;
    spanId: z.ZodOptional<z.ZodString>;
    evaluation: z.ZodObject<{
        score: z.ZodOptional<z.ZodUnion<[z.ZodNumber, z.ZodString, z.ZodBoolean]>>;
        error: z.ZodOptional<z.ZodString>;
        details: z.ZodOptional<z.ZodObject<{
            reasoning: z.ZodOptional<z.ZodString>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            reasoning: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            reasoning: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">>>;
    }, "strip", z.ZodTypeAny, {
        score?: string | number | boolean | undefined;
        error?: string | undefined;
        details?: z.objectOutputType<{
            reasoning: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }, {
        score?: string | number | boolean | undefined;
        error?: string | undefined;
        details?: z.objectInputType<{
            reasoning: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    evaluation: {
        score?: string | number | boolean | undefined;
        error?: string | undefined;
        details?: z.objectOutputType<{
            reasoning: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    };
    sampleIndex?: number | undefined;
    testCaseId?: string | undefined;
    traceId?: string | undefined;
    spanId?: string | undefined;
}, {
    evaluation: {
        score?: string | number | boolean | undefined;
        error?: string | undefined;
        details?: z.objectInputType<{
            reasoning: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    };
    sampleIndex?: number | undefined;
    testCaseId?: string | undefined;
    traceId?: string | undefined;
    spanId?: string | undefined;
}>;
type EvalResponse = z.infer<typeof EvalResponseSchema>;
declare const EvalResponsesSchema: z.ZodArray<z.ZodObject<{
    sampleIndex: z.ZodOptional<z.ZodNumber>;
    testCaseId: z.ZodOptional<z.ZodString>;
    traceId: z.ZodOptional<z.ZodString>;
    spanId: z.ZodOptional<z.ZodString>;
    evaluation: z.ZodObject<{
        score: z.ZodOptional<z.ZodUnion<[z.ZodNumber, z.ZodString, z.ZodBoolean]>>;
        error: z.ZodOptional<z.ZodString>;
        details: z.ZodOptional<z.ZodObject<{
            reasoning: z.ZodOptional<z.ZodString>;
        }, "passthrough", z.ZodTypeAny, z.objectOutputType<{
            reasoning: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">, z.objectInputType<{
            reasoning: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough">>>;
    }, "strip", z.ZodTypeAny, {
        score?: string | number | boolean | undefined;
        error?: string | undefined;
        details?: z.objectOutputType<{
            reasoning: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }, {
        score?: string | number | boolean | undefined;
        error?: string | undefined;
        details?: z.objectInputType<{
            reasoning: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    evaluation: {
        score?: string | number | boolean | undefined;
        error?: string | undefined;
        details?: z.objectOutputType<{
            reasoning: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    };
    sampleIndex?: number | undefined;
    testCaseId?: string | undefined;
    traceId?: string | undefined;
    spanId?: string | undefined;
}, {
    evaluation: {
        score?: string | number | boolean | undefined;
        error?: string | undefined;
        details?: z.objectInputType<{
            reasoning: z.ZodOptional<z.ZodString>;
        }, z.ZodTypeAny, "passthrough"> | undefined;
    };
    sampleIndex?: number | undefined;
    testCaseId?: string | undefined;
    traceId?: string | undefined;
    spanId?: string | undefined;
}>, "many">;
type EvalResponses = z.infer<typeof EvalResponsesSchema>;
type EvaluatorFn<DataPoint extends typeof BaseDataPointSchema = typeof BaseDataPointSchema, CustomOptions extends z.ZodTypeAny = z.ZodTypeAny> = (input: z.infer<DataPoint>, evaluatorOptions?: z.infer<CustomOptions>) => Promise<EvalResponse>;
type EvaluatorAction<DataPoint extends typeof BaseDataPointSchema = typeof BaseDataPointSchema, CustomOptions extends z.ZodTypeAny = z.ZodTypeAny> = Action<typeof EvalRequestSchema, typeof EvalResponsesSchema> & {
    __dataPointType?: DataPoint;
    __configSchema?: CustomOptions;
};
declare const EvalRequestSchema: z.ZodObject<{
    dataset: z.ZodArray<z.ZodObject<{
        input: z.ZodUnknown;
        output: z.ZodOptional<z.ZodUnknown>;
        context: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
        reference: z.ZodOptional<z.ZodUnknown>;
        testCaseId: z.ZodOptional<z.ZodString>;
        traceIds: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        input?: unknown;
        output?: unknown;
        context?: unknown[] | undefined;
        reference?: unknown;
        testCaseId?: string | undefined;
        traceIds?: string[] | undefined;
    }, {
        input?: unknown;
        output?: unknown;
        context?: unknown[] | undefined;
        reference?: unknown;
        testCaseId?: string | undefined;
        traceIds?: string[] | undefined;
    }>, "many">;
    options: z.ZodUnknown;
}, "strip", z.ZodTypeAny, {
    dataset: {
        input?: unknown;
        output?: unknown;
        context?: unknown[] | undefined;
        reference?: unknown;
        testCaseId?: string | undefined;
        traceIds?: string[] | undefined;
    }[];
    options?: unknown;
}, {
    dataset: {
        input?: unknown;
        output?: unknown;
        context?: unknown[] | undefined;
        reference?: unknown;
        testCaseId?: string | undefined;
        traceIds?: string[] | undefined;
    }[];
    options?: unknown;
}>;
/**
 * Creates evaluator action for the provided {@link EvaluatorFn} implementation.
 */
declare function defineEvaluator<DataPoint extends typeof BaseDataPointSchema = typeof BaseDataPointSchema, EvaluatorOptions extends z.ZodTypeAny = z.ZodTypeAny>(options: {
    name: string;
    displayName: string;
    definition: string;
    dataPointType?: DataPoint;
    configSchema?: EvaluatorOptions;
    isBilled?: boolean;
}, runner: EvaluatorFn<DataPoint, EvaluatorOptions>): EvaluatorAction<DataPoint, EvaluatorOptions>;
type EvaluatorArgument<DataPoint extends typeof BaseDataPointSchema = typeof BaseDataPointSchema, CustomOptions extends z.ZodTypeAny = z.ZodTypeAny> = string | EvaluatorAction<DataPoint, CustomOptions> | EvaluatorReference<CustomOptions>;
/**
 * A veneer for interacting with evaluators.
 */
declare function evaluate<DataPoint extends typeof BaseDataPointSchema = typeof BaseDataPointSchema, EvaluatorOptions extends z.ZodTypeAny = z.ZodTypeAny>(params: {
    evaluator: EvaluatorArgument<DataPoint, EvaluatorOptions>;
    dataset: Dataset<DataPoint>;
    options?: z.infer<EvaluatorOptions>;
}): Promise<EvalResponses>;
declare const EvaluatorInfoSchema: z.ZodObject<{
    /** Friendly label for this evaluator */
    label: z.ZodOptional<z.ZodString>;
    metrics: z.ZodArray<z.ZodString, "many">;
}, "strip", z.ZodTypeAny, {
    metrics: string[];
    label?: string | undefined;
}, {
    metrics: string[];
    label?: string | undefined;
}>;
type EvaluatorInfo = z.infer<typeof EvaluatorInfoSchema>;
interface EvaluatorReference<CustomOptions extends z.ZodTypeAny> {
    name: string;
    configSchema?: CustomOptions;
    info?: EvaluatorInfo;
}
/**
 * Helper method to configure a {@link EvaluatorReference} to a plugin.
 */
declare function evaluatorRef<CustomOptionsSchema extends z.ZodTypeAny = z.ZodTypeAny>(options: EvaluatorReference<CustomOptionsSchema>): EvaluatorReference<CustomOptionsSchema>;

export { ATTR_PREFIX, type BaseDataPoint, BaseDataPointSchema, type Dataset, EVALUATOR_METADATA_KEY_DEFINITION, EVALUATOR_METADATA_KEY_DISPLAY_NAME, EVALUATOR_METADATA_KEY_IS_BILLED, type EvalResponse, EvalResponseSchema, type EvalResponses, EvalResponsesSchema, type EvaluatorAction, type EvaluatorArgument, type EvaluatorInfo, EvaluatorInfoSchema, type EvaluatorReference, SPAN_STATE_ATTR, type Score, ScoreSchema, defineEvaluator, evaluate, evaluatorRef };
