import { Action, JSONSchema7 } from '@genkit-ai/core';
import z__default from 'zod';
import { ToolDefinition } from './model.js';

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

type ToolAction<I extends z__default.ZodTypeAny = z__default.ZodTypeAny, O extends z__default.ZodTypeAny = z__default.ZodTypeAny> = Action<I, O> & {
    __action: {
        metadata: {
            type: 'tool';
        };
    };
};
type ToolArgument<I extends z__default.ZodTypeAny = z__default.ZodTypeAny, O extends z__default.ZodTypeAny = z__default.ZodTypeAny> = string | ToolAction<I, O> | Action<I, O> | ToolDefinition;
declare function asTool<I extends z__default.ZodTypeAny, O extends z__default.ZodTypeAny>(action: Action<I, O>): ToolAction<I, O>;
declare function resolveTools<O extends z__default.ZodTypeAny = z__default.ZodTypeAny, CustomOptions extends z__default.ZodTypeAny = z__default.ZodTypeAny>(tools?: ToolArgument[]): Promise<ToolAction[]>;
declare function toToolDefinition(tool: Action<z__default.ZodTypeAny, z__default.ZodTypeAny>): ToolDefinition;
declare function defineTool<I extends z__default.ZodTypeAny, O extends z__default.ZodTypeAny>({ name, description, inputSchema, inputJsonSchema, outputSchema, outputJsonSchema, metadata, }: {
    name: string;
    description: string;
    inputSchema?: I;
    inputJsonSchema?: JSONSchema7;
    outputSchema?: O;
    outputJsonSchema?: JSONSchema7;
    metadata?: Record<string, any>;
}, fn: (input: z__default.infer<I>) => Promise<z__default.infer<O>>): ToolAction<I, O>;

export { type ToolAction, type ToolArgument, asTool, defineTool, resolveTools, toToolDefinition };
