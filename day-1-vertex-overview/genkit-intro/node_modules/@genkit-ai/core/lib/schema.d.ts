import { JSONSchemaType, ErrorObject } from 'ajv';
import { z } from 'zod';
export { z } from 'zod';
import { GenkitError } from './error.js';
import './statusTypes.js';

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

type JSONSchema = JSONSchemaType<any> | any;
interface ProvidedSchema {
    jsonSchema?: JSONSchema;
    schema?: z.ZodTypeAny;
}
declare class ValidationError extends GenkitError {
    constructor({ data, errors, schema, }: {
        data: any;
        errors: ValidationErrorDetail[];
        schema: JSONSchema;
    });
}
/**
 * Convertes a Zod schema into a JSON schema, utilizing an in-memory cache for known objects.
 * @param options Provide a json schema and/or zod schema. JSON schema has priority.
 * @returns A JSON schema.
 */
declare function toJsonSchema({ jsonSchema, schema, }: ProvidedSchema): JSONSchema | undefined;
interface ValidationErrorDetail {
    path: string;
    message: string;
}
type ValidationResponse = {
    valid: true;
    errors: never;
} | {
    valid: false;
    errors: ErrorObject[];
};
declare function validateSchema(data: unknown, options: ProvidedSchema): {
    valid: boolean;
    errors?: any[];
    schema: JSONSchema;
};
declare function parseSchema<T = unknown>(data: unknown, options: ProvidedSchema): T;
declare function defineSchema<T extends z.ZodTypeAny>(name: string, schema: T): T;
declare function defineJsonSchema(name: string, jsonSchema: JSONSchema): any;

export { type JSONSchema, type ProvidedSchema, ValidationError, type ValidationErrorDetail, type ValidationResponse, defineJsonSchema, defineSchema, parseSchema, toJsonSchema, validateSchema };
