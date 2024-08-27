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
 * Interrupt error is an internal error thrown by flow states to interrupt execution of the step.
 */
declare class InterruptError extends Error {
}
/**
 * Extracts error message from the given error object, or if input is not an error then just turn the error into a string.
 */
declare function getErrorMessage(e: any): string;
/**
 * Extracts stack trace from the given error object, or if input is not an error then returns undefined.
 */
declare function getErrorStack(e: any): string | undefined;
/**
 * Exception thrown when flow is not found in the flow state store.
 */
declare class FlowNotFoundError extends Error {
    constructor(msg: string);
}
/**
 * Exception thrown when flow execution is not completed yet.
 */
declare class FlowStillRunningError extends Error {
    readonly flowId: string;
    constructor(flowId: string);
}
/**
 * Exception thrown when flow execution resulted in an error.
 */
declare class FlowExecutionError extends Error {
    readonly flowId: string;
    readonly originalMessage: string;
    readonly originalStacktrace?: any;
    constructor(flowId: string, originalMessage: string, originalStacktrace?: any);
}

export { FlowExecutionError, FlowNotFoundError, FlowStillRunningError, InterruptError, getErrorMessage, getErrorStack };
