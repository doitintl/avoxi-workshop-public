import { JSONSchema7 } from 'json-schema';
import * as z from 'zod';
import { z as z$1 } from 'zod';
import './statusTypes.js';
import { FlowStateStore } from './flowTypes.js';
import { TelemetryConfig, LoggerConfig } from './telemetryTypes.js';
import { TraceStore } from './tracing/types.js';
import { JSONSchema } from './schema.js';

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

interface Provider<T> {
    id: string;
    value: T;
}
interface PluginProvider {
    name: string;
    initializer: () => InitializedPlugin | void | Promise<InitializedPlugin | void>;
}
interface InitializedPlugin {
    models?: Action<z$1.ZodTypeAny, z$1.ZodTypeAny>[];
    retrievers?: Action<z$1.ZodTypeAny, z$1.ZodTypeAny>[];
    embedders?: Action<z$1.ZodTypeAny, z$1.ZodTypeAny>[];
    indexers?: Action<z$1.ZodTypeAny, z$1.ZodTypeAny>[];
    evaluators?: Action<z$1.ZodTypeAny, z$1.ZodTypeAny>[];
    flowStateStore?: Provider<FlowStateStore> | Provider<FlowStateStore>[];
    traceStore?: Provider<TraceStore> | Provider<TraceStore>[];
    telemetry?: {
        instrumentation?: Provider<TelemetryConfig>;
        logger?: Provider<LoggerConfig>;
    };
}
type PluginInit = (...args: any[]) => InitializedPlugin | void | Promise<InitializedPlugin | void>;
type Plugin<T extends any[]> = (...args: T) => PluginProvider;
/**
 * Defines a Genkit plugin.
 */
declare function genkitPlugin<T extends PluginInit>(pluginName: string, initFn: T): Plugin<Parameters<T>>;

type AsyncProvider<T> = () => Promise<T>;
/**
 * Type of a runnable action.
 */
type ActionType = 'custom' | 'retriever' | 'indexer' | 'embedder' | 'evaluator' | 'flow' | 'model' | 'prompt' | 'tool';
/**
 * Looks up a registry key (action type and key) in the registry.
 */
declare function lookupAction<I extends z.ZodTypeAny, O extends z.ZodTypeAny, R extends Action<I, O>>(key: string): Promise<R>;
/**
 * Registers an action in the registry.
 */
declare function registerAction<I extends z.ZodTypeAny, O extends z.ZodTypeAny>(type: ActionType, action: Action<I, O>): void;
type ActionsRecord = Record<string, Action<z.ZodTypeAny, z.ZodTypeAny>>;
/**
 * Returns all actions in the registry.
 */
declare function listActions(): Promise<ActionsRecord>;
/**
 * Registers a trace store provider for the given environment.
 */
declare function registerTraceStore(env: string, traceStoreProvider: AsyncProvider<TraceStore>): void;
/**
 * Looks up the trace store for the given environment.
 */
declare function lookupTraceStore(env: string): Promise<TraceStore | undefined>;
/**
 * Registers a flow state store provider for the given environment.
 */
declare function registerFlowStateStore(env: string, flowStateStoreProvider: AsyncProvider<FlowStateStore>): void;
/**
 * Looks up the flow state store for the given environment.
 */
declare function lookupFlowStateStore(env: string): Promise<FlowStateStore | undefined>;
/**
 * Registers a flow state store for the given environment.
 */
declare function registerPluginProvider(name: string, provider: PluginProvider): void;
declare function lookupPlugin(name: string): PluginProvider;
/**
 *
 */
declare function initializePlugin(name: string): Promise<void | InitializedPlugin>;
declare function registerSchema(name: string, data: {
    schema?: z.ZodTypeAny;
    jsonSchema?: JSONSchema;
}): void;
declare function lookupSchema(name: string): {
    schema?: z.ZodTypeAny | undefined;
    jsonSchema?: any;
};
declare function __hardResetRegistryForTesting(): void;

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

interface ActionMetadata<I extends z.ZodTypeAny, O extends z.ZodTypeAny, M extends Record<string, any> = Record<string, any>> {
    actionType?: ActionType;
    name: string;
    description?: string;
    inputSchema?: I;
    inputJsonSchema?: JSONSchema7;
    outputSchema?: O;
    outputJsonSchema?: JSONSchema7;
    metadata?: M;
}
type Action<I extends z.ZodTypeAny = z.ZodTypeAny, O extends z.ZodTypeAny = z.ZodTypeAny, M extends Record<string, any> = Record<string, any>> = ((input: z.infer<I>) => Promise<z.infer<O>>) & {
    __action: ActionMetadata<I, O, M>;
};
type SideChannelData = Record<string, any>;
type ActionParams<I extends z.ZodTypeAny, O extends z.ZodTypeAny, M extends Record<string, any> = Record<string, any>> = {
    name: string | {
        pluginId: string;
        actionId: string;
    };
    description?: string;
    inputSchema?: I;
    inputJsonSchema?: JSONSchema7;
    outputSchema?: O;
    outputJsonSchema?: JSONSchema7;
    metadata?: M;
    use?: Middleware<z.infer<I>, z.infer<O>>[];
};
interface Middleware<I = any, O = any> {
    (req: I, next: (req?: I) => Promise<O>): Promise<O>;
}
declare function actionWithMiddleware<I extends z.ZodTypeAny, O extends z.ZodTypeAny, M extends Record<string, any> = Record<string, any>>(action: Action<I, O, M>, middleware: Middleware<z.infer<I>, z.infer<O>>[]): Action<I, O, M>;
/**
 * Creates an action with the provided config.
 */
declare function action<I extends z.ZodTypeAny, O extends z.ZodTypeAny, M extends Record<string, any> = Record<string, any>>(config: ActionParams<I, O, M>, fn: (input: z.infer<I>) => Promise<z.infer<O>>): Action<I, O>;
/**
 * Defines an action with the given config and registers it in the registry.
 */
declare function defineAction<I extends z.ZodTypeAny, O extends z.ZodTypeAny, M extends Record<string, any> = Record<string, any>>(config: ActionParams<I, O, M> & {
    actionType: ActionType;
}, fn: (input: z.infer<I>) => Promise<z.infer<O>>): Action<I, O>;
type StreamingCallback<T> = (chunk: T) => void;
/**
 * Executes provided function with streaming callback in async local storage which can be retrieved
 * using {@link getStreamingCallback}.
 */
declare function runWithStreamingCallback<S, O>(streamingCallback: StreamingCallback<S> | undefined, fn: () => O): O;
/**
 * Retrieves the {@link StreamingCallback} previously set by {@link runWithStreamingCallback}
 */
declare function getStreamingCallback<S>(): StreamingCallback<S> | undefined;

export { type ActionMetadata as A, type InitializedPlugin as I, type Middleware as M, type Provider as P, type SideChannelData as S, __hardResetRegistryForTesting as _, type Action as a, actionWithMiddleware as b, action as c, defineAction as d, type StreamingCallback as e, type PluginProvider as f, getStreamingCallback as g, type Plugin as h, genkitPlugin as i, type AsyncProvider as j, type ActionType as k, lookupAction as l, registerAction as m, listActions as n, registerTraceStore as o, lookupTraceStore as p, registerFlowStateStore as q, runWithStreamingCallback as r, lookupFlowStateStore as s, registerPluginProvider as t, lookupPlugin as u, initializePlugin as v, registerSchema as w, lookupSchema as x };
