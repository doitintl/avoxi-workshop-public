import { FlowStateStore } from './flowTypes.js';
import { f as PluginProvider } from './action-CnIb9v86.js';
export { I as InitializedPlugin, h as Plugin, P as Provider, i as genkitPlugin } from './action-CnIb9v86.js';
import { TelemetryOptions, TelemetryConfig } from './telemetryTypes.js';
import { TraceStore } from './tracing/types.js';
import 'zod';
import 'json-schema';
import './statusTypes.js';
import './schema.js';
import 'ajv';
import './error.js';
import '@opentelemetry/sdk-node';

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

declare let config: Config;
interface ConfigOptions {
    plugins?: PluginProvider[];
    traceStore?: string;
    flowStateStore?: string;
    enableTracingAndMetrics?: boolean;
    logLevel?: 'error' | 'warn' | 'info' | 'debug';
    promptDir?: string;
    telemetry?: TelemetryOptions;
    defaultModel?: {
        name: string | {
            name: string;
        };
        config?: Record<string, any>;
    };
}
declare class Config {
    /** Developer-configured options. */
    readonly options: ConfigOptions;
    readonly configuredEnvs: Set<string>;
    private telemetryConfig;
    private loggerConfig?;
    constructor(options: ConfigOptions);
    /**
     * Returns a flow state store instance for the running environment.
     * If no store is configured, will throw an error.
     */
    getFlowStateStore(): Promise<FlowStateStore>;
    /**
     * Returns a trace store instance for the running environment.
     * If no store is configured, will return undefined.
     */
    getTraceStore(): Promise<TraceStore | undefined>;
    /**
     * Returns the configuration for exporting Telemetry data for the current
     * environment.
     */
    getTelemetryConfig(): Promise<TelemetryConfig>;
    /**
     * Configures the system.
     */
    private configure;
    /**
     * Sets up the tracing and logging as configured.
     *
     * Note: the logging configuration must come after tracing has been enabled to
     * ensure that all tracing instrumentations are applied.
     * See limitations described here:
     * https://github.com/open-telemetry/opentelemetry-js/tree/main/experimental/packages/opentelemetry-instrumentation#limitations
     */
    setupTracingAndLogging(): Promise<void>;
    /**
     * Resolves flow state store provided by the specified plugin.
     */
    private resolveFlowStateStore;
    /**
     * Resolves trace store provided by the specified plugin.
     */
    private resolveTraceStore;
    /**
     * Resolves the telemetry configuration provided by the specified plugin.
     */
    private resolveTelemetryConfig;
    /**
     * Resolves the logging configuration provided by the specified plugin.
     */
    private resolveLoggerConfig;
}
/**
 * Configures Genkit with a set of options. This should be called from `genkit.configig.js`.
 */
declare function configureGenkit(options: ConfigOptions): Config;
/**
 * Locates `genkit.config.js` and loads the file so that the config can be registered.
 */
declare function initializeGenkit(cfg?: Config): void;
/**
 * @returns The current environment that the app code is running in.
 */
declare function getCurrentEnv(): string;
/** Whether current env is `dev`. */
declare function isDevEnv(): boolean;
declare function __hardResetConfigForTesting(): void;

export { type ConfigOptions, PluginProvider, __hardResetConfigForTesting, config, configureGenkit, getCurrentEnv, initializeGenkit, isDevEnv };
