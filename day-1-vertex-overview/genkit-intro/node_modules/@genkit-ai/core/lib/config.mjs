import {
  __async
} from "./chunk-XEFTB2OF.mjs";
import fs from "fs";
import path from "path";
import { LocalFileFlowStateStore } from "./localFileFlowStateStore.js";
import { logger } from "./logging.js";
import * as registry from "./registry.js";
import { enableTracingAndMetrics } from "./tracing.js";
import { LocalFileTraceStore } from "./tracing/localFileTraceStore.js";
export * from "./plugin.js";
let config;
class Config {
  constructor(options) {
    this.configuredEnvs = /* @__PURE__ */ new Set(["dev"]);
    this.options = options;
    this.telemetryConfig = () => __async(this, null, function* () {
      return {
        getConfig() {
          return {};
        }
      };
    });
    this.configure();
  }
  /**
   * Returns a flow state store instance for the running environment.
   * If no store is configured, will throw an error.
   */
  getFlowStateStore() {
    return __async(this, null, function* () {
      const flowStateStore = yield registry.lookupFlowStateStore(getCurrentEnv());
      if (!flowStateStore) {
        throw new Error("No flow store is configured.");
      }
      return flowStateStore;
    });
  }
  /**
   * Returns a trace store instance for the running environment.
   * If no store is configured, will return undefined.
   */
  getTraceStore() {
    return __async(this, null, function* () {
      return yield registry.lookupTraceStore(getCurrentEnv());
    });
  }
  /**
   * Returns the configuration for exporting Telemetry data for the current
   * environment.
   */
  getTelemetryConfig() {
    return this.telemetryConfig();
  }
  /**
   * Configures the system.
   */
  configure() {
    var _a, _b, _c;
    if (this.options.logLevel) {
      logger.setLogLevel(this.options.logLevel);
    }
    (_a = this.options.plugins) == null ? void 0 : _a.forEach((plugin) => {
      logger.debug(`Registering plugin ${plugin.name}...`);
      registry.registerPluginProvider(plugin.name, {
        name: plugin.name,
        initializer() {
          return __async(this, null, function* () {
            logger.info(`Initializing plugin ${plugin.name}:`);
            return yield plugin.initializer();
          });
        }
      });
    });
    if ((_b = this.options.telemetry) == null ? void 0 : _b.logger) {
      const loggerPluginName = this.options.telemetry.logger;
      logger.debug("Registering logging exporters...");
      logger.debug(`  - all environments: ${loggerPluginName}`);
      this.loggerConfig = () => __async(this, null, function* () {
        return this.resolveLoggerConfig(loggerPluginName);
      });
    }
    if ((_c = this.options.telemetry) == null ? void 0 : _c.instrumentation) {
      const telemetryPluginName = this.options.telemetry.instrumentation;
      logger.debug("Registering telemetry exporters...");
      logger.debug(`  - all environments: ${telemetryPluginName}`);
      this.telemetryConfig = () => __async(this, null, function* () {
        return this.resolveTelemetryConfig(telemetryPluginName);
      });
    }
    logger.debug("Registering flow state stores...");
    if (isDevEnv()) {
      registry.registerFlowStateStore(
        "dev",
        () => __async(this, null, function* () {
          return new LocalFileFlowStateStore();
        })
      );
      logger.debug("Registered dev flow state store.");
    }
    if (this.options.flowStateStore) {
      const flowStorePluginName = this.options.flowStateStore;
      logger.debug(`  - prod: ${flowStorePluginName}`);
      this.configuredEnvs.add("prod");
      registry.registerFlowStateStore(
        "prod",
        () => this.resolveFlowStateStore(flowStorePluginName)
      );
    }
    logger.debug("Registering trace stores...");
    if (isDevEnv()) {
      registry.registerTraceStore("dev", () => __async(this, null, function* () {
        return new LocalFileTraceStore();
      }));
      logger.debug("Registered dev trace store.");
    }
    if (this.options.traceStore) {
      const traceStorePluginName = this.options.traceStore;
      logger.debug(`  - prod: ${traceStorePluginName}`);
      this.configuredEnvs.add("prod");
      registry.registerTraceStore(
        "prod",
        () => this.resolveTraceStore(traceStorePluginName)
      );
      if (isDevEnv()) {
        logger.info(
          "In dev mode `traceStore` is defaulted to local file store."
        );
      }
    } else {
      logger.info(
        "`traceStore` is not specified in the config; Traces are not going to be persisted in prod."
      );
    }
  }
  /**
   * Sets up the tracing and logging as configured.
   *
   * Note: the logging configuration must come after tracing has been enabled to
   * ensure that all tracing instrumentations are applied.
   * See limitations described here:
   * https://github.com/open-telemetry/opentelemetry-js/tree/main/experimental/packages/opentelemetry-instrumentation#limitations
   */
  setupTracingAndLogging() {
    return __async(this, null, function* () {
      if (this.options.enableTracingAndMetrics) {
        enableTracingAndMetrics(
          yield this.getTelemetryConfig(),
          yield this.getTraceStore()
        );
      }
      if (this.loggerConfig) {
        logger.init(yield this.loggerConfig());
      }
    });
  }
  /**
   * Resolves flow state store provided by the specified plugin.
   */
  resolveFlowStateStore(pluginName) {
    return __async(this, null, function* () {
      let flowStoreId;
      if (pluginName.includes("/")) {
        const tokens = pluginName.split("/", 2);
        pluginName = tokens[0];
        flowStoreId = tokens[1];
      }
      const plugin = yield registry.initializePlugin(pluginName);
      let provider = plugin == null ? void 0 : plugin.flowStateStore;
      if (!provider) {
        throw new Error(
          "Unable to resolve provided `flowStateStore` for plugin: " + pluginName
        );
      }
      if (!Array.isArray(provider)) {
        provider = [provider];
      }
      if (provider.length === 1 && !flowStoreId) {
        return provider[0].value;
      }
      if (provider.length > 1 && !flowStoreId) {
        throw new Error(
          `Plugin ${pluginName} provides more than one flow state store implementation (${provider.map((p2) => p2.id).join(", ")}), please specify the flow state store id (e.g. "${pluginName}/${provider[0].id}")`
        );
      }
      const p = provider.find((p2) => p2.id === flowStoreId);
      if (!p) {
        throw new Error(
          `Plugin ${pluginName} does not provide flow state store ${flowStoreId}`
        );
      }
      return p.value;
    });
  }
  /**
   * Resolves trace store provided by the specified plugin.
   */
  resolveTraceStore(pluginName) {
    return __async(this, null, function* () {
      let traceStoreId;
      if (pluginName.includes("/")) {
        const tokens = pluginName.split("/", 2);
        pluginName = tokens[0];
        traceStoreId = tokens[1];
      }
      const plugin = yield registry.initializePlugin(pluginName);
      let provider = plugin == null ? void 0 : plugin.traceStore;
      if (!provider) {
        throw new Error(
          "Unable to resolve provided `traceStore` for plugin: " + pluginName
        );
      }
      if (!Array.isArray(provider)) {
        provider = [provider];
      }
      if (provider.length === 1 && !traceStoreId) {
        return provider[0].value;
      }
      if (provider.length > 1 && !traceStoreId) {
        throw new Error(
          `Plugin ${pluginName} provides more than one trace store implementation (${provider.map((p2) => p2.id).join(", ")}), please specify the trace store id (e.g. "${pluginName}/${provider[0].id}")`
        );
      }
      const p = provider.find((p2) => p2.id === traceStoreId);
      if (!p) {
        throw new Error(
          `Plugin ${pluginName} does not provide trace store ${traceStoreId}`
        );
      }
      return p.value;
    });
  }
  /**
   * Resolves the telemetry configuration provided by the specified plugin.
   */
  resolveTelemetryConfig(pluginName) {
    return __async(this, null, function* () {
      var _a;
      const plugin = yield registry.initializePlugin(pluginName);
      const provider = (_a = plugin == null ? void 0 : plugin.telemetry) == null ? void 0 : _a.instrumentation;
      if (!provider) {
        throw new Error(
          "Unable to resolve provider `telemetry.instrumentation` for plugin: " + pluginName
        );
      }
      return provider.value;
    });
  }
  /**
   * Resolves the logging configuration provided by the specified plugin.
   */
  resolveLoggerConfig(pluginName) {
    return __async(this, null, function* () {
      var _a;
      const plugin = yield registry.initializePlugin(pluginName);
      const provider = (_a = plugin == null ? void 0 : plugin.telemetry) == null ? void 0 : _a.logger;
      if (!provider) {
        throw new Error(
          "Unable to resolve provider `telemetry.logger` for plugin: " + pluginName
        );
      }
      return provider.value;
    });
  }
}
function configureGenkit(options) {
  if (config) {
    logger.warn("configureGenkit was already called");
  }
  config = new Config(options);
  config.setupTracingAndLogging();
  return config;
}
function initializeGenkit(cfg) {
  if (config || cfg) {
    return;
  }
  const configPath = findGenkitConfig();
  if (!configPath) {
    throw Error(
      "Unable to find genkit.config.js in any of the parent directories."
    );
  }
  require(configPath);
}
function getCurrentEnv() {
  return process.env.GENKIT_ENV || "prod";
}
function isDevEnv() {
  return getCurrentEnv() === "dev";
}
function findGenkitConfig() {
  var _a;
  let current = (_a = require == null ? void 0 : require.main) == null ? void 0 : _a.filename;
  if (!current) {
    throw new Error("Unable to resolve package root.");
  }
  while (path.resolve(current, "..") !== current) {
    if (fs.existsSync(path.resolve(current, "genkit.config.js"))) {
      return path.resolve(current, "genkit.config.js");
    }
    current = path.resolve(current, "..");
  }
  return void 0;
}
function __hardResetConfigForTesting() {
  config = void 0;
}
export {
  __hardResetConfigForTesting,
  config,
  configureGenkit,
  getCurrentEnv,
  initializeGenkit,
  isDevEnv
};
//# sourceMappingURL=config.mjs.map