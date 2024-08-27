"use strict";
var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __reExport = (target, mod, secondTarget) => (__copyProps(target, mod, "default"), secondTarget && __copyProps(secondTarget, mod, "default"));
var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
  // If the importer is in node compatibility mode or this is not an ESM
  // file that has been converted to a CommonJS file using a Babel-
  // compatible transform (i.e. "__esModule" has not been set), then set
  // "default" to the CommonJS "module.exports" for node compatibility.
  isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
  mod
));
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);
var __async = (__this, __arguments, generator) => {
  return new Promise((resolve, reject) => {
    var fulfilled = (value) => {
      try {
        step(generator.next(value));
      } catch (e) {
        reject(e);
      }
    };
    var rejected = (value) => {
      try {
        step(generator.throw(value));
      } catch (e) {
        reject(e);
      }
    };
    var step = (x) => x.done ? resolve(x.value) : Promise.resolve(x.value).then(fulfilled, rejected);
    step((generator = generator.apply(__this, __arguments)).next());
  });
};
var config_exports = {};
__export(config_exports, {
  __hardResetConfigForTesting: () => __hardResetConfigForTesting,
  config: () => config,
  configureGenkit: () => configureGenkit,
  getCurrentEnv: () => getCurrentEnv,
  initializeGenkit: () => initializeGenkit,
  isDevEnv: () => isDevEnv
});
module.exports = __toCommonJS(config_exports);
var import_fs = __toESM(require("fs"));
var import_path = __toESM(require("path"));
var import_localFileFlowStateStore = require("./localFileFlowStateStore.js");
var import_logging = require("./logging.js");
var registry = __toESM(require("./registry.js"));
var import_tracing = require("./tracing.js");
var import_localFileTraceStore = require("./tracing/localFileTraceStore.js");
__reExport(config_exports, require("./plugin.js"), module.exports);
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
      import_logging.logger.setLogLevel(this.options.logLevel);
    }
    (_a = this.options.plugins) == null ? void 0 : _a.forEach((plugin) => {
      import_logging.logger.debug(`Registering plugin ${plugin.name}...`);
      registry.registerPluginProvider(plugin.name, {
        name: plugin.name,
        initializer() {
          return __async(this, null, function* () {
            import_logging.logger.info(`Initializing plugin ${plugin.name}:`);
            return yield plugin.initializer();
          });
        }
      });
    });
    if ((_b = this.options.telemetry) == null ? void 0 : _b.logger) {
      const loggerPluginName = this.options.telemetry.logger;
      import_logging.logger.debug("Registering logging exporters...");
      import_logging.logger.debug(`  - all environments: ${loggerPluginName}`);
      this.loggerConfig = () => __async(this, null, function* () {
        return this.resolveLoggerConfig(loggerPluginName);
      });
    }
    if ((_c = this.options.telemetry) == null ? void 0 : _c.instrumentation) {
      const telemetryPluginName = this.options.telemetry.instrumentation;
      import_logging.logger.debug("Registering telemetry exporters...");
      import_logging.logger.debug(`  - all environments: ${telemetryPluginName}`);
      this.telemetryConfig = () => __async(this, null, function* () {
        return this.resolveTelemetryConfig(telemetryPluginName);
      });
    }
    import_logging.logger.debug("Registering flow state stores...");
    if (isDevEnv()) {
      registry.registerFlowStateStore(
        "dev",
        () => __async(this, null, function* () {
          return new import_localFileFlowStateStore.LocalFileFlowStateStore();
        })
      );
      import_logging.logger.debug("Registered dev flow state store.");
    }
    if (this.options.flowStateStore) {
      const flowStorePluginName = this.options.flowStateStore;
      import_logging.logger.debug(`  - prod: ${flowStorePluginName}`);
      this.configuredEnvs.add("prod");
      registry.registerFlowStateStore(
        "prod",
        () => this.resolveFlowStateStore(flowStorePluginName)
      );
    }
    import_logging.logger.debug("Registering trace stores...");
    if (isDevEnv()) {
      registry.registerTraceStore("dev", () => __async(this, null, function* () {
        return new import_localFileTraceStore.LocalFileTraceStore();
      }));
      import_logging.logger.debug("Registered dev trace store.");
    }
    if (this.options.traceStore) {
      const traceStorePluginName = this.options.traceStore;
      import_logging.logger.debug(`  - prod: ${traceStorePluginName}`);
      this.configuredEnvs.add("prod");
      registry.registerTraceStore(
        "prod",
        () => this.resolveTraceStore(traceStorePluginName)
      );
      if (isDevEnv()) {
        import_logging.logger.info(
          "In dev mode `traceStore` is defaulted to local file store."
        );
      }
    } else {
      import_logging.logger.info(
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
        (0, import_tracing.enableTracingAndMetrics)(
          yield this.getTelemetryConfig(),
          yield this.getTraceStore()
        );
      }
      if (this.loggerConfig) {
        import_logging.logger.init(yield this.loggerConfig());
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
    import_logging.logger.warn("configureGenkit was already called");
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
  while (import_path.default.resolve(current, "..") !== current) {
    if (import_fs.default.existsSync(import_path.default.resolve(current, "genkit.config.js"))) {
      return import_path.default.resolve(current, "genkit.config.js");
    }
    current = import_path.default.resolve(current, "..");
  }
  return void 0;
}
function __hardResetConfigForTesting() {
  config = void 0;
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  __hardResetConfigForTesting,
  config,
  configureGenkit,
  getCurrentEnv,
  initializeGenkit,
  isDevEnv,
  ...require("./plugin.js")
});
//# sourceMappingURL=config.js.map