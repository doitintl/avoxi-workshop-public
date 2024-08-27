"use strict";
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
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
var registry_exports = {};
__export(registry_exports, {
  __hardResetRegistryForTesting: () => __hardResetRegistryForTesting,
  initializePlugin: () => initializePlugin,
  listActions: () => listActions,
  lookupAction: () => lookupAction,
  lookupFlowStateStore: () => lookupFlowStateStore,
  lookupPlugin: () => lookupPlugin,
  lookupSchema: () => lookupSchema,
  lookupTraceStore: () => lookupTraceStore,
  registerAction: () => registerAction,
  registerFlowStateStore: () => registerFlowStateStore,
  registerPluginProvider: () => registerPluginProvider,
  registerSchema: () => registerSchema,
  registerTraceStore: () => registerTraceStore
});
module.exports = __toCommonJS(registry_exports);
var import_logging = require("./logging.js");
var import_reflectionApi = require("./reflectionApi.js");
const ACTIONS_BY_ID = "genkit__ACTIONS_BY_ID";
const TRACE_STORES_BY_ENV = "genkit__TRACE_STORES_BY_ENV";
const FLOW_STATE_STORES_BY_ENV = "genkit__FLOW_STATE_STORES_BY_ENV";
const PLUGINS_BY_NAME = "genkit__PLUGINS_BY_NAME";
const SCHEMAS_BY_NAME = "genkit__SCHEMAS_BY_NAME";
function actionsById() {
  if (global[ACTIONS_BY_ID] === void 0) {
    global[ACTIONS_BY_ID] = {};
  }
  return global[ACTIONS_BY_ID];
}
function traceStoresByEnv() {
  if (global[TRACE_STORES_BY_ENV] === void 0) {
    global[TRACE_STORES_BY_ENV] = {};
  }
  return global[TRACE_STORES_BY_ENV];
}
function flowStateStoresByEnv() {
  if (global[FLOW_STATE_STORES_BY_ENV] === void 0) {
    global[FLOW_STATE_STORES_BY_ENV] = {};
  }
  return global[FLOW_STATE_STORES_BY_ENV];
}
function pluginsByName() {
  if (global[PLUGINS_BY_NAME] === void 0) {
    global[PLUGINS_BY_NAME] = {};
  }
  return global[PLUGINS_BY_NAME];
}
function schemasByName() {
  if (global[SCHEMAS_BY_NAME] === void 0) {
    global[SCHEMAS_BY_NAME] = {};
  }
  return global[SCHEMAS_BY_NAME];
}
function lookupAction(key) {
  return __async(this, null, function* () {
    const pluginName = parsePluginName(key);
    if (!actionsById()[key] && pluginName) {
      yield initializePlugin(pluginName);
    }
    return actionsById()[key];
  });
}
function parsePluginName(registryKey) {
  const tokens = registryKey.split("/");
  if (tokens.length === 4) {
    return tokens[2];
  }
  return void 0;
}
function registerAction(type, action) {
  import_logging.logger.info(`Registering ${type}: ${action.__action.name}`);
  const key = `/${type}/${action.__action.name}`;
  if (actionsById().hasOwnProperty(key)) {
    import_logging.logger.warn(
      `WARNING: ${key} already has an entry in the registry. Overwriting.`
    );
  }
  actionsById()[key] = action;
}
function listActions() {
  return __async(this, null, function* () {
    for (const pluginName of Object.keys(pluginsByName())) {
      yield initializePlugin(pluginName);
    }
    return Object.assign({}, actionsById());
  });
}
function registerTraceStore(env, traceStoreProvider) {
  traceStoresByEnv()[env] = traceStoreProvider;
}
const traceStoresByEnvCache = {};
function lookupTraceStore(env) {
  return __async(this, null, function* () {
    if (!traceStoresByEnv()[env]) {
      return void 0;
    }
    const cached = traceStoresByEnvCache[env];
    if (!cached) {
      const newStore = traceStoresByEnv()[env]();
      traceStoresByEnvCache[env] = newStore;
      return newStore;
    }
    return cached;
  });
}
function registerFlowStateStore(env, flowStateStoreProvider) {
  flowStateStoresByEnv()[env] = flowStateStoreProvider;
}
const flowStateStoresByEnvCache = {};
function lookupFlowStateStore(env) {
  return __async(this, null, function* () {
    if (!flowStateStoresByEnv()[env]) {
      return void 0;
    }
    const cached = flowStateStoresByEnvCache[env];
    if (!cached) {
      const newStore = flowStateStoresByEnv()[env]();
      flowStateStoresByEnvCache[env] = newStore;
      return newStore;
    }
    return cached;
  });
}
function registerPluginProvider(name, provider) {
  let cached;
  pluginsByName()[name] = {
    name: provider.name,
    initializer: () => {
      if (cached) {
        return cached;
      }
      cached = provider.initializer();
      return cached;
    }
  };
}
function lookupPlugin(name) {
  return pluginsByName()[name];
}
function initializePlugin(name) {
  return __async(this, null, function* () {
    if (pluginsByName()[name]) {
      return yield pluginsByName()[name].initializer();
    }
    return void 0;
  });
}
function registerSchema(name, data) {
  schemasByName()[name] = data;
}
function lookupSchema(name) {
  return schemasByName()[name];
}
if (process.env.GENKIT_ENV === "dev") {
  (0, import_reflectionApi.startReflectionApi)();
}
function __hardResetRegistryForTesting() {
  delete global[ACTIONS_BY_ID];
  delete global[TRACE_STORES_BY_ENV];
  delete global[FLOW_STATE_STORES_BY_ENV];
  delete global[PLUGINS_BY_NAME];
  deleteAll(flowStateStoresByEnvCache);
  deleteAll(traceStoresByEnvCache);
}
function deleteAll(map) {
  Object.keys(map).forEach((key) => delete map[key]);
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  __hardResetRegistryForTesting,
  initializePlugin,
  listActions,
  lookupAction,
  lookupFlowStateStore,
  lookupPlugin,
  lookupSchema,
  lookupTraceStore,
  registerAction,
  registerFlowStateStore,
  registerPluginProvider,
  registerSchema,
  registerTraceStore
});
//# sourceMappingURL=registry.js.map