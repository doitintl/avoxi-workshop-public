import {
  __async
} from "./chunk-XEFTB2OF.mjs";
import { logger } from "./logging.js";
import { startReflectionApi } from "./reflectionApi.js";
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
  logger.info(`Registering ${type}: ${action.__action.name}`);
  const key = `/${type}/${action.__action.name}`;
  if (actionsById().hasOwnProperty(key)) {
    logger.warn(
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
  startReflectionApi();
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
export {
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
};
//# sourceMappingURL=registry.mjs.map