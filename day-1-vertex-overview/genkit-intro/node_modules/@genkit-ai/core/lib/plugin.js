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
var plugin_exports = {};
__export(plugin_exports, {
  genkitPlugin: () => genkitPlugin
});
module.exports = __toCommonJS(plugin_exports);
function genkitPlugin(pluginName, initFn) {
  return (...args) => ({
    name: pluginName,
    initializer: () => __async(this, null, function* () {
      const initializedPlugin = (yield initFn(...args)) || {};
      validatePluginActions(pluginName, initializedPlugin);
      return initializedPlugin;
    })
  });
}
function validatePluginActions(pluginName, plugin) {
  var _a, _b, _c, _d, _e;
  if (!plugin) {
    return;
  }
  (_a = plugin.models) == null ? void 0 : _a.forEach((model) => validateNaming(pluginName, model));
  (_b = plugin.retrievers) == null ? void 0 : _b.forEach(
    (retriever) => validateNaming(pluginName, retriever)
  );
  (_c = plugin.embedders) == null ? void 0 : _c.forEach((embedder) => validateNaming(pluginName, embedder));
  (_d = plugin.indexers) == null ? void 0 : _d.forEach((indexer) => validateNaming(pluginName, indexer));
  (_e = plugin.evaluators) == null ? void 0 : _e.forEach(
    (evaluator) => validateNaming(pluginName, evaluator)
  );
}
function validateNaming(pluginName, action) {
  const nameParts = action.__action.name.split("/");
  if (nameParts[0] !== pluginName) {
    const err = `Plugin name ${pluginName} not found in action name ${action.__action.name}. Action names must follow the pattern {pluginName}/{actionName}`;
    throw new Error(err);
  }
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  genkitPlugin
});
//# sourceMappingURL=plugin.js.map