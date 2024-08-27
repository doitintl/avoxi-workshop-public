import {
  __async
} from "./chunk-XEFTB2OF.mjs";
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
export {
  genkitPlugin
};
//# sourceMappingURL=plugin.mjs.map