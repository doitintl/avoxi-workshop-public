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
var steps_exports = {};
__export(steps_exports, {
  run: () => run,
  runAction: () => runAction,
  runMap: () => runMap
});
module.exports = __toCommonJS(steps_exports);
var import_utils = require("./utils.js");
function runAction(action, input) {
  return run(action.__action.name, input, () => action(input));
}
function run(name, funcOrInput, fn) {
  const func = arguments.length === 3 ? fn : funcOrInput;
  const input = arguments.length === 3 ? funcOrInput : void 0;
  if (!func) {
    throw new Error("unable to resolve run function");
  }
  const ctx = (0, import_utils.getActiveContext)();
  if (!ctx)
    throw new Error("can only be run from a flow");
  return ctx.run({ name }, input, func);
}
function runMap(stepName, input, fn) {
  return Promise.all(input.map((f) => run(stepName, () => fn(f))));
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  run,
  runAction,
  runMap
});
//# sourceMappingURL=steps.js.map