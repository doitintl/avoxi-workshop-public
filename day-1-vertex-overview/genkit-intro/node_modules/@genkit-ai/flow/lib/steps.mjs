import "./chunk-7OAPEGJQ.mjs";
import { getActiveContext } from "./utils.js";
function runAction(action, input) {
  return run(action.__action.name, input, () => action(input));
}
function run(name, funcOrInput, fn) {
  const func = arguments.length === 3 ? fn : funcOrInput;
  const input = arguments.length === 3 ? funcOrInput : void 0;
  if (!func) {
    throw new Error("unable to resolve run function");
  }
  const ctx = getActiveContext();
  if (!ctx)
    throw new Error("can only be run from a flow");
  return ctx.run({ name }, input, func);
}
function runMap(stepName, input, fn) {
  return Promise.all(input.map((f) => run(stepName, () => fn(f))));
}
export {
  run,
  runAction,
  runMap
};
//# sourceMappingURL=steps.mjs.map