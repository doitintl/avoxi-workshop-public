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
var types_exports = {};
__export(types_exports, {
  FlowActionInputSchema: () => FlowActionInputSchema,
  FlowInvokeEnvelopeMessageSchema: () => FlowInvokeEnvelopeMessageSchema
});
module.exports = __toCommonJS(types_exports);
var import_zod = require("zod");
const FlowInvokeEnvelopeMessageSchema = import_zod.z.object({
  // Start new flow.
  start: import_zod.z.object({
    input: import_zod.z.unknown().optional(),
    labels: import_zod.z.record(import_zod.z.string(), import_zod.z.string()).optional()
  }).optional(),
  // Schedule new flow.
  schedule: import_zod.z.object({
    input: import_zod.z.unknown().optional(),
    delay: import_zod.z.number().optional()
  }).optional(),
  // Run previously scheduled flow.
  runScheduled: import_zod.z.object({
    flowId: import_zod.z.string()
  }).optional(),
  // Retry failed step (only if step is setup for retry)
  retry: import_zod.z.object({
    flowId: import_zod.z.string()
  }).optional(),
  // Resume an interrupted flow.
  resume: import_zod.z.object({
    flowId: import_zod.z.string(),
    payload: import_zod.z.unknown().optional()
  }).optional(),
  // State check for a given flow ID. No side effects, can be used to check flow state.
  state: import_zod.z.object({
    flowId: import_zod.z.string()
  }).optional()
});
const FlowActionInputSchema = FlowInvokeEnvelopeMessageSchema.extend({
  auth: import_zod.z.unknown().optional()
});
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  FlowActionInputSchema,
  FlowInvokeEnvelopeMessageSchema
});
//# sourceMappingURL=types.js.map