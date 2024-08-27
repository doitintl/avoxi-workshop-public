import "./chunk-XEFTB2OF.mjs";
import * as z from "zod";
var StatusCodes = /* @__PURE__ */ ((StatusCodes2) => {
  StatusCodes2[StatusCodes2["OK"] = 0] = "OK";
  StatusCodes2[StatusCodes2["CANCELLED"] = 1] = "CANCELLED";
  StatusCodes2[StatusCodes2["UNKNOWN"] = 2] = "UNKNOWN";
  StatusCodes2[StatusCodes2["INVALID_ARGUMENT"] = 3] = "INVALID_ARGUMENT";
  StatusCodes2[StatusCodes2["DEADLINE_EXCEEDED"] = 4] = "DEADLINE_EXCEEDED";
  StatusCodes2[StatusCodes2["NOT_FOUND"] = 5] = "NOT_FOUND";
  StatusCodes2[StatusCodes2["ALREADY_EXISTS"] = 6] = "ALREADY_EXISTS";
  StatusCodes2[StatusCodes2["PERMISSION_DENIED"] = 7] = "PERMISSION_DENIED";
  StatusCodes2[StatusCodes2["UNAUTHENTICATED"] = 16] = "UNAUTHENTICATED";
  StatusCodes2[StatusCodes2["RESOURCE_EXHAUSTED"] = 8] = "RESOURCE_EXHAUSTED";
  StatusCodes2[StatusCodes2["FAILED_PRECONDITION"] = 9] = "FAILED_PRECONDITION";
  StatusCodes2[StatusCodes2["ABORTED"] = 10] = "ABORTED";
  StatusCodes2[StatusCodes2["OUT_OF_RANGE"] = 11] = "OUT_OF_RANGE";
  StatusCodes2[StatusCodes2["UNIMPLEMENTED"] = 12] = "UNIMPLEMENTED";
  StatusCodes2[StatusCodes2["INTERNAL"] = 13] = "INTERNAL";
  StatusCodes2[StatusCodes2["UNAVAILABLE"] = 14] = "UNAVAILABLE";
  StatusCodes2[StatusCodes2["DATA_LOSS"] = 15] = "DATA_LOSS";
  return StatusCodes2;
})(StatusCodes || {});
const StatusNameSchema = z.enum([
  "CANCELLED",
  "UNKNOWN",
  "INVALID_ARGUMENT",
  "DEADLINE_EXCEEDED",
  "NOT_FOUND",
  "ALREADY_EXISTS",
  "PERMISSION_DENIED",
  "UNAUTHENTICATED",
  "RESOURCE_EXHAUSTED",
  "FAILED_PRECONDITION",
  "ABORTED",
  "OUT_OF_RANGE",
  "UNIMPLEMENTED",
  "INTERNAL",
  "UNAVAILABLE",
  "DATA_LOSS"
]);
const StatusCodesSchema = z.nativeEnum(StatusCodes);
const StatusSchema = z.object({
  code: StatusCodesSchema,
  message: z.string(),
  details: z.any().optional()
});
export {
  StatusCodes,
  StatusNameSchema,
  StatusSchema
};
//# sourceMappingURL=statusTypes.mjs.map