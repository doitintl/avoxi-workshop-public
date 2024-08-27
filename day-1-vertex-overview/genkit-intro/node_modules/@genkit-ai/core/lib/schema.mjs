import "./chunk-XEFTB2OF.mjs";
import Ajv from "ajv";
import addFormats from "ajv-formats";
import { z } from "zod";
import zodToJsonSchema from "zod-to-json-schema";
import { GenkitError } from "./error.js";
import { registerSchema } from "./registry.js";
const ajv = new Ajv();
addFormats(ajv);
const jsonSchemas = /* @__PURE__ */ new WeakMap();
const validators = /* @__PURE__ */ new WeakMap();
class ValidationError extends GenkitError {
  constructor({
    data,
    errors,
    schema
  }) {
    super({
      status: "INVALID_ARGUMENT",
      message: `Schema validation failed. Parse Errors:

${errors.map((e) => `- ${e.path}: ${e.message}`).join("\n")}

Provided data:

${JSON.stringify(data, null, 2)}

Required JSON schema:

${JSON.stringify(schema, null, 2)}`,
      detail: { errors, schema }
    });
  }
}
function toJsonSchema({
  jsonSchema,
  schema
}) {
  if (!jsonSchema && !schema)
    return null;
  if (jsonSchema)
    return jsonSchema;
  if (jsonSchemas.has(schema))
    return jsonSchemas.get(schema);
  const outSchema = zodToJsonSchema(schema, {
    $refStrategy: "none",
    removeAdditionalStrategy: "strict"
  });
  jsonSchemas.set(schema, outSchema);
  return outSchema;
}
function toErrorDetail(error) {
  return {
    path: error.instancePath.substring(1).replace(/\//g, ".") || "(root)",
    message: error.message
  };
}
function validateSchema(data, options) {
  var _a;
  const toValidate = toJsonSchema(options);
  if (!toValidate) {
    return { valid: true, schema: toValidate };
  }
  const validator = validators.get(toValidate) || ajv.compile(toValidate);
  const valid = validator(data);
  const errors = (_a = validator.errors) == null ? void 0 : _a.map((e) => e);
  return { valid, errors: errors == null ? void 0 : errors.map(toErrorDetail), schema: toValidate };
}
function parseSchema(data, options) {
  const { valid, errors, schema } = validateSchema(data, options);
  if (!valid)
    throw new ValidationError({ data, errors, schema });
  return data;
}
function defineSchema(name, schema) {
  registerSchema(name, { schema });
  return schema;
}
function defineJsonSchema(name, jsonSchema) {
  registerSchema(name, { jsonSchema });
  return jsonSchema;
}
export {
  ValidationError,
  defineJsonSchema,
  defineSchema,
  parseSchema,
  toJsonSchema,
  validateSchema,
  z
};
//# sourceMappingURL=schema.mjs.map