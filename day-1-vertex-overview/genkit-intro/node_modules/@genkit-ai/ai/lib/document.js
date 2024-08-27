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
var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
  // If the importer is in node compatibility mode or this is not an ESM
  // file that has been converted to a CommonJS file using a Babel-
  // compatible transform (i.e. "__esModule" has not been set), then set
  // "default" to the CommonJS "module.exports" for node compatibility.
  isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
  mod
));
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);
var document_exports = {};
__export(document_exports, {
  Document: () => Document,
  DocumentDataSchema: () => DocumentDataSchema,
  MediaPartSchema: () => MediaPartSchema,
  PartSchema: () => PartSchema,
  TextPartSchema: () => TextPartSchema
});
module.exports = __toCommonJS(document_exports);
var import_zod = __toESM(require("zod"));
const EmptyPartSchema = import_zod.default.object({
  text: import_zod.default.never().optional(),
  media: import_zod.default.never().optional()
});
const TextPartSchema = EmptyPartSchema.extend({
  /** The text of the document. */
  text: import_zod.default.string()
});
const MediaPartSchema = EmptyPartSchema.extend({
  media: import_zod.default.object({
    /** The media content type. Inferred from data uri if not provided. */
    contentType: import_zod.default.string().optional(),
    /** A `data:` or `https:` uri containing the media content.  */
    url: import_zod.default.string()
  })
});
const PartSchema = import_zod.default.union([TextPartSchema, MediaPartSchema]);
const DocumentDataSchema = import_zod.default.object({
  content: import_zod.default.array(PartSchema),
  metadata: import_zod.default.record(import_zod.default.string(), import_zod.default.any()).optional()
});
class Document {
  constructor(data) {
    this.content = data.content;
    this.metadata = data.metadata;
  }
  static fromText(text, metadata) {
    return new Document({
      content: [{ text }],
      metadata
    });
  }
  /**
   * Concatenates all `text` parts present in the document with no delimiter.
   * @returns A string of all concatenated text parts.
   */
  text() {
    return this.content.map((part) => part.text || "").join("");
  }
  /**
   * Returns the first media part detected in the document. Useful for extracting
   * (for example) an image.
   * @returns The first detected `media` part in the document.
   */
  media() {
    var _a;
    return ((_a = this.content.find((part) => part.media)) == null ? void 0 : _a.media) || null;
  }
  toJSON() {
    return {
      content: this.content,
      metadata: this.metadata
    };
  }
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  Document,
  DocumentDataSchema,
  MediaPartSchema,
  PartSchema,
  TextPartSchema
});
//# sourceMappingURL=document.js.map