import "./chunk-7OAPEGJQ.mjs";
import z from "zod";
const EmptyPartSchema = z.object({
  text: z.never().optional(),
  media: z.never().optional()
});
const TextPartSchema = EmptyPartSchema.extend({
  /** The text of the document. */
  text: z.string()
});
const MediaPartSchema = EmptyPartSchema.extend({
  media: z.object({
    /** The media content type. Inferred from data uri if not provided. */
    contentType: z.string().optional(),
    /** A `data:` or `https:` uri containing the media content.  */
    url: z.string()
  })
});
const PartSchema = z.union([TextPartSchema, MediaPartSchema]);
const DocumentDataSchema = z.object({
  content: z.array(PartSchema),
  metadata: z.record(z.string(), z.any()).optional()
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
export {
  Document,
  DocumentDataSchema,
  MediaPartSchema,
  PartSchema,
  TextPartSchema
};
//# sourceMappingURL=document.mjs.map