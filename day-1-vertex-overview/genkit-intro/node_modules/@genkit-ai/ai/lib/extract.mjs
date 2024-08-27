import "./chunk-7OAPEGJQ.mjs";
import JSON5 from "json5";
import { Allow, parse } from "partial-json";
function parsePartialJson(jsonString) {
  return JSON5.parse(JSON.stringify(parse(jsonString, Allow.ALL)));
}
function extractJson(text, throwOnBadJson) {
  let openingChar;
  let closingChar;
  let startPos;
  let nestingCount = 0;
  for (let i = 0; i < text.length; i++) {
    const char = text[i].replace(/\u00A0/g, " ");
    if (!openingChar && (char === "{" || char === "[")) {
      openingChar = char;
      closingChar = char === "{" ? "}" : "]";
      startPos = i;
      nestingCount++;
    } else if (char === openingChar) {
      nestingCount++;
    } else if (char === closingChar) {
      nestingCount--;
      if (!nestingCount) {
        return JSON5.parse(text.substring(startPos || 0, i + 1));
      }
    }
  }
  if (startPos !== void 0 && nestingCount > 0) {
    try {
      return parsePartialJson(text.substring(startPos));
    } catch (e) {
      if (throwOnBadJson) {
        throw new Error(`Invalid JSON extracted from model output: ${text}`);
      }
      return null;
    }
  }
  if (throwOnBadJson) {
    throw new Error(`Invalid JSON extracted from model output: ${text}`);
  }
  return null;
}
export {
  extractJson,
  parsePartialJson
};
//# sourceMappingURL=extract.mjs.map