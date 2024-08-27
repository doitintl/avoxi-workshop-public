import {
  __async,
  __spreadProps,
  __spreadValues
} from "../chunk-7OAPEGJQ.mjs";
import { Document } from "../document.js";
function downloadRequestMedia(options) {
  return (req, next) => __async(this, null, function* () {
    const { default: fetch } = yield import("node-fetch");
    const newReq = __spreadProps(__spreadValues({}, req), {
      messages: yield Promise.all(
        req.messages.map((message) => __async(this, null, function* () {
          const content = yield Promise.all(
            message.content.map((part) => __async(this, null, function* () {
              if (!part.media || !part.media.url.startsWith("http") || (options == null ? void 0 : options.filter) && !(options == null ? void 0 : options.filter(part))) {
                return part;
              }
              const response = yield fetch(part.media.url, {
                size: options == null ? void 0 : options.maxBytes
              });
              if (response.status !== 200)
                throw new Error(
                  `HTTP error downloading media '${part.media.url}': ${yield response.text()}`
                );
              const contentType = part.media.contentType || response.headers.get("content-type") || "";
              return {
                media: {
                  contentType,
                  url: `data:${contentType};base64,${Buffer.from(
                    yield response.arrayBuffer()
                  ).toString("base64")}`
                }
              };
            }))
          );
          return __spreadProps(__spreadValues({}, message), {
            content
          });
        }))
      )
    });
    return next(newReq);
  });
}
function validateSupport(options) {
  const supports = options.supports || {};
  return (req, next) => __async(this, null, function* () {
    var _a, _b, _c, _d;
    function invalid(message) {
      throw new Error(
        `Model '${options.name}' does not support ${message}. Request: ${JSON.stringify(
          req,
          null,
          2
        )}`
      );
    }
    if (supports.media === false && req.messages.some((message) => message.content.some((part) => part.media)))
      invalid("media, but media was provided");
    if (supports.tools === false && ((_a = req.tools) == null ? void 0 : _a.length))
      invalid("tool use, but tools were provided");
    if (supports.multiturn === false && req.messages.length > 1)
      invalid(`multiple messages, but ${req.messages.length} were provided`);
    if (typeof supports.output !== "undefined" && ((_b = req.output) == null ? void 0 : _b.format) && !supports.output.includes((_c = req.output) == null ? void 0 : _c.format))
      invalid(`requested output format '${(_d = req.output) == null ? void 0 : _d.format}'`);
    return next();
  });
}
function lastUserMessage(messages) {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user") {
      return messages[i];
    }
  }
}
function conformOutput() {
  return (req, next) => __async(this, null, function* () {
    var _a, _b;
    const lastMessage = lastUserMessage(req.messages);
    if (!lastMessage)
      return next(req);
    const outputPartIndex = lastMessage.content.findIndex(
      (p) => {
        var _a2;
        return ((_a2 = p.metadata) == null ? void 0 : _a2.purpose) === "output";
      }
    );
    const outputPart = outputPartIndex >= 0 ? lastMessage.content[outputPartIndex] : void 0;
    if (!((_a = req.output) == null ? void 0 : _a.schema) || outputPart && !((_b = outputPart == null ? void 0 : outputPart.metadata) == null ? void 0 : _b.pending)) {
      return next(req);
    }
    const instructions = `

Output should be in JSON format and conform to the following schema:

\`\`\`
${JSON.stringify(req.output.schema)}
\`\`\`
`;
    if (outputPart) {
      lastMessage.content[outputPartIndex] = __spreadProps(__spreadValues({}, outputPart), {
        metadata: {
          purpose: "output",
          source: "default"
        },
        text: instructions
      });
    } else {
      lastMessage == null ? void 0 : lastMessage.content.push({
        text: instructions,
        metadata: { purpose: "output", source: "default" }
      });
    }
    return next(req);
  });
}
function simulateSystemPrompt(options) {
  const preface = (options == null ? void 0 : options.preface) || "SYSTEM INSTRUCTIONS:\n";
  const acknowledgement = (options == null ? void 0 : options.acknowledgement) || "Understood.";
  return (req, next) => {
    const messages = [...req.messages];
    for (let i = 0; i < messages.length; i++) {
      if (req.messages[i].role === "system") {
        const systemPrompt = messages[i].content;
        messages.splice(
          i,
          1,
          { role: "user", content: [{ text: preface }, ...systemPrompt] },
          { role: "model", content: [{ text: acknowledgement }] }
        );
        break;
      }
    }
    return next(__spreadProps(__spreadValues({}, req), { messages }));
  };
}
const CONTEXT_PREFACE = "\n\nUse the following information to complete your task:\n\n";
const CONTEXT_ITEM_TEMPLATE = (d, index, options) => {
  var _a, _b;
  let out = "- ";
  if (options == null ? void 0 : options.citationKey) {
    out += `[${d.metadata[options.citationKey]}]: `;
  } else if ((options == null ? void 0 : options.citationKey) === void 0) {
    out += `[${((_a = d.metadata) == null ? void 0 : _a["ref"]) || ((_b = d.metadata) == null ? void 0 : _b["id"]) || index}]: `;
  }
  out += d.text() + "\n";
  return out;
};
function augmentWithContext(options) {
  const preface = typeof (options == null ? void 0 : options.preface) === "undefined" ? CONTEXT_PREFACE : options.preface;
  const itemTemplate = (options == null ? void 0 : options.itemTemplate) || CONTEXT_ITEM_TEMPLATE;
  return (req, next) => {
    var _a, _b, _c;
    if (!((_a = req.context) == null ? void 0 : _a.length))
      return next(req);
    const userMessage = lastUserMessage(req.messages);
    if (!userMessage)
      return next(req);
    const contextPartIndex = userMessage == null ? void 0 : userMessage.content.findIndex(
      (p) => {
        var _a2;
        return ((_a2 = p.metadata) == null ? void 0 : _a2.purpose) === "context";
      }
    );
    const contextPart = contextPartIndex >= 0 && userMessage.content[contextPartIndex];
    if (contextPart && !((_b = contextPart.metadata) == null ? void 0 : _b.pending)) {
      return next(req);
    }
    let out = `${preface || ""}`;
    (_c = req.context) == null ? void 0 : _c.forEach((d, i) => {
      out += itemTemplate(new Document(d), i, options);
    });
    out += "\n";
    if (contextPartIndex >= 0) {
      userMessage.content[contextPartIndex] = __spreadProps(__spreadValues({}, contextPart), {
        text: out,
        metadata: { purpose: "context" }
      });
    } else {
      userMessage.content.push({ text: out, metadata: { purpose: "context" } });
    }
    return next(req);
  };
}
export {
  CONTEXT_PREFACE,
  augmentWithContext,
  conformOutput,
  downloadRequestMedia,
  simulateSystemPrompt,
  validateSupport
};
//# sourceMappingURL=middleware.mjs.map