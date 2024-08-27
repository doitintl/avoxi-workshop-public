"use strict";
var __defProp = Object.defineProperty;
var __defProps = Object.defineProperties;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropDescs = Object.getOwnPropertyDescriptors;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getOwnPropSymbols = Object.getOwnPropertySymbols;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __propIsEnum = Object.prototype.propertyIsEnumerable;
var __knownSymbol = (name, symbol) => {
  return (symbol = Symbol[name]) ? symbol : Symbol.for("Symbol." + name);
};
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __spreadValues = (a, b) => {
  for (var prop in b || (b = {}))
    if (__hasOwnProp.call(b, prop))
      __defNormalProp(a, prop, b[prop]);
  if (__getOwnPropSymbols)
    for (var prop of __getOwnPropSymbols(b)) {
      if (__propIsEnum.call(b, prop))
        __defNormalProp(a, prop, b[prop]);
    }
  return a;
};
var __spreadProps = (a, b) => __defProps(a, __getOwnPropDescs(b));
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
var __async = (__this, __arguments, generator) => {
  return new Promise((resolve, reject) => {
    var fulfilled = (value) => {
      try {
        step(generator.next(value));
      } catch (e) {
        reject(e);
      }
    };
    var rejected = (value) => {
      try {
        step(generator.throw(value));
      } catch (e) {
        reject(e);
      }
    };
    var step = (x) => x.done ? resolve(x.value) : Promise.resolve(x.value).then(fulfilled, rejected);
    step((generator = generator.apply(__this, __arguments)).next());
  });
};
var __await = function(promise, isYieldStar) {
  this[0] = promise;
  this[1] = isYieldStar;
};
var __asyncGenerator = (__this, __arguments, generator) => {
  var resume = (k, v, yes, no) => {
    try {
      var x = generator[k](v), isAwait = (v = x.value) instanceof __await, done = x.done;
      Promise.resolve(isAwait ? v[0] : v).then((y) => isAwait ? resume(k === "return" ? k : "next", v[1] ? { done: y.done, value: y.value } : y, yes, no) : yes({ value: y, done })).catch((e) => resume("throw", e, yes, no));
    } catch (e) {
      no(e);
    }
  };
  var method = (k) => it[k] = (x) => new Promise((yes, no) => resume(k, x, yes, no));
  var it = {};
  return generator = generator.apply(__this, __arguments), it[__knownSymbol("asyncIterator")] = () => it, method("next"), method("throw"), method("return"), it;
};
var generate_exports = {};
__export(generate_exports, {
  Candidate: () => Candidate,
  GenerateResponse: () => GenerateResponse,
  GenerateResponseChunk: () => GenerateResponseChunk,
  Message: () => Message,
  NoValidCandidatesError: () => NoValidCandidatesError,
  generate: () => generate,
  generateStream: () => generateStream,
  toGenerateRequest: () => toGenerateRequest
});
module.exports = __toCommonJS(generate_exports);
var import_core = require("@genkit-ai/core");
var import_registry = require("@genkit-ai/core/registry");
var import_schema = require("@genkit-ai/core/schema");
var import_extract = require("./extract.js");
var import_tool = require("./tool.js");
class Message {
  constructor(message) {
    this.role = message.role;
    this.content = message.content;
  }
  /**
   * If a message contains a `data` part, it is returned. Otherwise, the `output()`
   * method extracts the first valid JSON object or array from the text contained in
   * the message and returns it.
   *
   * @returns The structured output contained in the message.
   */
  output() {
    return this.data() || (0, import_extract.extractJson)(this.text());
  }
  toolResponseParts() {
    const res = this.content.filter((part) => !!part.toolResponse);
    return res;
  }
  /**
   * Concatenates all `text` parts present in the message with no delimiter.
   * @returns A string of all concatenated text parts.
   */
  text() {
    return this.content.map((part) => part.text || "").join("");
  }
  /**
   * Returns the first media part detected in the message. Useful for extracting
   * (for example) an image from a generation expected to create one.
   * @returns The first detected `media` part in the message.
   */
  media() {
    var _a;
    return ((_a = this.content.find((part) => part.media)) == null ? void 0 : _a.media) || null;
  }
  /**
   * Returns the first detected `data` part of a message.
   * @returns The first `data` part detected in the message (if any).
   */
  data() {
    var _a;
    return (_a = this.content.find((part) => part.data)) == null ? void 0 : _a.data;
  }
  /**
   * Returns all tool request found in this message.
   * @returns Array of all tool request found in this message.
   */
  toolRequests() {
    return this.content.filter(
      (part) => !!part.toolRequest
    );
  }
  /**
   * Converts the Message to a plain JS object.
   * @returns Plain JS object representing the data contained in the message.
   */
  toJSON() {
    return {
      role: this.role,
      content: [...this.content]
    };
  }
}
class Candidate {
  constructor(candidate, request) {
    this.message = new Message(candidate.message);
    this.index = candidate.index;
    this.usage = candidate.usage || {};
    this.finishReason = candidate.finishReason;
    this.finishMessage = candidate.finishMessage || "";
    this.custom = candidate.custom;
    this.request = request;
  }
  /**
   * If a candidate's message contains a `data` part, it is returned. Otherwise, the `output()`
   * method extracts the first valid JSON object or array from the text contained in
   * the candidate's message and returns it.
   *
   * @returns The structured output contained in the candidate.
   */
  output() {
    return this.message.output();
  }
  /**
   * Concatenates all `text` parts present in the candidate's message with no delimiter.
   * @returns A string of all concatenated text parts.
   */
  text() {
    return this.message.text();
  }
  /**
   * Returns the first detected media part in the candidate's message. Useful for extracting
   * (for example) an image from a generation expected to create one.
   * @returns The first detected `media` part in the candidate.
   */
  media() {
    return this.message.media();
  }
  /**
   * Returns the first detected `data` part of a candidate's message.
   * @returns The first `data` part detected in the candidate (if any).
   */
  data() {
    return this.message.data();
  }
  /**
   * Returns all tool request found in this candidate.
   * @returns Array of all tool request found in this candidate.
   */
  toolRequests() {
    return this.message.toolRequests();
  }
  /**
   * Determine whether this candidate has output that conforms to a provided schema.
   *
   * @param request A request containing output schema to validate against. If not provided, uses request embedded in candidate.
   * @returns True if output matches request schema or if no request schema is provided.
   */
  hasValidOutput(request) {
    var _a, _b, _c;
    const o = this.output();
    if (!request && !this.request) {
      return true;
    }
    const { valid } = (0, import_schema.validateSchema)(o, {
      jsonSchema: ((_a = request == null ? void 0 : request.output) == null ? void 0 : _a.schema) || ((_c = (_b = this.request) == null ? void 0 : _b.output) == null ? void 0 : _c.schema)
    });
    return valid;
  }
  /**
   * Appends the message generated by this candidate to the messages already
   * present in the generation request. The result of this method can be safely
   * serialized to JSON for persistence in a database.
   * @returns A serializable list of messages compatible with `generate({history})`.
   */
  toHistory() {
    var _a;
    if (!this.request)
      throw new Error(
        "Can't construct history for candidate without request data."
      );
    return [...(_a = this.request) == null ? void 0 : _a.messages, this.message.toJSON()];
  }
  /**
   * Converts the Candidate to a plain JS object.
   * @returns Plain JS object representing the data contained in the candidate.
   */
  toJSON() {
    var _a, _b;
    return {
      message: this.message.toJSON(),
      index: this.index,
      usage: this.usage,
      finishReason: this.finishReason,
      finishMessage: this.finishMessage,
      custom: !!this.custom && ((_b = (_a = this.custom).toJSON) == null ? void 0 : _b.call(_a)) || this.custom
    };
  }
}
class GenerateResponse {
  /**
   * If the selected candidate's message contains a `data` part, it is returned. Otherwise,
   * the `output()` method extracts the first valid JSON object or array from the text
   * contained in the selected candidate's message and returns it.
   *
   * @param index The candidate index from which to extract output. If not provided, finds first candidate that conforms to output schema.
   * @returns The structured output contained in the selected candidate.
   */
  output(index) {
    var _a, _b;
    if (index === void 0) {
      const c = this.candidates.find((c2) => c2.hasValidOutput(this.request));
      return (c == null ? void 0 : c.output()) || ((_a = this.candidates[0]) == null ? void 0 : _a.output());
    }
    return ((_b = this.candidates[index]) == null ? void 0 : _b.output()) || null;
  }
  /**
   * Concatenates all `text` parts present in the candidate's message with no delimiter.
   * @param index The candidate index from which to extract text, defaults to first candidate.
   * @returns A string of all concatenated text parts.
   */
  text(index = 0) {
    var _a;
    return ((_a = this.candidates[index]) == null ? void 0 : _a.text()) || "";
  }
  /**
   * Returns the first detected media part in the selected candidate's message. Useful for
   * extracting (for example) an image from a generation expected to create one.
   * @param index The candidate index from which to extract media, defaults to first candidate.
   * @returns The first detected `media` part in the candidate.
   */
  media(index = 0) {
    var _a;
    return ((_a = this.candidates[index]) == null ? void 0 : _a.media()) || null;
  }
  /**
   * Returns the first detected `data` part of the selected candidate's message.
   * @param index The candidate index from which to extract data, defaults to first candidate.
   * @returns The first `data` part detected in the candidate (if any).
   */
  data(index = 0) {
    var _a;
    return ((_a = this.candidates[index]) == null ? void 0 : _a.data()) || null;
  }
  /**
   * Returns all tool request found in the candidate.
   * @param index The candidate index from which to extract tool requests, defaults to first candidate.
   * @returns Array of all tool request found in the candidate.
   */
  toolRequests(index = 0) {
    return this.candidates[index].toolRequests();
  }
  /**
   * Appends the message generated by the selected candidate to the messages already
   * present in the generation request. The result of this method can be safely
   * serialized to JSON for persistence in a database.
   * @param index The candidate index to utilize during conversion, defaults to first candidate.
   * @returns A serializable list of messages compatible with `generate({history})`.
   */
  toHistory(index = 0) {
    return this.candidates[index].toHistory();
  }
  constructor(response, request) {
    this.candidates = (response.candidates || []).map(
      (candidate) => new Candidate(candidate, request)
    );
    this.usage = response.usage || {};
    this.custom = response.custom || {};
    this.request = request;
  }
  toJSON() {
    var _a, _b;
    return {
      candidates: this.candidates.map((candidate) => candidate.toJSON()),
      usage: this.usage,
      custom: ((_b = (_a = this.custom).toJSON) == null ? void 0 : _b.call(_a)) || this.custom,
      request: this.request
    };
  }
}
class GenerateResponseChunk {
  constructor(data, accumulatedChunks) {
    this.index = data.index;
    this.content = data.content || [];
    this.custom = data.custom;
    this.accumulatedChunks = accumulatedChunks;
  }
  /**
   * Concatenates all `text` parts present in the chunk with no delimiter.
   * @returns A string of all concatenated text parts.
   */
  text() {
    return this.content.map((part) => part.text || "").join("");
  }
  /**
   * Returns the first media part detected in the chunk. Useful for extracting
   * (for example) an image from a generation expected to create one.
   * @returns The first detected `media` part in the chunk.
   */
  media() {
    var _a;
    return ((_a = this.content.find((part) => part.media)) == null ? void 0 : _a.media) || null;
  }
  /**
   * Returns the first detected `data` part of a chunk.
   * @returns The first `data` part detected in the chunk (if any).
   */
  data() {
    var _a;
    return (_a = this.content.find((part) => part.data)) == null ? void 0 : _a.data;
  }
  /**
   * Returns all tool request found in this chunk.
   * @returns Array of all tool request found in this chunk.
   */
  toolRequests() {
    return this.content.filter(
      (part) => !!part.toolRequest
    );
  }
  /**
   * Attempts to extract the longest valid JSON substring from the accumulated chunks.
   * @returns The longest valid JSON substring found in the accumulated chunks.
   */
  output() {
    if (!this.accumulatedChunks)
      return null;
    const accumulatedText = this.accumulatedChunks.map((chunk) => chunk.content.map((part) => part.text || "").join("")).join("");
    return (0, import_extract.extractJson)(accumulatedText, false);
  }
  toJSON() {
    return { index: this.index, content: this.content, custom: this.custom };
  }
}
function getRoleFromPart(part) {
  if (part.toolRequest !== void 0)
    return "model";
  if (part.toolResponse !== void 0)
    return "tool";
  if (part.text !== void 0)
    return "user";
  if (part.media !== void 0)
    return "user";
  if (part.data !== void 0)
    return "user";
  throw new Error("No recognized fields in content");
}
function inferRoleFromParts(parts) {
  const uniqueRoles = /* @__PURE__ */ new Set();
  for (const part of parts) {
    const role = getRoleFromPart(part);
    uniqueRoles.add(role);
    if (uniqueRoles.size > 1) {
      throw new Error("Contents contain mixed roles");
    }
  }
  return Array.from(uniqueRoles)[0];
}
function toGenerateRequest(options) {
  return __async(this, null, function* () {
    var _a, _b, _c, _d, _e;
    const promptMessage = { role: "user", content: [] };
    if (typeof options.prompt === "string") {
      promptMessage.content.push({ text: options.prompt });
    } else if (Array.isArray(options.prompt)) {
      promptMessage.role = inferRoleFromParts(options.prompt);
      promptMessage.content.push(...options.prompt);
    } else {
      promptMessage.role = inferRoleFromParts([options.prompt]);
      promptMessage.content.push(options.prompt);
    }
    const messages = [...options.history || [], promptMessage];
    let tools;
    if (options.tools) {
      tools = yield (0, import_tool.resolveTools)(options.tools);
    }
    const out = {
      messages,
      candidates: options.candidates,
      config: options.config,
      context: options.context,
      tools: (tools == null ? void 0 : tools.map((tool) => (0, import_tool.toToolDefinition)(tool))) || [],
      output: {
        format: ((_a = options.output) == null ? void 0 : _a.format) || (((_b = options.output) == null ? void 0 : _b.schema) || ((_c = options.output) == null ? void 0 : _c.jsonSchema) ? "json" : "text"),
        schema: (0, import_schema.toJsonSchema)({
          schema: (_d = options.output) == null ? void 0 : _d.schema,
          jsonSchema: (_e = options.output) == null ? void 0 : _e.jsonSchema
        })
      }
    };
    if (!out.output.schema)
      delete out.output.schema;
    return out;
  });
}
const isValidCandidate = (candidate, tools) => {
  const toolCalls = candidate.message.content.filter(
    (part) => !!part.toolRequest
  );
  return toolCalls.every((toolCall) => {
    var _a;
    const tool = tools == null ? void 0 : tools.find(
      (tool2) => {
        var _a2;
        return tool2.__action.name === ((_a2 = toolCall.toolRequest) == null ? void 0 : _a2.name);
      }
    );
    if (!tool)
      return false;
    const { valid } = (0, import_schema.validateSchema)((_a = toolCall.toolRequest) == null ? void 0 : _a.input, {
      schema: tool.__action.inputSchema,
      jsonSchema: tool.__action.inputJsonSchema
    });
    return valid;
  });
};
function resolveModel(options) {
  return __async(this, null, function* () {
    var _a, _b;
    let model = options.model;
    if (!model) {
      if ((_b = (_a = import_core.config) == null ? void 0 : _a.options) == null ? void 0 : _b.defaultModel) {
        model = typeof import_core.config.options.defaultModel.name === "string" ? import_core.config.options.defaultModel.name : import_core.config.options.defaultModel.name.name;
        if ((!options.config || Object.keys(options.config).length === 0) && import_core.config.options.defaultModel.config) {
          options.config = import_core.config.options.defaultModel.config;
        }
      } else {
        throw new Error("Unable to resolve model.");
      }
    }
    if (typeof model === "string") {
      return yield (0, import_registry.lookupAction)(`/model/${model}`);
    } else if (model.hasOwnProperty("info")) {
      const ref = model;
      return yield (0, import_registry.lookupAction)(`/model/${ref.name}`);
    } else {
      return model;
    }
  });
}
class NoValidCandidatesError extends import_core.GenkitError {
  constructor({
    message,
    response,
    detail
  }) {
    super({
      status: "FAILED_PRECONDITION",
      message,
      detail
    });
    this.detail = __spreadValues({ response }, detail);
  }
}
function generate(options) {
  return __async(this, null, function* () {
    var _a, _b, _c, _d, _e;
    const resolvedOptions = yield Promise.resolve(options);
    const model = yield resolveModel(resolvedOptions);
    if (!model) {
      throw new Error(`Model ${JSON.stringify(resolvedOptions.model)} not found`);
    }
    let tools;
    if ((_a = resolvedOptions.tools) == null ? void 0 : _a.length) {
      if (!((_c = (_b = model.__action.metadata) == null ? void 0 : _b.model.supports) == null ? void 0 : _c.tools)) {
        throw new Error(
          `Model ${JSON.stringify(resolvedOptions.model)} does not support tools, but some tools were supplied to generate(). Please call generate() without tools if you would like to use this model.`
        );
      }
      tools = yield (0, import_tool.resolveTools)(resolvedOptions.tools);
    }
    const request = yield toGenerateRequest(resolvedOptions);
    const accumulatedChunks = [];
    const response = yield (0, import_core.runWithStreamingCallback)(
      resolvedOptions.streamingCallback ? (chunk) => {
        accumulatedChunks.push(chunk);
        if (resolvedOptions.streamingCallback) {
          resolvedOptions.streamingCallback(
            new GenerateResponseChunk(chunk, accumulatedChunks)
          );
        }
      } : void 0,
      () => __async(this, null, function* () {
        return new GenerateResponse(yield model(request), request);
      })
    );
    if (!response.candidates.some(
      (c) => ["stop", "length"].includes(c.finishReason)
    )) {
      throw new NoValidCandidatesError({
        message: `All candidates returned finishReason issues: ${JSON.stringify(response.candidates.map((c) => c.finishReason))}`,
        response
      });
    }
    if (((_d = resolvedOptions.output) == null ? void 0 : _d.schema) || ((_e = resolvedOptions.output) == null ? void 0 : _e.jsonSchema)) {
      const candidateErrors = response.candidates.map((c) => {
        var _a2, _b2;
        if (c.text() === "" && c.data() === null)
          return null;
        try {
          (0, import_schema.parseSchema)(c.output(), {
            jsonSchema: (_a2 = resolvedOptions.output) == null ? void 0 : _a2.jsonSchema,
            schema: (_b2 = resolvedOptions.output) == null ? void 0 : _b2.schema
          });
          return null;
        } catch (e) {
          return e;
        }
      });
      if (candidateErrors.every((c) => !!c)) {
        throw new NoValidCandidatesError({
          message: `Generation resulted in no candidates matching provided output schema.${candidateErrors.map((e, i) => `

Candidate[${i}] ${e.toString()}`)}`,
          response,
          detail: {
            candidateErrors
          }
        });
      }
    }
    let selected;
    for (const candidate of response.candidates) {
      if (isValidCandidate(candidate, tools || [])) {
        selected = candidate;
        break;
      }
    }
    if (!selected) {
      throw new Error("No valid candidates found");
    }
    const toolCalls = selected.message.content.filter(
      (part) => !!part.toolRequest
    );
    if (resolvedOptions.returnToolRequests || toolCalls.length === 0) {
      return response;
    }
    const toolResponses = yield Promise.all(
      toolCalls.map((part) => __async(this, null, function* () {
        var _a2;
        if (!part.toolRequest) {
          throw Error(
            "Tool request expected but not provided in tool request part"
          );
        }
        const tool = tools == null ? void 0 : tools.find(
          (tool2) => {
            var _a3;
            return tool2.__action.name === ((_a3 = part.toolRequest) == null ? void 0 : _a3.name);
          }
        );
        if (!tool) {
          throw Error("Tool not found");
        }
        return {
          toolResponse: {
            name: part.toolRequest.name,
            ref: part.toolRequest.ref,
            output: yield tool((_a2 = part.toolRequest) == null ? void 0 : _a2.input)
          }
        };
      }))
    );
    resolvedOptions.history = request.messages;
    resolvedOptions.history.push(selected.message);
    resolvedOptions.prompt = toolResponses;
    return yield generate(resolvedOptions);
  });
}
function createPromise() {
  let resolve, reject;
  let promise = new Promise((res, rej) => [resolve, reject] = [res, rej]);
  return { resolve, reject, promise };
}
function generateStream(options) {
  return __async(this, null, function* () {
    let firstChunkSent = false;
    return new Promise(
      (initialResolve, initialReject) => {
        const {
          resolve: finalResolve,
          reject: finalReject,
          promise: finalPromise
        } = createPromise();
        let provideNextChunk, nextChunk;
        ({ resolve: provideNextChunk, promise: nextChunk } = createPromise());
        function chunkStream() {
          return __asyncGenerator(this, null, function* () {
            while (true) {
              const next = yield new __await(nextChunk);
              if (!next)
                break;
              yield next;
            }
          });
        }
        try {
          generate(__spreadProps(__spreadValues({}, options), {
            streamingCallback: (chunk) => {
              firstChunkSent = true;
              provideNextChunk(chunk);
              ({ resolve: provideNextChunk, promise: nextChunk } = createPromise());
            }
          })).then((result) => {
            provideNextChunk(null);
            finalResolve(result);
          });
        } catch (e) {
          if (!firstChunkSent) {
            initialReject(e);
            return;
          }
          provideNextChunk(null);
          finalReject(e);
        }
        initialResolve({
          response: () => finalPromise,
          stream: chunkStream
        });
      }
    );
  });
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  Candidate,
  GenerateResponse,
  GenerateResponseChunk,
  Message,
  NoValidCandidatesError,
  generate,
  generateStream,
  toGenerateRequest
});
//# sourceMappingURL=generate.js.map