"use strict";
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
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
var client_exports = {};
__export(client_exports, {
  runFlow: () => runFlow,
  streamFlow: () => streamFlow
});
module.exports = __toCommonJS(client_exports);
const __flowStreamDelimiter = "\n";
function streamFlow({
  url,
  input,
  headers
}) {
  let chunkStreamController = void 0;
  const chunkStream = new ReadableStream({
    start(controller) {
      chunkStreamController = controller;
    },
    pull() {
    },
    cancel() {
    }
  });
  const operationPromise = __flowRunEnvelope({
    url,
    input,
    streamingCallback: (c) => {
      chunkStreamController == null ? void 0 : chunkStreamController.enqueue(c);
    },
    headers
  });
  operationPromise.then((o) => {
    chunkStreamController == null ? void 0 : chunkStreamController.close();
    return o;
  });
  return {
    output() {
      return operationPromise.then((op) => {
        var _a2, _b, _c, _d;
        if (!op.done) {
          throw new Error(`flow ${op.name} did not finish execution`);
        }
        if ((_a2 = op.result) == null ? void 0 : _a2.error) {
          throw new Error(
            `${op.name}: ${(_b = op.result) == null ? void 0 : _b.error}
${(_c = op.result) == null ? void 0 : _c.stacktrace}`
          );
        }
        return (_d = op.result) == null ? void 0 : _d.response;
      });
    },
    stream() {
      return __asyncGenerator(this, null, function* () {
        const reader = chunkStream.getReader();
        while (true) {
          const chunk = yield new __await(reader.read());
          if (chunk.value) {
            yield chunk.value;
          }
          if (chunk.done) {
            break;
          }
        }
        return yield new __await(operationPromise);
      });
    }
  };
}
function __flowRunEnvelope(_0) {
  return __async(this, arguments, function* ({
    url,
    input,
    streamingCallback,
    headers
  }) {
    let response;
    response = yield fetch(url + "?stream=true", {
      method: "POST",
      body: JSON.stringify({
        data: input
      }),
      headers: __spreadValues({
        "Content-Type": "application/json"
      }, headers)
    });
    if (!response.body) {
      throw new Error("Response body is empty");
    }
    var reader = response.body.getReader();
    var decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const result = yield reader.read();
      const decodedValue = decoder.decode(result.value);
      if (decodedValue) {
        buffer += decodedValue;
      }
      while (buffer.includes(__flowStreamDelimiter)) {
        streamingCallback(
          JSON.parse(buffer.substring(0, buffer.indexOf(__flowStreamDelimiter)))
        );
        buffer = buffer.substring(
          buffer.indexOf(__flowStreamDelimiter) + __flowStreamDelimiter.length
        );
      }
      if (result.done) {
        return JSON.parse(buffer);
      }
    }
  });
}
function runFlow(_0) {
  return __async(this, arguments, function* ({
    url,
    payload,
    headers
  }) {
    const response = yield fetch(url, {
      method: "POST",
      body: JSON.stringify({
        data: payload
      }),
      headers: __spreadValues({
        "Content-Type": "application/json"
      }, headers)
    });
    const wrappedDesult = yield response.json();
    return wrappedDesult.result;
  });
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  runFlow,
  streamFlow
});
//# sourceMappingURL=client.js.map