import {
  __async,
  __asyncGenerator,
  __await,
  __spreadValues
} from "../chunk-7OAPEGJQ.mjs";
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
export {
  runFlow,
  streamFlow
};
//# sourceMappingURL=client.mjs.map