"use strict";
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getOwnPropSymbols = Object.getOwnPropertySymbols;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __propIsEnum = Object.prototype.propertyIsEnumerable;
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
var exporter_exports = {};
__export(exporter_exports, {
  TraceStoreExporter: () => TraceStoreExporter
});
module.exports = __toCommonJS(exporter_exports);
var import_api = require("@opentelemetry/api");
var import_core = require("@opentelemetry/core");
var import_logging = require("../logging.js");
var import_utils = require("../utils.js");
class TraceStoreExporter {
  constructor(traceStore) {
    this.traceStore = traceStore;
  }
  /**
   * Export spans.
   * @param spans
   * @param resultCallback
   */
  export(spans, resultCallback) {
    this._sendSpans(spans, resultCallback);
  }
  /**
   * Shutdown the exporter.
   */
  shutdown() {
    this._sendSpans([]);
    return this.forceFlush();
  }
  /**
   * Converts span info into trace store format.
   * @param span
   */
  _exportInfo(span) {
    const spanData = {
      spanId: span.spanContext().spanId,
      traceId: span.spanContext().traceId,
      startTime: transformTime(span.startTime),
      endTime: transformTime(span.endTime),
      attributes: __spreadValues({}, span.attributes),
      displayName: span.name,
      links: span.links,
      spanKind: import_api.SpanKind[span.kind],
      parentSpanId: span.parentSpanId,
      sameProcessAsParentSpan: { value: !span.spanContext().isRemote },
      status: span.status,
      timeEvents: {
        timeEvent: span.events.map((e) => {
          var _a;
          return {
            time: transformTime(e.time),
            annotation: {
              attributes: (_a = e.attributes) != null ? _a : {},
              description: e.name
            }
          };
        })
      }
    };
    if (span.instrumentationLibrary !== void 0) {
      spanData.instrumentationLibrary = {
        name: span.instrumentationLibrary.name
      };
      if (span.instrumentationLibrary.schemaUrl !== void 0) {
        spanData.instrumentationLibrary.schemaUrl = span.instrumentationLibrary.schemaUrl;
      }
      if (span.instrumentationLibrary.version !== void 0) {
        spanData.instrumentationLibrary.version = span.instrumentationLibrary.version;
      }
    }
    (0, import_utils.deleteUndefinedProps)(spanData);
    return spanData;
  }
  /**
   * Exports any pending spans in exporter
   */
  forceFlush() {
    return Promise.resolve();
  }
  _sendSpans(spans, done) {
    return __async(this, null, function* () {
      const traces = {};
      for (const span of spans) {
        if (!traces[span.spanContext().traceId]) {
          traces[span.spanContext().traceId] = [];
        }
        traces[span.spanContext().traceId].push(span);
      }
      let error = false;
      for (const traceId of Object.keys(traces)) {
        try {
          yield this.save(traceId, traces[traceId]);
        } catch (e) {
          error = true;
          import_logging.logger.error("Failed to save trace ${traceId}", e);
        }
        if (done) {
          return done({
            code: error ? import_core.ExportResultCode.FAILED : import_core.ExportResultCode.SUCCESS
          });
        }
      }
    });
  }
  save(traceId, spans) {
    return __async(this, null, function* () {
      const data = {
        traceId,
        spans: {}
      };
      for (const span of spans) {
        const convertedSpan = this._exportInfo(span);
        data.spans[convertedSpan.spanId] = convertedSpan;
        if (!convertedSpan.parentSpanId) {
          data.displayName = convertedSpan.displayName;
          data.startTime = convertedSpan.startTime;
          data.endTime = convertedSpan.endTime;
        }
      }
      yield this.traceStore.save(traceId, data);
    });
  }
}
function transformTime(time) {
  return (0, import_core.hrTimeToMilliseconds)(time);
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  TraceStoreExporter
});
//# sourceMappingURL=exporter.js.map