import {
  __async,
  __spreadValues
} from "../chunk-XEFTB2OF.mjs";
import { SpanKind } from "@opentelemetry/api";
import {
  ExportResultCode,
  hrTimeToMilliseconds
} from "@opentelemetry/core";
import { logger } from "../logging.js";
import { deleteUndefinedProps } from "../utils.js";
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
      spanKind: SpanKind[span.kind],
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
    deleteUndefinedProps(spanData);
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
          logger.error("Failed to save trace ${traceId}", e);
        }
        if (done) {
          return done({
            code: error ? ExportResultCode.FAILED : ExportResultCode.SUCCESS
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
  return hrTimeToMilliseconds(time);
}
export {
  TraceStoreExporter
};
//# sourceMappingURL=exporter.mjs.map