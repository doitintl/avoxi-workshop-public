"use strict";
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
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
var __reExport = (target, mod, secondTarget) => (__copyProps(target, mod, "default"), secondTarget && __copyProps(secondTarget, mod, "default"));
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
var tracing_exports = {};
__export(tracing_exports, {
  cleanUpTracing: () => cleanUpTracing,
  enableTracingAndMetrics: () => enableTracingAndMetrics,
  flushTracing: () => flushTracing
});
module.exports = __toCommonJS(tracing_exports);
var import_sdk_node = require("@opentelemetry/sdk-node");
var import_sdk_trace_base = require("@opentelemetry/sdk-trace-base");
var import_config = require("./config.js");
var import_logging = require("./logging.js");
var import_exporter = require("./tracing/exporter.js");
var import_multiSpanProcessor = require("./tracing/multiSpanProcessor.js");
__reExport(tracing_exports, require("./tracing/exporter.js"), module.exports);
__reExport(tracing_exports, require("./tracing/instrumentation.js"), module.exports);
__reExport(tracing_exports, require("./tracing/localFileTraceStore.js"), module.exports);
__reExport(tracing_exports, require("./tracing/processor.js"), module.exports);
__reExport(tracing_exports, require("./tracing/types.js"), module.exports);
const processors = [];
let telemetrySDK = null;
let nodeOtelConfig = null;
function enableTracingAndMetrics(telemetryConfig, traceStore, traceStoreOptions = {}) {
  if (traceStore) {
    addProcessor(
      createTraceStoreProcessor(
        traceStore,
        traceStoreOptions.processor || "batch"
      )
    );
  }
  nodeOtelConfig = telemetryConfig.getConfig() || {};
  addProcessor(nodeOtelConfig.spanProcessor);
  nodeOtelConfig.spanProcessor = new import_multiSpanProcessor.MultiSpanProcessor(processors);
  telemetrySDK = new import_sdk_node.NodeSDK(nodeOtelConfig);
  telemetrySDK.start();
  process.on("SIGTERM", () => __async(this, null, function* () {
    return yield cleanUpTracing();
  }));
}
function cleanUpTracing() {
  return __async(this, null, function* () {
    return new Promise((resolve) => {
      if (telemetrySDK) {
        const metricFlush = maybeFlushMetrics();
        return metricFlush.then(() => {
          return telemetrySDK.shutdown().then(() => {
            import_logging.logger.debug("OpenTelemetry SDK shut down.");
            telemetrySDK = null;
            resolve();
          });
        });
      } else {
        resolve();
      }
    });
  });
}
function createTraceStoreProcessor(traceStore, processor) {
  const exporter = new import_exporter.TraceStoreExporter(traceStore);
  return processor === "simple" || (0, import_config.getCurrentEnv)() === "dev" ? new import_sdk_trace_base.SimpleSpanProcessor(exporter) : new import_sdk_trace_base.BatchSpanProcessor(exporter);
}
function addProcessor(processor) {
  if (processor)
    processors.push(processor);
}
function maybeFlushMetrics() {
  if (nodeOtelConfig == null ? void 0 : nodeOtelConfig.metricReader) {
    return nodeOtelConfig.metricReader.forceFlush();
  }
  return Promise.resolve();
}
function flushTracing() {
  return __async(this, null, function* () {
    yield Promise.all(processors.map((p) => p.forceFlush()));
  });
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  cleanUpTracing,
  enableTracingAndMetrics,
  flushTracing,
  ...require("./tracing/exporter.js"),
  ...require("./tracing/instrumentation.js"),
  ...require("./tracing/localFileTraceStore.js"),
  ...require("./tracing/processor.js"),
  ...require("./tracing/types.js")
});
//# sourceMappingURL=tracing.js.map