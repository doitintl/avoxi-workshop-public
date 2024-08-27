import {
  __async
} from "./chunk-XEFTB2OF.mjs";
import { NodeSDK } from "@opentelemetry/sdk-node";
import {
  BatchSpanProcessor,
  SimpleSpanProcessor
} from "@opentelemetry/sdk-trace-base";
import { getCurrentEnv } from "./config.js";
import { logger } from "./logging.js";
import { TraceStoreExporter } from "./tracing/exporter.js";
import { MultiSpanProcessor } from "./tracing/multiSpanProcessor.js";
export * from "./tracing/exporter.js";
export * from "./tracing/instrumentation.js";
export * from "./tracing/localFileTraceStore.js";
export * from "./tracing/processor.js";
export * from "./tracing/types.js";
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
  nodeOtelConfig.spanProcessor = new MultiSpanProcessor(processors);
  telemetrySDK = new NodeSDK(nodeOtelConfig);
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
            logger.debug("OpenTelemetry SDK shut down.");
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
  const exporter = new TraceStoreExporter(traceStore);
  return processor === "simple" || getCurrentEnv() === "dev" ? new SimpleSpanProcessor(exporter) : new BatchSpanProcessor(exporter);
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
export {
  cleanUpTracing,
  enableTracingAndMetrics,
  flushTracing
};
//# sourceMappingURL=tracing.mjs.map