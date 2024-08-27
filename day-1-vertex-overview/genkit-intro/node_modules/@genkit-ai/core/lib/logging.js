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
var logging_exports = {};
__export(logging_exports, {
  logger: () => logger
});
module.exports = __toCommonJS(logging_exports);
const LOG_LEVELS = ["debug", "info", "warn", "error"];
class Logger {
  constructor() {
    this.defaultLogger = {
      shouldLog(targetLevel) {
        return LOG_LEVELS.indexOf(this.level) <= LOG_LEVELS.indexOf(targetLevel);
      },
      debug(...args) {
        this.shouldLog("debug") && console.debug(...args);
      },
      info(...args) {
        this.shouldLog("info") && console.info(...args);
      },
      warn(...args) {
        this.shouldLog("warn") && console.warn(...args);
      },
      error(...args) {
        this.shouldLog("error") && console.error(...args);
      },
      level: "info"
    };
    this.logger = this.defaultLogger;
  }
  init(config) {
    return __async(this, null, function* () {
      this.logger = yield config.getLogger(process.env.GENKIT_ENV || "prod");
    });
  }
  info(...args) {
    this.logger.info.apply(this.logger, args);
  }
  debug(...args) {
    this.logger.debug.apply(this.logger, args);
  }
  error(...args) {
    this.logger.error.apply(this.logger, args);
  }
  warn(...args) {
    this.logger.warn.apply(this.logger, args);
  }
  setLogLevel(level) {
    this.logger.level = level;
  }
  logStructured(msg, metadata) {
    this.logger.info(msg, metadata);
  }
  logStructuredError(msg, metadata) {
    this.logger.error(msg, metadata);
  }
}
const logger = new Logger();
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  logger
});
//# sourceMappingURL=logging.js.map