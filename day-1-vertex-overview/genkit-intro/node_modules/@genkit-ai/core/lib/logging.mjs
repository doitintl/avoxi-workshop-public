import {
  __async
} from "./chunk-XEFTB2OF.mjs";
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
export {
  logger
};
//# sourceMappingURL=logging.mjs.map