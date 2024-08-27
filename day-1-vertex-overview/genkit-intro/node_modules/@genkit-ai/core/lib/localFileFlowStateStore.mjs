import {
  __async
} from "./chunk-XEFTB2OF.mjs";
import crypto from "crypto";
import fs from "fs";
import os from "os";
import path from "path";
import {
  FlowStateSchema
} from "./flowTypes.js";
import { logger } from "./logging.js";
class LocalFileFlowStateStore {
  constructor() {
    var _a;
    const rootHash = crypto.createHash("md5").update(((_a = require == null ? void 0 : require.main) == null ? void 0 : _a.filename) || "unknown").digest("hex");
    this.storeRoot = path.resolve(os.tmpdir(), `.genkit/${rootHash}/flows`);
    fs.mkdirSync(this.storeRoot, { recursive: true });
    logger.info("Using DevFlowStateStore. Root: " + this.storeRoot);
  }
  load(id) {
    return __async(this, null, function* () {
      const filePath = path.resolve(this.storeRoot, `${id}`);
      if (!fs.existsSync(filePath)) {
        return void 0;
      }
      const data = fs.readFileSync(filePath, "utf8");
      return FlowStateSchema.parse(JSON.parse(data));
    });
  }
  save(id, state) {
    return __async(this, null, function* () {
      logger.debug("save flow state " + id);
      fs.writeFileSync(
        path.resolve(this.storeRoot, `${id}`),
        JSON.stringify(state)
      );
    });
  }
  list(query) {
    return __async(this, null, function* () {
      const files = fs.readdirSync(this.storeRoot);
      files.sort((a, b) => {
        return fs.statSync(path.resolve(this.storeRoot, `${b}`)).mtime.getTime() - fs.statSync(path.resolve(this.storeRoot, `${a}`)).mtime.getTime();
      });
      const startFrom = (query == null ? void 0 : query.continuationToken) ? parseInt(query == null ? void 0 : query.continuationToken) : 0;
      const stopAt = startFrom + ((query == null ? void 0 : query.limit) || 10);
      const flowStates = files.slice(startFrom, stopAt).map((id) => {
        const filePath = path.resolve(this.storeRoot, `${id}`);
        const data = fs.readFileSync(filePath, "utf8");
        return FlowStateSchema.parse(JSON.parse(data));
      });
      return {
        flowStates,
        continuationToken: files.length > stopAt ? stopAt.toString() : void 0
      };
    });
  }
}
export {
  LocalFileFlowStateStore
};
//# sourceMappingURL=localFileFlowStateStore.mjs.map