import "./chunk-XEFTB2OF.mjs";
class GenkitError extends Error {
  constructor({
    status,
    message,
    detail,
    source
  }) {
    super(`${source ? `${source}: ` : ""}${status}: ${message}`);
    this.status = status;
    this.detail = detail;
  }
}
export {
  GenkitError
};
//# sourceMappingURL=error.mjs.map