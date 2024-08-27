import "./chunk-XEFTB2OF.mjs";
function deleteUndefinedProps(obj) {
  for (const prop in obj) {
    if (obj[prop] === void 0) {
      delete obj[prop];
    } else {
      if (typeof obj[prop] === "object") {
        deleteUndefinedProps(obj[prop]);
      }
    }
  }
}
export {
  deleteUndefinedProps
};
//# sourceMappingURL=utils.mjs.map