function isAbortLikeError(error) {
  if (!error) return false;
  if (typeof error === "object" && error !== null) {
    if (error.name === "AbortError") {
      return true;
    }
    if (typeof error.message === "string") {
      const message = error.message.toLowerCase();
      if (message.includes("signal is aborted") || message.includes("aborterror") || message.includes("aborted")) {
        return true;
      }
    }
  }
  return false;
}

module.exports = {
  isAbortLikeError,
};
