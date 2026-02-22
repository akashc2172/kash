chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || message.type !== "fetchText" || !message.url) {
    return;
  }

  (async () => {
    try {
      const response = await fetch(String(message.url), { credentials: "omit" });
      const text = await response.text();
      sendResponse({
        ok: response.ok,
        status: response.status,
        text
      });
    } catch (err) {
      sendResponse({ ok: false, status: 0, error: err?.message || "Fetch failed" });
    }
  })();

  return true;
});
