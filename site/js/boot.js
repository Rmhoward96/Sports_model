/* Boot: runs last on every page, after app.js and the redesign scripts. */
async function boot() {
  injectStylesOnce();
  mountGradientBackground();
  render();
  if (page !== "settings") setInterval(render, REFRESH_MS);  // settings has no live data; a re-render would blur inputs mid-edit
}
if (typeof window !== "undefined" && window.document && !window.__CA_TEST__) boot();
