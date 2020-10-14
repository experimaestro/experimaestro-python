const { createProxyMiddleware } = require("http-proxy-middleware");

module.exports = {
  verbose: true,
  before: (app) => {
    if (process.env.XPM_PROXY) {
      console.log("Using proxy " + process.env.XPM_PROXY);
      app.use(
        "/ws",
        createProxyMiddleware("/ws", {
          target: `ws://${process.env.XPM_PROXY}/ws`,
          ws: true,
        })
      );
    }
  },
};
