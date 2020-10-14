const { createProxyMiddleware } = require("http-proxy-middleware");

module.exports = {
  verbose: true,
  before: (app) => {
    if (process.env.XPM_PROXY) {
      app.use(
        createProxyMiddleware({
          target: `ws://${process.env.XPM_PROXY}/ws`,
          ws: true,
        })
      );
    }
  },
};
