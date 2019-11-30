const proxy = require('http-proxy-middleware');

module.exports = function(app) {
  if (process.env.XPM_PROXY) {
    console.log("Using proxy " + process.env.XPM_PROXY);
    app.use(proxy('/ws', { target: `http://${process.env.XPM_PROXY}/ws`, ws: true }));
  }
};