const path = require("path");
const prefixer = require("postcss-prefix-selector");
const url = require("postcss-url");

module.exports = (context) => {
  const options = context.options || { hot: false };
  // console.log("Post-CSS context", options)

  let _url;

  // if (options.hot) {
  //   _url = url({
  //     url: function (asset) {
  //       if (/^(material-icons|boostrap|@fortawesome.*)/.test(asset.url)) {
  //         let targetURL = `http://localhost:5000/node_modules/${asset.url}`;
  //         //console.log(asset.url, "->", targetURL)
  //         return targetURL;
  //       } else {
  //         // console.log("Ignoring", asset.url)
  //       }
  //     },
  //   });
  // } else {
  //   _url = url({
  //     url: "copy",
  //     basePath: path.resolve("node_modules"),
  //     assetsPath: "assets", //path.resolve(options.destdir, 'assets')
  //   });
  // }

  return {
    plugins: [_url],
  };
};
