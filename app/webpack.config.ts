/**
 * Babel will compile modern JavaScript down to a format compatible with older browsers, but it will also increase your
 * final bundle size and build speed. Edit the `browserslist` property in the package.json file to define which
 * browsers Babel should target.
 *
 * Browserslist documentation: https://github.com/browserslist/browserslist#browserslist-
 */
const useBabel = true;

/**
 * This option controls whether or not development builds should be compiled with Babel. Change this to `true` if you
 * intend to test with older browsers during development, but it could significantly impact your build speed.
 */
const useBabelInDevelopment = false;

/**
 * Define paths to any stylesheets you wish to include at the top of the CSS bundle. Any styles compiled from svelte
 * will be added to the bundle after these. In other words, these are global styles for your svelte app. You can also
 * specify paths to SCSS or SASS files, and they will be compiled automatically.
 */
const stylesheets = ["./src/theme/theme.scss"];

/**
 * Change this to `true` to generate source maps alongside your production bundle. This is useful for debugging, but
 * will increase total bundle size and expose your source code.
 */
const sourceMapsInProduction = false;

// const HtmlWebpackPlugin = require('html-webpack-plugin');
const CopyPlugin = require("copy-webpack-plugin");

const ReactRefresh = require("@pmmmwh/react-refresh-webpack-plugin");

const proxy_matcher = (s: string) => s.match(/^\/(api|services)/)

/*********************************************************************************************************************/
/**********                                             Webpack                                             **********/
/*********************************************************************************************************************/

import type Webpack from "webpack";
import type WebpackDev from "webpack-dev-server";
import MiniCssExtractPlugin from "mini-css-extract-plugin";
import CSSMinimizerPlugin from "css-minimizer-webpack-plugin";
import TerserPlugin from "terser-webpack-plugin";

import { CleanWebpackPlugin } from "clean-webpack-plugin";

import fs from "fs";
import path from "path";

const mode = process.env.NODE_ENV ?? "development";
const isProduction = mode === "production";
const isDevelopment = !isProduction;

const ws_port = process.env.XPM_WS_PORT ?? "12345";
console.log(`Experimaestro to be reached on port ${ws_port}`);

const config: Configuration = {
  mode: isProduction ? "production" : "development",
  entry: {
    index: [...stylesheets, "./src/index.tsx"],
  },
  resolve: {
    extensions: [".mjs", ".js", ".ts", ".tsx"],
    mainFields: ["browser", "module", "main"],
  },
  output: {
    path: path.resolve(__dirname, "../src/experimaestro/server/data"),
    publicPath: "/",
    filename: "[name].js",
    chunkFilename: "[name].[id].js",
  },
  module: {
    rules: [
      // Rule: SASS
      {
        test: /\.(scss|sass)$/,
        use: [
          {
            loader: MiniCssExtractPlugin.loader,
          },
          {
            loader: "css-loader",
            options: {
              importLoaders: 2,
            },
          },
          {
            loader: "postcss-loader",
            options: {
              postcssOptions: {
                ctx: {},
              },
            },
          },
          "sass-loader",
        ],
      },

      // Rule: CSS
      {
        test: /\.css$/,
        use: [
          {
            loader: MiniCssExtractPlugin.loader,
          },
          {
            loader: "css-loader",
            options: {
              importLoaders: 1,
            },
          },
          {
            loader: "postcss-loader",
            options: {
              postcssOptions: {
                ctx: {
                  hello: 2,
                },
              },
            },
          },
        ],
      },

      // Rule: JS/TS and react
      {
        test: /\.(ts|js)x?$/i,
        exclude: /node_modules/,
        use: {
          loader: "babel-loader",
          options: {
            presets: [
              "@babel/preset-env",
              "@babel/preset-react",
              "@babel/preset-typescript",
            ],
            plugins: ["react-refresh/babel"],
          },
        },
      },
    ],
  },
  devServer: {
    hot: true,
    port: 5000,
    static: "./public",
    setupMiddlewares: (middlewares, devServer) => {
      if (!devServer) {
        throw new Error("webpack-dev-server is not defined");
      }

      middlewares.unshift({
        path: "/",
        middleware: (req: any, res: any, next: any) => {
          if (req.query.token) {
            // Sets the cookie
            res.cookie("token", req.query.token);
            res.redirect(307, "/");
          } else if (req.cookie.token == undefined) {
            res.redirect(307, "/login.html");
          } else {
            next();
          }
        },
      });

      // devServer.app!.get('/setup-middleware/some/path', (_, response) => {
      //   response.send('setup-middlewares option GET');
      // });

      return middlewares;
    },
    proxy: [
        {
          context: ['/services', '/api'],
          target: `ws://localhost:${ws_port}`,
          ws: true,
          // Forwards the cookies
          cookieDomainRewrite: "",
        }
    ],
  },
  target: isDevelopment ? "web" : "browserslist",
  plugins: [
    new CopyPlugin({
      patterns: [{ from: "public", to: "." }],
    }),
    new MiniCssExtractPlugin({
      filename: "[name].css",
    }),
    new ReactRefresh(),
  ],
  devtool: isProduction && !sourceMapsInProduction ? false : "source-map",
  stats: {
    chunks: false,
    chunkModules: false,
    modules: false,
    assets: true,
    entrypoints: false,
  },
};

/**
 * This interface combines configuration from `webpack` and `webpack-dev-server`. You can add or override properties
 * in this interface to change the config object type used above.
 */
export interface Configuration
  extends Webpack.Configuration,
    WebpackDev.Configuration {}

/*********************************************************************************************************************/
/**********                                             Advanced                                            **********/
/*********************************************************************************************************************/

// Configuration for production bundles
if (isProduction) {
  // Clean the build directory for production builds
  if (config.plugins === undefined) {
    config.plugins = [];
  }
  config.plugins.push(new CleanWebpackPlugin());

  // Minify and treeshake JS
  if (config.optimization === undefined) {
    config.optimization = {};
  }

  // Minify CSS files
  if (config.optimization.minimizer == undefined) {
    config.optimization.minimizer = [];
  }

  config.optimization.minimizer.push(
    new TerserPlugin({
      parallel: true,
      terserOptions: {
        // https://github.com/webpack-contrib/terser-webpack-plugin#terseroptions
        compress: {
          drop_console: true,
        },
      },
    })
  );

  config.optimization.minimizer.push(
    new CSSMinimizerPlugin({
      // sourceMap: sourceMapsInProduction ? {inline: false, annotation: true, }: false,
      parallel: true,
      minimizerOptions: {
        preset: [
          "default",
          {
            discardComments: { removeAll: !sourceMapsInProduction },
          },
        ],
      },
    })
  );

  config.optimization.minimize = true;
}

// Parse as JSON5 to add support for comments in tsconfig.json parsing.
require("require-json5").replace();

// Load path aliases from the tsconfig.json file
const tsconfigPath = path.resolve(__dirname, "tsconfig.json");
const tsconfig = fs.existsSync(tsconfigPath) ? require(tsconfigPath) : {};

if ("compilerOptions" in tsconfig && "paths" in tsconfig.compilerOptions) {
  const aliases = tsconfig.compilerOptions.paths;

  for (const alias in aliases) {
    const paths = aliases[alias].map((p: string) => path.resolve(__dirname, p));

    // Our tsconfig uses glob path formats, whereas webpack just wants directories
    // We'll need to transform the glob format into a format acceptable to webpack

    const wpAlias = alias.replace(/(\\|\/)\*$/, "");
    const wpPaths = paths.map((p: string) => p.replace(/(\\|\/)\*$/, ""));

    if (config.resolve && config.resolve.alias) {
      if (!(wpAlias in config.resolve.alias) && wpPaths.length) {
        // @ts-ignore
        config.resolve.alias[wpAlias] =
          wpPaths.length > 1 ? wpPaths : wpPaths[0];
      }
    }
  }
}

// Babel
if (useBabel && (isProduction || useBabelInDevelopment)) {
  const loader = {
    loader: "babel-loader",
    options: {
      sourceType: "unambiguous",
      presets: [
        [
          // Docs: https://babeljs.io/docs/en/babel-preset-env
          "@babel/preset-env",
          {
            debug: false,
            corejs: { version: 3 },
            useBuiltIns: "usage",
          },
        ],
      ],
      plugins: ["@babel/plugin-transform-runtime"],
    },
  };

  config.module?.rules?.unshift({
    test: /\.(?:m?js|ts)$/,
    include: [path.resolve(__dirname, "src"), path.resolve("node_modules")],
    exclude: [
      /node_modules[/\\](css-loader|core-js|webpack|regenerator-runtime)/,
    ],
    use: loader,
  });
}

export default config;
