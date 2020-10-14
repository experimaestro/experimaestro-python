import svelte from "rollup-plugin-svelte-hot";
import resolve from "@rollup/plugin-node-resolve";
import commonjs from "@rollup/plugin-commonjs";
import livereload from "rollup-plugin-livereload";
import { terser } from "rollup-plugin-terser";
import hmr, { autoCreate } from "rollup-plugin-hot";
import alias from "@rollup/plugin-alias";
import typescript from "@rollup/plugin-typescript";
import sveltePreprocess from "svelte-preprocess";
// import url from '@rollup/plugin-url';
import styles from "rollup-plugin-styles";
const path = require("path");

// Set this to true to pass the --single flag to sirv (this serves your
// index.html for any unmatched route, which is a requirement for SPA
// routers using History API / pushState)
//
// NOTE This will have no effect when running with Nollup. For Nollup, you'd
// have to add the --history-api-fallback yourself in your package.json
// scripts (see: https://github.com/PepsRyuu/nollup/#nollup-options)
//
const spa = false;

// NOTE The NOLLUP env variable is picked by various HMR plugins to switch
// in compat mode. You should not change its name (and set the env variable
// yourself if you launch nollup with custom comands).
const nollup = !!process.env.NOLLUP;
const watch = !!process.env.ROLLUP_WATCH;
const useLiveReload = !!process.env.LIVERELOAD;

const dev = watch || useLiveReload;
const production = !dev;

const hot = watch && !useLiveReload;
const destDir = production
  ? "../experimaestro/server/data/static"
  : "public/build";

console.log("Use livereload", useLiveReload, " / hot", hot);

console.log("Mode: ", hot ? "hot" : dev ? "dev" : "prod");

// function typeCheck() {
//   return {
//     writeBundle() {
//       require('child_process').spawn('svelte-check', {
//         stdio: ['ignore', 'inherit', 'inherit'],
//         shell: true
//       });
//     }
//   }
// }

function serve() {
  let started = false;
  return {
    name: "svelte/template:serve",
    writeBundle() {
      if (!started) {
        started = true;
        const flags = ["run", "start", "--", "--dev", "--cors"];
        if (spa) {
          flags.push("--single");
        }
        require("child_process").spawn("npm", flags, {
          stdio: ["ignore", "inherit", "inherit"],
          shell: true,
        });
      }
    },
  };
}

export default {
  input: `src/index.ts`,
  output: {
    sourcemap: !production,
    format: "iife",
    file: `${destDir}/index.js`,
    name: "index",

    // A bit hacky, but necessary to resolve assets in CSS files (handled with postcss)
    assetFileNames: "[name][extname]",
  },
  plugins: [
    alias({
      entries: [{ find: /^xpm\//, replacement: `${__dirname}/src/` }],
    }),
    // typeCheck(),
    svelte({
      // Enable run-time checks when not in production
      dev: !production,

      // Auto-preprocess
      preprocess: sveltePreprocess(),

      // Emit CSS as "files" for other plugins to process
      emitCss: !hot, // TODO: what should be the correct setting?

      hot: hot && {
        // Optimistic will try to recover from runtime
        // errors during component init
        optimistic: true,
        // Turn on to disable preservation of local component
        // state -- i.e. non exported `let` variables
        noPreserveState: false,

        // See docs of rollup-plugin-svelte-hot for all available options:
        //
        // https://github.com/rixo/rollup-plugin-svelte-hot#usage
      },
    }),

    // If you have external dependencies installed from
    // npm, you'll most likely need these plugins. In
    // some cases you'll need additional configuration —
    // consult the documentation for details:
    // https://github.com/rollup/rollup-plugin-commonjs
    resolve({
      browser: true,
      dedupe: ["svelte"],
    }),

    typescript({ sourceMap: !production }),

    commonjs(),

    // url({
    //   include: ['**/*.svg', '**/*.png', '**/*.jpg', '**/*.gif', '**/*.woff', '**/*.ttf'],
    //   destDir: destDir,
    //   emitFiles: true
    // }),

    styles({
      sourceMap: true,
      mode: hot ? "inject" : ["extract", `index.css`],
      to: path.resolve(destDir, "index.css"),
      sass: {
        includePaths: ["./src/theme", "./node_modules"],
      },
      url: false, // handled by post-css
      config: {
        ctx: {
          hot: hot,
          destdir: destDir,
        },
      },
    }),

    // In dev mode, call `npm run start:dev` once
    // the bundle has been generated
    dev && !nollup && serve(),

    // Watch the `public` directory and refresh the
    // browser on changes when not in production
    useLiveReload && livereload("public"),

    // If we're building for production (npm run build
    // instead of npm run dev), minify
    production && terser(),

    hot &&
      autoCreate({
        include: "src/**/*",
        // Set false to prevent recreating a file that has just been
        // deleted (Rollup watch will crash when you do that though).
        recreate: true,
      }),

    hot &&
      hmr({
        public: "public",
        inMemory: true,
        // This is needed, otherwise Terser (in npm run build) chokes
        // on import.meta. With this option, the plugin will replace
        // import.meta.hot in your code with module.hot, and will do
        // nothing else.
        compatModuleHot: !hot,
      }),
  ],
  watch: {
    clearScreen: false,
  },
};
