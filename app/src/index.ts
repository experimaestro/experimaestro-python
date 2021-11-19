/// <reference path="../node_modules/svelte/types/runtime/index.d.ts" />

import App from "./App.svelte";

const div = document.getElementById("root");
const app = div
  ? new App({
      target: div,
      props: {},
    })
  : null;

export default app;
