import App from "./App.svelte";
import "../sass/App.scss";

const div = document.getElementById("root");
const app = div
  ? new App({
      target: div,
      props: {},
    })
  : null;

export default app;
