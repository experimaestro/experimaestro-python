import { success } from "./ui/notifications";

export function copyToClibpoard(element: HTMLElement) {
  let range = document.createRange();
  let sel = window.getSelection()!;

  range.selectNode(element);
  sel.removeAllRanges();
  sel.addRange(range);

  document.execCommand("copy");

  success(`Information copied`);
  sel.removeAllRanges();

  // error(`Error copying job path`)
}
