export async function copyToClibpoard(content: string): Promise<void> {
  let clipboardElement = document.getElementById("clipboard-holder");
  if (!clipboardElement) throw "no clipboard element";

  clipboardElement.textContent = content;

  let range = document.createRange();
  let sel = window.getSelection()!;

  range.selectNode(clipboardElement);
  sel.removeAllRanges();
  sel.addRange(range);

  document.execCommand("copy");

  sel.removeAllRanges();
}
