export async function copyToClibpoard(content: string): Promise<void> {
  // Use modern APIs
  if (navigator.clipboard.writeText) {
    return navigator.clipboard.writeText(content)
  }

  // Try old things
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
