import { writable, Writable } from "svelte/store";

export type Message = {
  severity: string;
  message: string;
  title: string;
  delay?: number;
  id?: number;
};

export type Messages = Writable<Message[]>;

export let messages = writable<Message[]>([]);
let counter = 0;

function push(message: Message) {
  const id = counter++;
  let snackbar = {
    ...message,
    id: id,
  };

  messages.update((v) => [...v, snackbar]);
  if (message.delay) {
    setTimeout(function () {
      messages.update((v) => v.filter((s) => id != s.id));
    }, message.delay * 1000);
  }
}

export function info(message: string) {
  push({
    severity: "information",
    message: message,
    title: "Information",
    delay: 5,
  });
}

export function success(message: string) {
  let snackbar = {
    severity: "success",
    message: message,
    title: "Information",
    delay: 5,
  };
  push(snackbar);
}

export function warning(message: string) {
  let snackbar = {
    severity: "warning",
    message: message,
    delay: 10,
    title: "Warning",
  };
  push(snackbar);
}

export function error(message: string) {
  let snackbar = {
    severity: "danger",
    message: message,
    title: "Erreur",
  };
  push(snackbar);
}
