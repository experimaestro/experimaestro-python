import React from "react";
import { createUseStyles } from "react-jss";
import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { useSelector, useDispatch } from "react-redux";
import Toast from "react-bootstrap/Toast";
import ToastContainer from "react-bootstrap/ToastContainer";
import { SliceDispatchType } from "tc/store/utils";

export type MessageInput = {
  severity: string;
  message: string;
  title: string;
  delay?: number;
};

export type Message = MessageInput & {
  id: number;
};

type MessageStoreType = {
  messages: Message[];
};

let messageId = 0;

export const messageSlice = createSlice({
  name: "messages",
  initialState: {
    messages: [],
    counter: 0,
  } as MessageStoreType,
  reducers: {
    addMessage(state, action: PayloadAction<MessageInput>) {
      const message = action.payload;
      const id = messageId++;
      state.messages.push({
        ...message,
        id: id,
      });
    },
    closeMessage(state, action: PayloadAction<number>) {
      state.messages = state.messages.filter((s) => action.payload != s.id);
    },
  },
});

type MessageDispatchType = SliceDispatchType<typeof messageSlice>;

function showMessage(message: MessageInput) {
  return (dispatch: MessageDispatchType) => {
    const ownMessageId = messageId;
    dispatch(messageSlice.actions.addMessage(message));

    if (message.delay) {
      setTimeout(function () {
        dispatch(messageSlice.actions.closeMessage(ownMessageId));
      }, message.delay * 1000);
    }
  };
}

function info(message: string) {
  return showMessage({
    severity: "information",
    message: message,
    title: "Information",
    delay: 5,
  });
}

function success(message: string) {
  return showMessage({
    severity: "success",
    message: message,
    title: "Information",
    delay: 5,
  });
}

function warning(message: string) {
  return showMessage({
    severity: "warning",
    message: message,
    delay: 10,
    title: "Warning",
  });
}

function error(message: string) {
  return showMessage({
    severity: "danger",
    message: message,
    title: "Erreur",
  });
}

type DispatchThunk = (thunk: (dispatch: MessageDispatchType) => void) => void;

export function useMessages() {
  const dispatch = useDispatch() as DispatchThunk;
  return {
    success: (...args: Parameters<typeof success>) =>
      dispatch(success(...args)),
    info: (...args: Parameters<typeof info>) => dispatch(info(...args)),
    warning: (...args: Parameters<typeof warning>) =>
      dispatch(warning(...args)),
    error: (...args: Parameters<typeof error>) => dispatch(error(...args)),
  };
}

//
const useStyles = createUseStyles({
  container: {
    position: "fixed",
    top: "10px",
    left: "50%",
    transform: "translateX(-50%)",
    zIndex: 100,
  },
  body: {
    ".bg-danger &": {
      color: "white",
    },
    ".bg-success &": {
      color: "white",
    },
  },
});

export function Messages() {
  const classes = useStyles();
  const state = useSelector(
    (state: { messages: MessageStoreType }) => state.messages
  );
  const dispatch = useDispatch();

  return (
    <ToastContainer className={classes.container}>
      {state.messages.map((snackbar) => (
        <Toast
          key={snackbar.id}
          bg={snackbar.severity}
          onClose={() =>
            dispatch(messageSlice.actions.closeMessage(snackbar.id))
          }
          color={snackbar.severity}
        >
          <Toast.Header>{snackbar.title}</Toast.Header>
          <Toast.Body className={classes.body}>{snackbar.message}</Toast.Body>
        </Toast>
      ))}
    </ToastContainer>
  );
}
