import { configureStore } from "@reduxjs/toolkit";
import { useSelector, EqualityFn, useDispatch } from "react-redux";
import rootReducer from "./reducers";

const store = configureStore({
  reducer: rootReducer,
  // middleware: (getDefaultMiddleware) =>
  //   getDefaultMiddleware().concat(loggerMiddleware),
  // enhancers: [monitorReducersEnhancer],
});

if (process.env.NODE_ENV !== "production" && module.hot) {
  module.hot.accept("./reducers", () => store.replaceReducer(rootReducer));
}

export type AppDispatch = typeof store.dispatch;
export type StoreState = ReturnType<typeof store.getState>;
export const useAppDispatch: () => AppDispatch = useDispatch; // Export a hook that can be reused to resolve types
export const useAppSelector: <Selected = unknown>(
  selector: (state: StoreState) => Selected,
  equalityFn?: EqualityFn<Selected> | undefined
) => Selected = useSelector;

export default store;
