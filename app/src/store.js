// @flow

import { createStore, applyMiddleware } from 'redux'
import createSagaMiddleware from 'redux-saga'
import { composeWithDevTools } from 'redux-devtools-extension'
import rootSaga from './sagas'
import { reducer, initialState } from './reducer'
 
// --- Store


const sagaMiddleware = createSagaMiddleware();
const store = createStore(reducer, initialState, composeWithDevTools(applyMiddleware(sagaMiddleware)));
sagaMiddleware.run(rootSaga)

if (process.env.NODE_ENV !== 'production') {
    // $FlowFixMe
    if (module.hot) {
        module.hot.accept('./reducer', () => {
            store.replaceReducer(reducer);
        });
    }
}


export default store;