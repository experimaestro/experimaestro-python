// @flow

import React from 'react'
import ReactDOM from 'react-dom'
import './index.css'
import App from './App'
import registerServiceWorker from './registerServiceWorker'
import { Provider } from 'react-redux'
import store from './store'

let div = document.getElementById('root')
if (div) {
    const render = Component => {
        return ReactDOM.render(
            <Provider store={store}><App /></Provider>,
            div
        );
    };

    render(App);

    // $FlowFixMe
    if (module.hot) {
        module.hot.accept('./App', () => {
            const NextApp = require('./App').default;
            render(NextApp);
        });
    }

    registerServiceWorker();

}