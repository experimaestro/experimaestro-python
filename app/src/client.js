// @flow

import store from './store'
import { toast } from 'react-toastify'

/// Connects to the Websocket server
class Client {
    ws: WebSocket;
    waiting: Map<number, any>;
    queued: Array<any>;

    constructor() {
        console.log("Connecting to websocket");

        let location = window.location;
        var url = 'ws://'+location.hostname+(location.port ? ':'+location.port: '') + '/ws';

        this.ws = new WebSocket(url);
        this.ws.addEventListener('open', this.open);        
        this.ws.addEventListener('close', this.close);        
        this.ws.addEventListener('message', this.message);        
    }

    open = () => {
        store.dispatch({ type: "CONNECTED", payload: true });
    }

    close = () => {
        store.dispatch({ type: "CONNECTED", payload: false });
        toast("Websocket connexion closed", {type: "info"});
    }

    message = (event: any) => {    
        store.dispatch(JSON.parse(event.data));
    }


    /** Send without waiting for an answer */
    send = (data: any, message: ?string) => {
        if (this.ws.readyState === WebSocket.OPEN) {
            return this.ws.send(JSON.stringify(data));
        } else {
            console.log("Connection not ready");
            if (message) {
                toast("No websocket connection: could not " + message, {type: "error"});
            }
            return false;
        }
            
    }

    /** Wait for an answer */
    query = (data: any, timeout: number = 60) => {
        return this.ws.send(JSON.stringify(data));
    }
}

export default new Client();