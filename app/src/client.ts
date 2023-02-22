// ES6 import or TypeScript
import { io, Socket } from "socket.io-client"
import store from "./store"
import { actions } from "./reducers"

class Client {
  socket: Socket

  constructor() {
    this.socket = io({ path: "/api"});
    const socket = this.socket;
    this.socket.on('connect', function() {
      store.dispatch(actions.setConnected(true))
      socket.emit('refresh')
      socket.emit('services')
    });

    socket.on("disconnect", () => {
      store.dispatch(actions.setConnected(false))
    });

    this.socket.on('service.add', function(data) {
      store.dispatch(actions.addService(data))
    })
    this.socket.on('service.update', function(data) {
      store.dispatch(actions.updateService(data))
    })

    this.socket.on('job.add', function(data) {
      store.dispatch(actions.addJob(data))
    })

    this.socket.on('job.update', function(data) {
      store.dispatch(actions.updateJob(data))
    })

    socket.on("connect_error", (err) => {
      console.warn("Websocket disconnected:", err.message)
      if (err.message === "invalid token") {
        const new_url = `${window.location.protocol}//${window.location.host}/login.html`
        console.log("Redirecting to", new_url)
        window.location.href = new_url
      }
    })

  }

  job_details(jobid: string) {
    this.socket.emit('job.details', jobid)
  }
  job_kill(jobid: string) {
    this.socket.emit('job.kill', jobid)
  }
}

export default new Client();
