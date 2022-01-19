<script lang="ts">
    import { connected, experiment, jobs } from './store'
    import Messages from './ui/Messages.svelte'
    import { messages } from './ui/notifications'
    import { Styles } from 'sveltestrap';
    import Tasks from './Tasks.svelte'
    import client from './client'
</script>
<div>
    <Styles/>
    <div id="clipboard-holder" style="overflow: hidden; width:0; height: 0"></div>
    <header class="App-header">
      <h1 class="App-title">Experimaestro {$experiment ? " – " + $experiment : ""}  <i class={`fab fa-staylinked ws-status ${$connected ? "ws-link" : "ws-no-link" }`} /> </h1>
    </header>
    
    <!-- <ToastContainer toastClassName="dark-toast"/>
    <Experiments/>
     -->

    <Tasks jobs={$jobs}/>

    <Messages messages={messages}/>
  </div>