<script type="ts">
  import { Toast, ToastBody, ToastHeader } from 'sveltestrap';

  import { Messages } from 'xpm/ui/notifications'
  export let messages: Messages

  function closesnack(sid) {
    messages.update(snackbars => snackbars.filter(s => sid != s.id))
  }

</script>

<div class="messagetoast">
  {#each $messages as snackbar (snackbar.id)}
    <div class="p-3 bg-{snackbar.severity} mb-3">
      <Toast class="mr-1" color={snackbar.severity}>
        <ToastHeader toggle={() => closesnack(snackbar.id)}>{snackbar.title}</ToastHeader>
        <ToastBody>
          {snackbar.message}
        </ToastBody>
      </Toast>
    </div>
  {/each}
</div>

<style>
.messagetoast {
  position: fixed;
  top: 10px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 100;
}
</style>
