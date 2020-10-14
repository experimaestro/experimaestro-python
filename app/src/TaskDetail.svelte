<script>
  import {DateTime} from 'luxon'
  export let job
  import CopyToClipboard from "svelte-copy-to-clipboard";
  import { success, error } from 'xpm/ui/notifications'

  function formatms(t) {
    DateTime.fromMillis(1000 * t).toLocaleString(DateTime.DATETIME_FULL_WITH_SECONDS)
  }
</script>    

<div class="details">
<span class="what">Status</span><div>{job.status}</div>
<span class="what">Path</span><div>
  <CopyToClipboard let:copy={onCopy} text={job.locator} on:copy={() => success(`Job path copied`)} on:fail={() => error(`Error copying job path`)}><span class="clipboard" on:click={onCopy}>{job.locator}</span></CopyToClipboard> 
</div>
<span class="what">Submitted</span><div>{formatms(job.submitted)} </div>
<span class="what">Start</span><div>{formatms(job.start)}</div>
<span class="what">End</span><div>{formatms(job.end)}</div>
<span class="what">Tags</span><div>
  {#each job.tags as tag (tag[0])}
  <span class="tag">
    <span class="name">{tag[0]}</span><span class="value">{tag[1]}</span>
  </span>
  {/each}
  </div>
</div>

<style>
  .details {
    border: 1px solid black;
    display: grid;
    grid-template-columns: 1fr 3fr;
    padding: 5px;
    margin: 5px;
  }
</style>