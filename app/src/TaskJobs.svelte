<script>
    import { createEventDispatcher } from 'svelte'
    
    const dispatch = createEventDispatcher()
    import CopyToClipboard from "svelte-copy-to-clipboard";
    import { success, error } from 'xpm/ui/notifications'
    export let job

</script>

<div class="resource">
                    
    {#if job.status === "running"}
        <span class="status progressbar-container" title={`${job.progress*100}%`}>
            <span style={`right: ${(1-job.progress)*100}%`} class="progressbar"></span><div class="status-running">{job.status}</div>
        </span> 
        <i class="fa fa-skull-crossbones action" on:click={() => { dispatch('kill', job) } }/>
    {:else}
        <span class={`status status-${job.status}`}>{job.status}</span>
    {/if}

<i class="fas fa-eye action" title="Details" on:click={() =>  { dispatch('show', job) }}/>
<span class="job-id"><CopyToClipboard let:copy={onCopy} text={`${job.taskId}/${job.jobId}`} on:copy={() => success(`Job path copied`)} on:fail={() => error(`Error copying job path`)}><span class="clipboard" on:click={onCopy}>{job.taskId}</span></CopyToClipboard></span>
    {#each job.tags as tag}
     <span class="tag">
            <span class="name">{tag[0]}</span><span class="value">{tag[1]}</span>
        </span>
    {/each}
</div>  