<script lang="ts">
    import {
        Button, 
        Modal, ModalBody, ModalFooter, ModalHeader,
        Form, FormGroup, FormText, Input, Label
    } from 'sveltestrap';

    import TaskJobs from './TaskJobs.svelte'
    import TaskDetail from './TaskDetail.svelte'
    import client from './client';
    import { info } from './ui/notifications';

    export let jobs

    let searchtask : RegExp | null = null
    let tags : any[] = []
    function _jobfilter(job) {
        if (searchtask && job.taskId.match(searchtask) === null) return false;

        mainloop: for(let {tag, value} of tags) {
            for(let tv of job.tags) {
                if (tv[0].search(tag) !== -1 && tv[1].toString().search(value) !== -1)
                    continue mainloop;
            }
            return false
        }

        return true;
    }
    let jobfilter = _jobfilter

    function updateTaskSearch(e) {
        const inputValue = (e.target as HTMLInputElement).value
        searchtask = inputValue !== "" ? new RegExp(inputValue) : null
        jobfilter = _jobfilter
    }

    function updateTagSearch(e) {
        const tag = (e.target as HTMLInputElement).value
        let re = /(\S+):(?:([^"]\S*)|"([^"]+)")\s*/g;
        var match : any[] | null
        tags = []
        while ((match = re.exec(tag)) !== null) {
            tags.push({ tag: match[1], value: match[2] });
        }
        console.log("Tags", tags)
        jobfilter = _jobfilter

    }


    let killJob = null;

    function cancelKill() {
        killJob = null; 
        info("Action cancelled"); 
    }

    function kill() {
        if (killJob !== null) {
            client.send({ type: "kill", payload: killJob.jobId}, "cannot kill job " + killJob.jobId)
            killJob = null
        }
    }

    let showJob
</script>

<div id="resources">
    <div class="search">
        <div style="display: flex">
            <FormGroup>
                <Label for="searchtask">Task</Label>
                <Input id="searchtask" on:input={updateTaskSearch} placeholder="Filter task"/>
              </FormGroup>
              <FormGroup>
                <Label for="searchtags">Tags</Label>
                <Input id="searchtags" on:input={updateTagSearch} placeholder="Format tag:value..."/>
              </FormGroup>
        </div>
    </div>

    <Modal isOpen={killJob != null}>
        <ModalHeader>Are you sure?</ModalHeader>
        <ModalBody>
            Are you sure to kill job <b>{killJob.taskId}</b>?
        </ModalBody>
        <ModalFooter>
            <Button default on:click={cancelKill}>Cancel</Button>
            <Button on:click={kill}>OK</Button>
        </ModalFooter>
      </Modal>
      
      
      

    {#each jobs.ids as jobId (jobId)}
        {#if jobfilter(jobs.byId[jobId])}
            <TaskJobs job={jobs.byId[jobId]} on:kill={(event) => { killJob = event.detail }} on:show={event => { 
                showJob = showJob == event.detail ? null : event.detail
            }}/>
            {#if showJob && showJob.jobId == jobId}
                <TaskDetail job={jobs.byId[jobId]}/>
            {/if}
        {/if}
    {/each}
</div>
