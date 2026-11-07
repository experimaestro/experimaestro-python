# Carbon footprint of a project

Experimaestro records the energy used and the CO₂ emitted by every job, and can
add them up for a whole project. Install one extra dependency: your tasks need
no change.

```bash
pip install "experimaestro[carbon]"
```

## Why track it?

A training run is easy to estimate. The footprint of a project is not: it also
includes the sweeps, the attempts that crashed and the evaluations run again and
again. Since experimaestro launches all these jobs, it can account for them:

- **Know the cost of a project**, not just of its final run, and have a figure to
  give when a paper, a funder or an institution asks for one.
- **Weigh a result against its cost.** A small gain in a metric reads differently
  next to the emissions it took; comparing tasks and experiments shows where the
  energy actually goes.
- **See what failures cost.** Failed jobs and retries are recorded too, so you can
  tell how much energy went to attempts that produced nothing.
- **Benefit from reuse.** Job identifiers already avoid running the same
  computation twice: a reused job adds no new emissions, and an experiment sharing
  it does not count it again.

## A project, step by step

### 1. Declare the group

A group of experiments is an [experiment tag](../interfaces.md#experiment-tags).
Add the same tag to the YAML file of each experiment of the project:

```yaml
# retrieval-dense.yaml
id: retrieval-dense
file: experiment
tags: [project-x]
```

An experiment can be in several groups, and `experiment(..., tags=["project-x"])`
does the same in a Python script. Experiments run before you added the tag join
the group when they next run.

### 2. Look at the whole group

```console
$ experimaestro experiments carbon summary --group project-x -v --by-task
Carbon Impact Summary
--------------------------------------------------
  Group: project-x
  Mode: All runs (including retries)

Total Emissions:
  CO2:      6.70kg
  Energy:   134.00kWh
  Duration: 15.3 hours
  Jobs:     4 (unique: 4)

Per-Experiment Breakdown:
  retrieval-bm25: 400.0g (2 jobs)
  retrieval-dense: 6.60kg (3 jobs)
  (a job shared by several experiments is listed in each of them)

Per-Task Breakdown:
  mypkg.Train: 6.10kg (1 jobs)
  mypkg.Evaluate: 300.0g (2 jobs)
  mypkg.Preprocess: 300.0g (1 jobs)
```

The total counts each job once. Here both experiments use the same preprocessing
job, which is why their details (400 g + 6.60 kg) add up to more than the total.

- `-v` adds the details per experiment, and `--by-task` the details per type of
  task;
- `--json` gives the same information to a script or a report;
- `--group` can be repeated: experiments in any of these groups are selected.

Other selections combine with it: `--name` takes a regular expression on the
experiment IDs (useful for experiments that have no tags), and `--tag` restricts
to the [runs you tagged](../cli.md#experiment-management), such as those behind a
paper. Without any filter, the whole workspace is summed up.

### 3. See what the retries cost

By default, all the runs of each job are counted, as they all consumed energy.
`--latest-only` keeps only the last run of each job:

```console
$ experimaestro experiments carbon summary --group project-x --latest-only
...
Total Emissions:
  CO2:      4.70kg
```

The difference, 2 kg here, is what earlier attempts of these jobs emitted. The
second figure is closer to what reproducing the results would take; the first is
what the project really emitted.

### 4. Follow it in the monitoring interfaces

In the [TUI and the web UI](../interfaces.md), group the experiments by tag (`g`):
the header of each group shows the total of its experiments, computed as by
`carbon summary` (all the runs, each job counted once), and expanding the group
shows the CO₂ of each experiment. An experiment's own figure covers the latest run
of each of its jobs, so the header can be higher than the sum of the lines. The
details of a job show its energy and power too.

All of this works on [remote workspaces](../settings.md#remote-workspaces-ssh):
`experimaestro --ssh HOST:PATH experiments carbon summary --group project-x`. The
group totals in the UIs need experimaestro to be up to date on the remote host.

## Setup

Tracking starts on its own when the extra is installed; otherwise, experiments
start with a warning. Records are kept in the `carbon/` folder of the workspace,
with one record per job run.

The [settings file](../settings.md#carbon-tracking) configures it:

```yaml
carbon:
  # false turns tracking off for all the jobs
  enabled: true
  # Country used for the carbon intensity, instead of detecting it
  country_iso_code: FRA
```

| Platform | Backend | Status |
|----------|---------|--------|
| Linux x86-64, Windows | [CodeCarbon](https://mlco2.github.io/codecarbon/) (Intel RAPL, NVIDIA GPUs) | Main path |
| Linux ARM with an NVIDIA GPU | CodeCarbon | Supported |
| macOS on Apple Silicon | [Zeus](https://ml.energy/zeus) | Experimental, [see below](#apple-silicon) |
| Linux ARM without NVIDIA GPU, macOS on Intel | — | Not available |

## How far to trust the figures

These are **estimates, good for comparing runs and for orders of magnitude**.
Compare figures obtained on similar hardware and in the same region, and quote
them as estimates.

- **What is covered:** the energy of the CPU, GPU and RAM while a job runs,
  converted to CO₂ with the carbon intensity of a country.
- **What is not covered:** data-centre overhead such as cooling, manufacturing the
  hardware, storage, network, and the experiment process itself. The real
  footprint is higher.
- **Carbon intensity:** it is an average for the country. On a cluster, where IP
  detection can be misleading, set `country_iso_code`; check the `region` field of
  the records.
- **Measured or estimated:** CodeCarbon falls back on estimates when it cannot read
  the hardware counters (e.g. restricted access to RAPL), and on a node shared with
  other users the figures may not be exactly those of your job. See
  [its methodology](https://mlco2.github.io/codecarbon/methodology.html).
- **Removed runs:** a group or a filter finds its jobs through the runs still on
  disk, so what only ran in a run that was deleted (e.g. by the automatic cleanup
  of old runs) is no longer counted.

(apple-silicon)=
### Apple Silicon

Tracking on Apple Silicon is **experimental, and can largely undercount**:

- The figures come from counters of the whole chip, so they also include what else
  the machine is doing.
- On an M4 Max with macOS 27.2 on battery, these counters did not change under CPU
  load, except the GPU one: a job using the CPU was recorded with
  `cpu_power_w: 0` and almost no energy. Look at this value in
  `carbon/measures-*.jsonl` before relying on the results.
- The country is not detected: without `country_iso_code`, a world average of
  450 g of CO₂ per kWh is used, as for a country not known to the tracker.
- Installing `zeus[apple]` builds `scipy` if it has no wheels for your Python
  version (as for 3.15 at the time of writing), which requires a Fortran compiler.

## Turning it off

Set `enabled: false` in the `carbon` section of the settings. To only silence the
warning when the extra is not installed, set `warn_if_unavailable: false`;
`experiment(..., no_environmental_impact=True)` also removes the warning and the
experiment's consolidated figures, but does not stop the jobs from being tracked.
