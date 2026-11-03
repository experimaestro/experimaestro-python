import json
import sys
from pathlib import Path

from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider

from tasks import Evaluate

provider = WorkspaceStateProvider(Path(sys.argv[1]))
info = provider.load_xp_info("linear")
tags = provider.get_tags_map("linear")

for job_id, config in info.jobs.items():
    if isinstance(config, Evaluate):
        mse = json.loads(config.results_path.read_text())["mse"]
        print(tags[job_id], f"mse={mse:.4f}")
