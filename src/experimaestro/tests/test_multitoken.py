"""Tests for multi-token task scheduling

Verifies that:
1. Token limits are never exceeded when using multiple different tokens
2. No deadlock occurs when tasks require combinations of tokens
3. All tasks complete successfully
"""

import threading
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field as dataclass_field
from typing import List

from experimaestro import Task, Param, field
from experimaestro.tokens import CounterToken
from experimaestro.scheduler import JobState
from .utils import TemporaryExperiment


logger = logging.getLogger(__name__)


@dataclass
class TokenUsageTracker:
    """Thread-safe tracker for token usage during task execution"""

    memory_limit: int
    cpu_limit: int
    memory_used: int = 0
    cpu_used: int = 0
    max_memory_used: int = 0
    max_cpu_used: int = 0
    violations: List[str] = dataclass_field(default_factory=list)
    lock: threading.Lock = dataclass_field(default_factory=threading.Lock)

    def acquire(self, memory: int, cpu: int, task_id: str):
        """Record token acquisition"""
        with self.lock:
            self.memory_used += memory
            self.cpu_used += cpu
            self.max_memory_used = max(self.max_memory_used, self.memory_used)
            self.max_cpu_used = max(self.max_cpu_used, self.cpu_used)

            if self.memory_used > self.memory_limit:
                self.violations.append(
                    f"Memory limit exceeded: {self.memory_used} > {self.memory_limit} (task {task_id})"
                )
            if self.cpu_used > self.cpu_limit:
                self.violations.append(
                    f"CPU limit exceeded: {self.cpu_used} > {self.cpu_limit} (task {task_id})"
                )

            logger.debug(
                "Task %s acquired: memory=%d/%d, cpu=%d/%d",
                task_id,
                self.memory_used,
                self.memory_limit,
                self.cpu_used,
                self.cpu_limit,
            )

    def release(self, memory: int, cpu: int, task_id: str):
        """Record token release"""
        with self.lock:
            self.memory_used -= memory
            self.cpu_used -= cpu
            logger.debug(
                "Task %s released: memory=%d/%d, cpu=%d/%d",
                task_id,
                self.memory_used,
                self.memory_limit,
                self.cpu_used,
                self.cpu_limit,
            )


class MultiTokenTask(Task):
    """Task that requires multiple tokens and tracks usage"""

    task_id: Param[int]
    memory_tokens: Param[int]
    cpu_tokens: Param[int]
    execution_time: Param[float] = field(default=0.01)
    tracker_path: Param[Path]

    def execute(self):
        import json

        # Use files to track concurrent usage
        usage_file = self.tracker_path.parent / f"usage_{self.task_id}.json"

        # Record start
        start_info = {
            "task_id": self.task_id,
            "memory": self.memory_tokens,
            "cpu": self.cpu_tokens,
            "start_time": time.time(),
        }

        usage_file.write_text(json.dumps(start_info))

        # Simulate work
        time.sleep(self.execution_time)

        # Record end
        start_info["end_time"] = time.time()
        usage_file.write_text(json.dumps(start_info))


def analyze_execution(tracker_path: Path, memory_limit: int, cpu_limit: int):
    """Analyze task execution logs to verify token constraints were respected"""
    import json

    usage_files = list(tracker_path.parent.glob("usage_*.json"))
    events = []

    for f in usage_files:
        try:
            data = json.loads(f.read_text())
            events.append(("start", data["start_time"], data["memory"], data["cpu"]))
            if "end_time" in data:
                events.append(("end", data["end_time"], -data["memory"], -data["cpu"]))
        except Exception as e:
            logger.warning("Could not read %s: %s", f, e)

    # Sort by time
    events.sort(key=lambda x: x[1])

    # Simulate execution
    memory_used = 0
    cpu_used = 0
    max_memory = 0
    max_cpu = 0
    violations = []

    for _event_type, timestamp, memory_delta, cpu_delta in events:
        memory_used += memory_delta
        cpu_used += cpu_delta
        max_memory = max(max_memory, memory_used)
        max_cpu = max(max_cpu, cpu_used)

        if memory_used > memory_limit:
            violations.append(f"Memory exceeded at {timestamp}: {memory_used}")
        if cpu_used > cpu_limit:
            violations.append(f"CPU exceeded at {timestamp}: {cpu_used}")

    return max_memory, max_cpu, violations


def test_multitoken_basic():
    """Test basic multi-token scheduling with two tokens"""
    import json

    with TemporaryExperiment("multitoken", timeout_multiplier=3.0) as xp:
        # Create two tokens: memory (4 units) and cpu (2 cores)
        memory_limit = 4
        cpu_limit = 2

        memory_token = CounterToken("memory", xp.workdir / "token_memory", memory_limit)
        cpu_token = CounterToken("cpu", xp.workdir / "token_cpu", cpu_limit)

        # Create tracker info file
        tracker_path = xp.workdir / "tracker.json"
        tracker_path.write_text(
            json.dumps({"memory_limit": memory_limit, "cpu_limit": cpu_limit})
        )

        tasks = []

        # Create tasks with various token requirements:
        # - Some need 1 memory + 1 cpu
        # - Some need 2 memory + 1 cpu
        # - Some need 1 memory + 2 cpu
        task_configs = [
            (1, 1),  # Task 0: 1 mem, 1 cpu
            (2, 1),  # Task 1: 2 mem, 1 cpu
            (1, 2),  # Task 2: 1 mem, 2 cpu (uses all CPUs)
            (1, 1),  # Task 3: 1 mem, 1 cpu
            (2, 1),  # Task 4: 2 mem, 1 cpu
            (1, 1),  # Task 5: 1 mem, 1 cpu
            (3, 1),  # Task 6: 3 mem, 1 cpu
            (1, 1),  # Task 7: 1 mem, 1 cpu
        ]

        for i, (mem, cpu) in enumerate(task_configs):
            task = MultiTokenTask.C(
                task_id=i,
                memory_tokens=mem,
                cpu_tokens=cpu,
                execution_time=0.05,
                tracker_path=tracker_path,
            )
            task.add_dependencies(memory_token.dependency(mem))
            task.add_dependencies(cpu_token.dependency(cpu))
            tasks.append(task.submit())

        # Wait for all tasks to complete
        xp.wait()

        # Verify all tasks completed successfully
        for i, task in enumerate(tasks):
            state = task.__xpm__.job.state
            assert state == JobState.DONE, f"Task {i} ended with state {state}"

        # Analyze execution to verify token limits
        max_mem, max_cpu, violations = analyze_execution(
            tracker_path, memory_limit, cpu_limit
        )

        logger.info("Max memory used: %d/%d", max_mem, memory_limit)
        logger.info("Max CPU used: %d/%d", max_cpu, cpu_limit)

        assert not violations, f"Token violations detected: {violations}"
        assert max_mem <= memory_limit, f"Memory limit exceeded: {max_mem}"
        assert max_cpu <= cpu_limit, f"CPU limit exceeded: {max_cpu}"


def test_multitoken_stress():
    """Stress test with many tasks requiring various token combinations

    This test submits many tasks concurrently to stress the token
    acquisition/release mechanism and verify no deadlocks occur.
    """
    import json

    with TemporaryExperiment("multitoken_stress", timeout_multiplier=6.0) as xp:
        # Create tokens with limited capacity to force contention
        memory_limit = 8
        cpu_limit = 4

        memory_token = CounterToken(
            "memory_stress", xp.workdir / "token_memory", memory_limit
        )
        cpu_token = CounterToken("cpu_stress", xp.workdir / "token_cpu", cpu_limit)

        tracker_path = xp.workdir / "tracker.json"
        tracker_path.write_text(
            json.dumps({"memory_limit": memory_limit, "cpu_limit": cpu_limit})
        )

        tasks = []
        num_tasks = 20

        # Generate diverse task configurations
        import random

        random.seed(42)  # Reproducible

        for i in range(num_tasks):
            # Random token requirements within limits
            mem = random.randint(1, min(4, memory_limit))
            cpu = random.randint(1, min(2, cpu_limit))

            task = MultiTokenTask.C(
                task_id=i,
                memory_tokens=mem,
                cpu_tokens=cpu,
                execution_time=random.uniform(0.01, 0.05),
                tracker_path=tracker_path,
            )
            task.add_dependencies(memory_token.dependency(mem))
            task.add_dependencies(cpu_token.dependency(cpu))
            tasks.append(task.submit())

        # Wait for all tasks - if there's a deadlock, this will timeout
        xp.wait()

        # Verify all tasks completed
        completed = sum(1 for t in tasks if t.__xpm__.job.state == JobState.DONE)
        assert completed == num_tasks, f"Only {completed}/{num_tasks} tasks completed"

        # Analyze execution
        max_mem, max_cpu, violations = analyze_execution(
            tracker_path, memory_limit, cpu_limit
        )

        logger.info(
            "Stress test: max memory=%d/%d, max cpu=%d/%d",
            max_mem,
            memory_limit,
            max_cpu,
            cpu_limit,
        )

        assert not violations, f"Token violations: {violations}"


def test_multitoken_large_requirements():
    """Test tasks that require most of the available tokens

    Ensures that tasks requiring large token counts still complete
    and don't cause deadlock when competing for resources.
    """
    import json

    with TemporaryExperiment("multitoken_large", timeout_multiplier=3.0) as xp:
        memory_limit = 10
        cpu_limit = 4

        memory_token = CounterToken(
            "memory_large", xp.workdir / "token_memory", memory_limit
        )
        cpu_token = CounterToken("cpu_large", xp.workdir / "token_cpu", cpu_limit)

        tracker_path = xp.workdir / "tracker.json"
        tracker_path.write_text(
            json.dumps({"memory_limit": memory_limit, "cpu_limit": cpu_limit})
        )

        tasks = []

        # Mix of large and small tasks
        task_configs = [
            (8, 3),  # Large: needs most resources
            (1, 1),  # Small
            (7, 2),  # Large
            (2, 1),  # Small
            (9, 4),  # Very large: needs almost all
            (1, 1),  # Small
            (5, 2),  # Medium
            (3, 1),  # Medium
        ]

        for i, (mem, cpu) in enumerate(task_configs):
            task = MultiTokenTask.C(
                task_id=i,
                memory_tokens=mem,
                cpu_tokens=cpu,
                execution_time=0.03,
                tracker_path=tracker_path,
            )
            task.add_dependencies(memory_token.dependency(mem))
            task.add_dependencies(cpu_token.dependency(cpu))
            tasks.append(task.submit())

        xp.wait()

        # Verify completion
        for i, task in enumerate(tasks):
            state = task.__xpm__.job.state
            assert state == JobState.DONE, f"Task {i} state: {state}"

        max_mem, max_cpu, violations = analyze_execution(
            tracker_path, memory_limit, cpu_limit
        )

        assert not violations, f"Violations: {violations}"
        assert max_mem <= memory_limit
        assert max_cpu <= cpu_limit


def test_multitoken_single_task_all_tokens():
    """Test that a single task can acquire all available tokens"""
    import json

    with TemporaryExperiment("multitoken_all", timeout_multiplier=1.5) as xp:
        memory_limit = 4
        cpu_limit = 2

        memory_token = CounterToken(
            "memory_all", xp.workdir / "token_memory", memory_limit
        )
        cpu_token = CounterToken("cpu_all", xp.workdir / "token_cpu", cpu_limit)

        tracker_path = xp.workdir / "tracker.json"
        tracker_path.write_text(
            json.dumps({"memory_limit": memory_limit, "cpu_limit": cpu_limit})
        )

        # Single task requiring all tokens
        task = MultiTokenTask.C(
            task_id=0,
            memory_tokens=memory_limit,
            cpu_tokens=cpu_limit,
            execution_time=0.02,
            tracker_path=tracker_path,
        )
        task.add_dependencies(memory_token.dependency(memory_limit))
        task.add_dependencies(cpu_token.dependency(cpu_limit))
        submitted = task.submit()

        xp.wait()

        assert submitted.__xpm__.job.state == JobState.DONE


def test_multitoken_sequential_dependency():
    """Test that tasks waiting on tokens eventually run when tokens free up

    Submits tasks that together require more tokens than available,
    verifying they run sequentially without deadlock.
    """
    import json

    with TemporaryExperiment("multitoken_seq", timeout_multiplier=3.0) as xp:
        memory_limit = 2
        cpu_limit = 1

        memory_token = CounterToken(
            "memory_seq", xp.workdir / "token_memory", memory_limit
        )
        cpu_token = CounterToken("cpu_seq", xp.workdir / "token_cpu", cpu_limit)

        tracker_path = xp.workdir / "tracker.json"
        tracker_path.write_text(
            json.dumps({"memory_limit": memory_limit, "cpu_limit": cpu_limit})
        )

        tasks = []

        # All tasks require all tokens - must run one at a time
        for i in range(5):
            task = MultiTokenTask.C(
                task_id=i,
                memory_tokens=memory_limit,
                cpu_tokens=cpu_limit,
                execution_time=0.02,
                tracker_path=tracker_path,
            )
            task.add_dependencies(memory_token.dependency(memory_limit))
            task.add_dependencies(cpu_token.dependency(cpu_limit))
            tasks.append(task.submit())

        xp.wait()

        # All should complete
        for i, task in enumerate(tasks):
            assert task.__xpm__.job.state == JobState.DONE, f"Task {i} failed"

        # Verify sequential execution (max usage should equal limits)
        max_mem, max_cpu, violations = analyze_execution(
            tracker_path, memory_limit, cpu_limit
        )

        assert not violations
        # Since all tasks need all tokens, they must run sequentially
        assert max_mem <= memory_limit
        assert max_cpu <= cpu_limit
