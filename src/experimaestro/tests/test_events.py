"""Tests for event writing and reading system

This module tests the core event system:
- EventWriter: Writing events to JSONL files with rotation and status tracking
- EventReader: Reading events from JSONL files with ordering guarantees
"""

import json
import weakref
from pathlib import Path

import pytest

from experimaestro.scheduler.state_status import (
    EventBase,
    EventReader,
    EventWriter,
    JobProgressEvent,
    JobStateChangedEvent,
    WatchedDirectory,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class SimpleEventWriter(EventWriter):
    """Simple EventWriter subclass for testing"""

    def __init__(
        self,
        events_dir: Path,
        status_path: Path | None = None,
        initial_count: int = 0,
    ):
        super().__init__(initial_count)
        self._events_dir = events_dir
        self._status_path = status_path

    @property
    def events_dir(self) -> Path:
        return self._events_dir

    @property
    def status_path(self) -> Path | None:
        return self._status_path


@pytest.fixture
def events_dir(tmp_path):
    """Create a temporary events directory"""
    events_dir = tmp_path / ".events" / "test"
    events_dir.mkdir(parents=True)
    return events_dir


@pytest.fixture
def status_file(events_dir):
    """Create a status file for testing"""
    status_path = events_dir / "status.json"
    status_path.write_text(json.dumps({"state": "running"}))
    return status_path


# =============================================================================
# Tests: EventWriter Basics
# =============================================================================


class TestEventWriterBasics:
    """Basic tests for EventWriter functionality"""

    def test_creates_event_file_on_first_write(self, events_dir):
        """EventWriter should create the events directory and file on first write"""
        writer = SimpleEventWriter(events_dir)
        assert not (events_dir / "events-0.jsonl").exists()

        try:
            event = JobStateChangedEvent(job_id="test-job", state="running")
            writer.write_event(event)

            assert (events_dir / "events-0.jsonl").exists()

            # Read and verify
            with (events_dir / "events-0.jsonl").open() as f:
                line = f.readline()
                data = json.loads(line)
                assert data["job_id"] == "test-job"
                assert data["state"] == "running"
        finally:
            writer.close()

    def test_appends_to_existing_file(self, events_dir):
        """Multiple events should be appended to the same file"""
        writer = SimpleEventWriter(events_dir)
        try:
            for i in range(3):
                event = JobStateChangedEvent(job_id=f"job-{i}", state="running")
                writer.write_event(event)

            # Verify all events written
            with (events_dir / "events-0.jsonl").open() as f:
                lines = f.readlines()
            assert len(lines) == 3
        finally:
            writer.close()


# =============================================================================
# Tests: EventWriter Rotation
# =============================================================================


class TestEventWriterRotation:
    """Tests for EventWriter file rotation"""

    def test_rotates_at_max_events(self, events_dir):
        """EventWriter should rotate to new file after MAX_EVENTS_PER_FILE"""

        class SmallRotationWriter(SimpleEventWriter):
            MAX_EVENTS_PER_FILE = 3

        writer = SmallRotationWriter(events_dir)
        try:
            # Write 7 events (should create 3 files: 0, 1, 2)
            for i in range(7):
                event = JobStateChangedEvent(job_id=f"job-{i}", state="running")
                writer.write_event(event)

            # Verify files
            assert (events_dir / "events-0.jsonl").exists()
            assert (events_dir / "events-1.jsonl").exists()
            assert (events_dir / "events-2.jsonl").exists()

            # Check event counts
            with (events_dir / "events-0.jsonl").open() as f:
                assert len(f.readlines()) == 3
            with (events_dir / "events-1.jsonl").open() as f:
                assert len(f.readlines()) == 3
            with (events_dir / "events-2.jsonl").open() as f:
                assert len(f.readlines()) == 1

        finally:
            writer.close()

    def test_updates_status_on_first_write(self, events_dir, status_file):
        """EventWriter should update status.json with events_count on first write"""
        writer = SimpleEventWriter(events_dir, status_path=status_file)
        try:
            event = JobStateChangedEvent(job_id="test-job", state="running")
            writer.write_event(event)

            # Check status updated
            with status_file.open() as f:
                status = json.load(f)
            assert status.get("events_count") == 0
        finally:
            writer.close()

    def test_updates_status_on_rotation(self, events_dir, status_file):
        """EventWriter should update status.json with new events_count on rotation"""

        class SmallRotationWriter(SimpleEventWriter):
            MAX_EVENTS_PER_FILE = 3

        writer = SmallRotationWriter(events_dir, status_path=status_file)
        try:
            # Write 4 events (triggers rotation after 3)
            for i in range(4):
                event = JobStateChangedEvent(job_id=f"job-{i}", state="running")
                writer.write_event(event)

            # Check status has updated count
            with status_file.open() as f:
                status = json.load(f)
            assert status.get("events_count") == 1
        finally:
            writer.close()


# =============================================================================
# Tests: EventReader Basics
# =============================================================================


class TestEventReaderBasics:
    """Basic tests for EventReader functionality"""

    def test_reads_events_from_file(self, events_dir):
        """EventReader should read events from JSONL files"""
        # Create event file
        event_file = events_dir / "events-0.jsonl"
        with event_file.open("w") as f:
            for i in range(3):
                event = JobStateChangedEvent(job_id=f"job-{i}", state="running")
                f.write(event.to_json() + "\n")

        # Create reader
        reader = EventReader(
            WatchedDirectory(path=events_dir, glob_pattern="events-*.jsonl")
        )

        # Read events
        events = reader.read_new_events()
        assert len(events) == 3

        # Verify event content
        for i, (entity_id, event) in enumerate(events):
            assert isinstance(event, JobStateChangedEvent)
            assert event.job_id == f"job-{i}"

    def test_incremental_reading(self, events_dir):
        """EventReader should only read new events on subsequent calls"""
        event_file = events_dir / "events-0.jsonl"

        # Write initial events
        with event_file.open("w") as f:
            for i in range(3):
                event = JobStateChangedEvent(job_id=f"job-{i}", state="running")
                f.write(event.to_json() + "\n")

        reader = EventReader(
            WatchedDirectory(path=events_dir, glob_pattern="events-*.jsonl")
        )

        # First read
        events = reader.read_new_events()
        assert len(events) == 3

        # Second read - no new events
        events = reader.read_new_events()
        assert len(events) == 0

        # Add more events
        with event_file.open("a") as f:
            for i in range(3, 6):
                event = JobStateChangedEvent(job_id=f"job-{i}", state="done")
                f.write(event.to_json() + "\n")

        # Third read - only new events
        events = reader.read_new_events()
        assert len(events) == 3
        assert all(e.state == "done" for _, e in events)


# =============================================================================
# Tests: EventReader Ordering (using _process_file_change directly)
# =============================================================================


class TestEventReaderOrdering:
    """Tests for EventReader ordering guarantees using _process_file_change directly"""

    def test_processes_files_in_order(self, events_dir):
        """When processing file N, all earlier files should be processed first"""
        # Create multiple event files
        for file_num in range(3):
            event_file = events_dir / f"events-{file_num}.jsonl"
            with event_file.open("w") as f:
                for i in range(3):
                    event = JobStateChangedEvent(
                        job_id=f"file{file_num}-job-{i}", state="running"
                    )
                    f.write(event.to_json() + "\n")

        # Track received events
        received_events: list[str] = []

        def on_event(entity_id: str, event: EventBase):
            if isinstance(event, JobStateChangedEvent):
                received_events.append(event.job_id)

        reader = EventReader(
            WatchedDirectory(
                path=events_dir,
                glob_pattern="events-*.jsonl",
                on_event=on_event,
                events_count_resolver=lambda _: 0,
            )
        )

        # Register entity to be followed
        reader._followed_entities["test"] = reader.directories[0]

        # Process file 2 directly (should process 0 and 1 first)
        reader._process_file_change(events_dir / "events-2.jsonl")

        # Verify ordering: all file0 events, then file1, then file2
        assert len(received_events) == 9

        # Check file order
        file0_events = [e for e in received_events if e.startswith("file0-")]
        file1_events = [e for e in received_events if e.startswith("file1-")]
        file2_events = [e for e in received_events if e.startswith("file2-")]

        assert len(file0_events) == 3
        assert len(file1_events) == 3
        assert len(file2_events) == 3

        # Verify ordering: file0 events come before file1, file1 before file2
        file0_last_index = max(received_events.index(e) for e in file0_events)
        file1_first_index = min(received_events.index(e) for e in file1_events)
        file1_last_index = max(received_events.index(e) for e in file1_events)
        file2_first_index = min(received_events.index(e) for e in file2_events)

        assert file0_last_index < file1_first_index, (
            "File0 events should come before file1"
        )
        assert file1_last_index < file2_first_index, (
            "File1 events should come before file2"
        )

    def test_skips_files_below_events_count(self, events_dir):
        """Files below events_count from resolver should be skipped"""
        # Create event files
        for file_num in range(3):
            event_file = events_dir / f"events-{file_num}.jsonl"
            with event_file.open("w") as f:
                for i in range(3):
                    event = JobStateChangedEvent(
                        job_id=f"file{file_num}-job-{i}", state="running"
                    )
                    f.write(event.to_json() + "\n")

        # Track received events
        received_events: list[str] = []

        def on_event(entity_id: str, event: EventBase):
            if isinstance(event, JobStateChangedEvent):
                received_events.append(event.job_id)

        # Resolver returns 1 -> skip file 0
        reader = EventReader(
            WatchedDirectory(
                path=events_dir,
                glob_pattern="events-*.jsonl",
                on_event=on_event,
                events_count_resolver=lambda _: 1,
            )
        )

        # Register entity to be followed
        reader._followed_entities["test"] = reader.directories[0]

        # Process file 2 (should skip file 0, process file 1 first)
        reader._process_file_change(events_dir / "events-2.jsonl")

        # Should only have events from files 1 and 2
        assert len(received_events) == 6

        file0_events = [e for e in received_events if e.startswith("file0-")]
        file1_events = [e for e in received_events if e.startswith("file1-")]
        file2_events = [e for e in received_events if e.startswith("file2-")]

        assert len(file0_events) == 0, "File0 events should be skipped"
        assert len(file1_events) == 3
        assert len(file2_events) == 3

    def test_handles_missing_earlier_files(self, events_dir):
        """Should handle case where earlier files don't exist"""
        # Create only file 2
        event_file = events_dir / "events-2.jsonl"
        with event_file.open("w") as f:
            for i in range(3):
                event = JobStateChangedEvent(job_id=f"job-{i}", state="running")
                f.write(event.to_json() + "\n")

        # Track received events
        received_events: list[str] = []

        def on_event(entity_id: str, event: EventBase):
            if isinstance(event, JobStateChangedEvent):
                received_events.append(event.job_id)

        reader = EventReader(
            WatchedDirectory(
                path=events_dir,
                glob_pattern="events-*.jsonl",
                on_event=on_event,
                events_count_resolver=lambda _: 0,
            )
        )

        # Register entity to be followed
        reader._followed_entities["test"] = reader.directories[0]

        # Process file 2 (files 0 and 1 don't exist)
        reader._process_file_change(events_dir / "events-2.jsonl")

        # Should still get events from file 2
        assert len(received_events) == 3

    def test_job_event_files_extract_file_number(self, events_dir):
        """_extract_file_number should handle job-style event-{job_id}-{count}.jsonl"""
        reader = EventReader(
            WatchedDirectory(path=events_dir, glob_pattern="event-*.jsonl")
        )

        # Job event files
        assert reader._extract_file_number(events_dir / "event-abc123-0.jsonl") == 0
        assert reader._extract_file_number(events_dir / "event-abc123-5.jsonl") == 5
        assert reader._extract_file_number(events_dir / "event-abc123-12.jsonl") == 12

        # Experiment event files (still works)
        assert reader._extract_file_number(events_dir / "events-0.jsonl") == 0
        assert reader._extract_file_number(events_dir / "events-3.jsonl") == 3

        # Non-matching files
        assert reader._extract_file_number(events_dir / "other.jsonl") is None

    def test_job_event_files_extract_file_prefix(self, events_dir):
        """_extract_file_prefix should handle both naming patterns"""
        reader = EventReader(
            WatchedDirectory(path=events_dir, glob_pattern="event-*.jsonl")
        )

        assert (
            reader._extract_file_prefix(events_dir / "event-abc123-0.jsonl")
            == "event-abc123-"
        )
        assert reader._extract_file_prefix(events_dir / "events-3.jsonl") == "events-"
        assert reader._extract_file_prefix(events_dir / "other.jsonl") is None

    def test_job_event_files_ordering(self, events_dir):
        """Job event files (event-{job_id}-{count}.jsonl) should be processed in order"""
        job_id = "abc123"

        # Create job event files
        for file_num in range(3):
            event_file = events_dir / f"event-{job_id}-{file_num}.jsonl"
            with event_file.open("w") as f:
                for i in range(3):
                    event = JobStateChangedEvent(
                        job_id=f"file{file_num}-job-{i}", state="running"
                    )
                    f.write(event.to_json() + "\n")

        received_events: list[str] = []

        def on_event(entity_id: str, event: EventBase):
            if isinstance(event, JobStateChangedEvent):
                received_events.append(event.job_id)

        reader = EventReader(
            WatchedDirectory(
                path=events_dir,
                glob_pattern=f"event-{job_id}-*.jsonl",
                on_event=on_event,
            )
        )

        # Register entity to be followed
        reader._followed_entities["test"] = reader.directories[0]

        # Process file 2 directly (should process 0 and 1 first)
        reader._process_file_change(events_dir / f"event-{job_id}-2.jsonl")

        # All 9 events should be received
        assert len(received_events) == 9

        # Verify ordering: file0 before file1 before file2
        file0_events = [e for e in received_events if e.startswith("file0-")]
        file1_events = [e for e in received_events if e.startswith("file1-")]
        file2_events = [e for e in received_events if e.startswith("file2-")]

        assert len(file0_events) == 3
        assert len(file1_events) == 3
        assert len(file2_events) == 3

        file0_last = max(received_events.index(e) for e in file0_events)
        file1_first = min(received_events.index(e) for e in file1_events)
        file1_last = max(received_events.index(e) for e in file1_events)
        file2_first = min(received_events.index(e) for e in file2_events)

        assert file0_last < file1_first, "File0 events should come before file1"
        assert file1_last < file2_first, "File1 events should come before file2"

    def test_job_event_files_rotation_drains_previous(self, events_dir):
        """When a new job event file appears, previous file should be fully drained"""
        job_id = "deadbeef"

        # Simulate: file 0 has events that haven't been fully read,
        # then file 1 appears (rotation happened)
        event_file_0 = events_dir / f"event-{job_id}-0.jsonl"
        with event_file_0.open("w") as f:
            for i in range(5):
                event = JobProgressEvent(job_id=job_id, level=0, progress=i / 10.0)
                f.write(event.to_json() + "\n")

        event_file_1 = events_dir / f"event-{job_id}-1.jsonl"
        with event_file_1.open("w") as f:
            event = JobProgressEvent(job_id=job_id, level=0, progress=0.5)
            f.write(event.to_json() + "\n")

        received_events: list[EventBase] = []

        def on_event(entity_id: str, event: EventBase):
            received_events.append(event)

        reader = EventReader(
            WatchedDirectory(
                path=events_dir,
                glob_pattern=f"event-{job_id}-*.jsonl",
                on_event=on_event,
            )
        )

        reader._followed_entities["test"] = reader.directories[0]

        # Only process file 1 (simulates watchdog notifying about new file)
        # This should also drain file 0 first
        reader._process_file_change(event_file_1)

        # Should get all 5 events from file 0 + 1 from file 1 = 6 total
        assert len(received_events) == 6

        # The last event should be from file 1 (progress=0.5)
        last = received_events[-1]
        assert isinstance(last, JobProgressEvent)
        assert last.progress == 0.5


# =============================================================================
# Tests: EventReader Stress
# =============================================================================


class TestEventReaderStress:
    """Stress tests for EventReader using _process_file_change directly"""

    def test_many_files_ordering(self, events_dir):
        """Verify ordering with many event files"""
        num_files = 20
        events_per_file = 10

        # Create many event files
        for file_num in range(num_files):
            event_file = events_dir / f"events-{file_num}.jsonl"
            with event_file.open("w") as f:
                for i in range(events_per_file):
                    event = JobProgressEvent(
                        job_id=f"job-{file_num}-{i}",
                        level=0,
                        progress=i / events_per_file,
                    )
                    f.write(event.to_json() + "\n")

        # Track received events with their file numbers
        received_file_nums: list[int] = []

        def on_event(entity_id: str, event: EventBase):
            if isinstance(event, JobProgressEvent):
                # Extract file number from job_id: "job-{file_num}-{i}"
                parts = event.job_id.split("-")
                file_num = int(parts[1])
                received_file_nums.append(file_num)

        reader = EventReader(
            WatchedDirectory(
                path=events_dir,
                glob_pattern="events-*.jsonl",
                on_event=on_event,
                events_count_resolver=lambda _: 0,
            )
        )

        # Register entity to be followed
        reader._followed_entities["test"] = reader.directories[0]

        # Process the last file (should trigger processing all earlier files)
        reader._process_file_change(events_dir / f"events-{num_files - 1}.jsonl")

        # Verify total count
        assert len(received_file_nums) == num_files * events_per_file

        # Verify ordering: file N events should come before file N+1 events
        for i in range(len(received_file_nums) - 1):
            current_file = received_file_nums[i]
            next_file = received_file_nums[i + 1]
            # Either same file or moving to next file
            assert next_file >= current_file, (
                f"Out of order: file {current_file} event followed by file {next_file}"
            )

    def test_sequential_file_rotation_ordering(self, events_dir):
        """Process files sequentially (simulating rotation), verify all events ordered"""
        num_files = 10
        events_per_file = 5

        # Create event files
        for file_num in range(num_files):
            event_file = events_dir / f"events-{file_num}.jsonl"
            with event_file.open("w") as f:
                for i in range(events_per_file):
                    event = JobStateChangedEvent(
                        job_id=f"f{file_num}-{i}", state="running"
                    )
                    f.write(event.to_json() + "\n")

        received_events: list[str] = []

        def on_event(entity_id: str, event: EventBase):
            if isinstance(event, JobStateChangedEvent):
                received_events.append(event.job_id)

        reader = EventReader(
            WatchedDirectory(
                path=events_dir,
                glob_pattern="events-*.jsonl",
                on_event=on_event,
                events_count_resolver=lambda _: 0,
            )
        )

        # Register entity to be followed
        reader._followed_entities["test"] = reader.directories[0]

        # Process files in order (simulates watchdog notifications during rotation)
        for file_num in range(num_files):
            reader._process_file_change(events_dir / f"events-{file_num}.jsonl")

        # Extract file numbers from received events
        def get_file_num(job_id: str) -> int:
            return int(job_id.split("-")[0][1:])  # "f{num}-{i}" -> num

        # Verify events from each file are contiguous and in order
        seen_files = set()
        current_file = -1

        for job_id in received_events:
            file_num = get_file_num(job_id)
            if file_num != current_file:
                assert file_num not in seen_files, (
                    f"Events from file {file_num} appear after we moved to a later file"
                )
                seen_files.add(current_file)
                current_file = file_num

        # Should have seen all files
        assert len(seen_files) == num_files


# =============================================================================
# Tests: EventReader Entity Registration (on_created and follow)
# =============================================================================


class TestEventReaderEntityRegistration:
    """Tests for entity registration via on_created callback and follow method"""

    def test_on_created_called_for_new_entities(self, events_dir):
        """on_created should be called when a new entity is discovered"""
        # Create event file
        entity_dir = events_dir / "entity1"
        entity_dir.mkdir()
        event_file = entity_dir / "events-0.jsonl"
        with event_file.open("w") as f:
            event = JobStateChangedEvent(job_id="job-1", state="running")
            f.write(event.to_json() + "\n")

        created_entities: list[str] = []

        def on_created(entity_id: str, events: list) -> bool:
            created_entities.append(entity_id)
            return True  # Follow this entity

        reader = EventReader(
            WatchedDirectory(
                path=events_dir,
                glob_pattern="*/events-*.jsonl",
                on_created=on_created,
            )
        )

        # Start watching should discover existing entity
        reader.start_watching()

        assert "entity1" in created_entities

        reader.stop_watching()

    def test_on_created_false_ignores_entity(self, events_dir):
        """When on_created returns False, entity events should be ignored"""
        # Create event files for two entities
        for entity_num in [1, 2]:
            entity_dir = events_dir / f"entity{entity_num}"
            entity_dir.mkdir()
            event_file = entity_dir / "events-0.jsonl"
            with event_file.open("w") as f:
                event = JobStateChangedEvent(
                    job_id=f"job-{entity_num}", state="running"
                )
                f.write(event.to_json() + "\n")

        created_events: dict[str, list] = {}

        def on_created(entity_id: str, events: list) -> bool:
            created_events[entity_id] = events
            # Only follow entity1, ignore entity2
            return entity_id == "entity1"

        def on_event(entity_id: str, event: EventBase):
            pass

        reader = EventReader(
            WatchedDirectory(
                path=events_dir,
                glob_pattern="*/events-*.jsonl",
                on_created=on_created,
                on_event=on_event,
            )
        )

        reader.start_watching()

        # Both entities should have on_created called with their events
        assert "entity1" in created_events
        assert len(created_events["entity1"]) == 1
        assert "entity2" in created_events
        # entity2 was rejected, but on_created was still called
        assert len(created_events["entity2"]) == 1

        reader.stop_watching()

    def test_follow_registers_entity(self, events_dir):
        """follow() should register an entity and replay its events"""
        # Create event file
        entity_dir = events_dir / "new_entity"
        entity_dir.mkdir()
        event_file = entity_dir / "events-0.jsonl"
        with event_file.open("w") as f:
            for i in range(3):
                event = JobStateChangedEvent(job_id=f"job-{i}", state="running")
                f.write(event.to_json() + "\n")

        received_events: list[str] = []

        def on_created(entity_id: str, events: list) -> bool:
            # Reject all entities initially
            return False

        def on_event(entity_id: str, event: EventBase):
            if isinstance(event, JobStateChangedEvent):
                received_events.append(event.job_id)

        dir_config = WatchedDirectory(
            path=events_dir,
            glob_pattern="*/events-*.jsonl",
            on_created=on_created,
            on_event=on_event,
        )

        reader = EventReader(dir_config)
        reader.start_watching()

        # No events yet (all entities rejected)
        assert len(received_events) == 0

        # Now explicitly follow the entity — returns events for bulk update
        result = reader.follow("new_entity", dir_config)
        assert isinstance(result, list)

        # Events are returned for bulk consolidation, not replayed via callback
        assert len(result) == 3
        assert len(received_events) == 0

        reader.stop_watching()

    def test_follow_bypasses_on_created(self, events_dir):
        """follow() should bypass on_created and always succeed"""
        created_entities: list[str] = []

        def on_created(entity_id: str, events: list) -> bool:
            created_entities.append(entity_id)
            return False  # Would reject, but follow bypasses this

        def on_event(entity_id: str, event: EventBase):
            pass

        dir_config = WatchedDirectory(
            path=events_dir,
            glob_pattern="*/events-*.jsonl",
            on_created=on_created,
            on_event=on_event,
        )

        reader = EventReader(dir_config)

        # follow() should succeed even though on_created would reject
        result = reader.follow("some_entity", dir_config)
        assert isinstance(result, list)
        # on_created should NOT be called by follow()
        assert "some_entity" not in created_entities

    def test_follow_idempotent(self, events_dir):
        """Calling follow() twice for same entity should be idempotent"""
        # Create event file
        entity_dir = events_dir / "entity1"
        entity_dir.mkdir()
        event_file = entity_dir / "events-0.jsonl"
        with event_file.open("w") as f:
            event = JobStateChangedEvent(job_id="job-1", state="running")
            f.write(event.to_json() + "\n")

        received_events: list[str] = []

        def on_event(entity_id: str, event: EventBase):
            if isinstance(event, JobStateChangedEvent):
                received_events.append(event.job_id)

        dir_config = WatchedDirectory(
            path=events_dir,
            glob_pattern="*/events-*.jsonl",
            on_event=on_event,
        )

        reader = EventReader(dir_config)

        # Follow twice
        result1 = reader.follow("entity1", dir_config)
        result2 = reader.follow("entity1", dir_config)

        # First follow returns events, second is a no-op
        assert len(result1) == 1
        assert len(result2) == 0
        # No events replayed via callback (follow returns them)
        assert len(received_events) == 0


# =============================================================================
# Tests: EventWriter + EventReader Integration
# =============================================================================


class TestEventWriterReaderIntegration:
    """Integration tests for EventWriter and EventReader working together"""

    def test_writer_reader_roundtrip(self, events_dir, status_file):
        """Events written by EventWriter should be readable by EventReader"""

        class SmallRotationWriter(SimpleEventWriter):
            MAX_EVENTS_PER_FILE = 3

        writer = SmallRotationWriter(events_dir, status_path=status_file)

        # Write events
        try:
            for i in range(10):
                event = JobStateChangedEvent(job_id=f"job-{i}", state="running")
                writer.write_event(event)
        finally:
            writer.close()

        # Read events
        received_events: list[str] = []

        def on_event(entity_id: str, event: EventBase):
            if isinstance(event, JobStateChangedEvent):
                received_events.append(event.job_id)

        # Resolver returns 0 to read all events
        reader = EventReader(
            WatchedDirectory(
                path=events_dir,
                glob_pattern="events-*.jsonl",
                on_event=on_event,
                events_count_resolver=lambda _: 0,
            )
        )

        # Register entity to be followed
        reader._followed_entities["test"] = reader.directories[0]

        # Process all files
        for i in range(4):  # 10 events / 3 per file = 4 files
            event_file = events_dir / f"events-{i}.jsonl"
            if event_file.exists():
                reader._process_file_change(event_file)

        assert len(received_events) == 10

    def test_resume_from_events_count(self, events_dir, status_file):
        """EventReader should resume from events_count via resolver"""

        class SmallRotationWriter(SimpleEventWriter):
            MAX_EVENTS_PER_FILE = 3

        writer = SmallRotationWriter(events_dir, status_path=status_file)

        # Write 9 events (3 files: events-0, events-1, events-2)
        try:
            for i in range(9):
                event = JobStateChangedEvent(job_id=f"job-{i}", state="running")
                writer.write_event(event)
        finally:
            writer.close()

        # Verify status has been updated
        with status_file.open() as f:
            status = json.load(f)
        # After 9 events with rotation at 3:
        # - File 0: 3 events, rotation -> events_count=1
        # - File 1: 3 events, rotation -> events_count=2
        # - File 2: 3 events, rotation -> events_count=3
        assert status.get("events_count") == 3

        # Create reader with resolver returning 3 (all files already processed)
        received_events: list[str] = []

        def on_event(entity_id: str, event: EventBase):
            if isinstance(event, JobStateChangedEvent):
                received_events.append(event.job_id)

        reader = EventReader(
            WatchedDirectory(
                path=events_dir,
                glob_pattern="events-*.jsonl",
                on_event=on_event,
                events_count_resolver=lambda _: 3,
            )
        )

        # Register entity to be followed
        reader._followed_entities["test"] = reader.directories[0]

        # Process file 3 (doesn't exist, so nothing happens)
        # This simulates what happens when monitoring resumes after a crash
        reader._process_file_change(events_dir / "events-3.jsonl")

        # Should have no events (file 3 doesn't exist, and files 0-2 are below events_count)
        assert len(received_events) == 0


# =============================================================================
# Tests: EventReader file rotation detection (simulates NFS without watchdog)
# =============================================================================


class TestEventReaderFileRotation:
    """Tests that EventReader detects new event files via polling when watchdog
    doesn't fire on_created (e.g., on NFS/network filesystems)."""

    def test_detects_new_file_after_rotation_via_poll(self, events_dir):
        """When event-9.jsonl is being tailed and event-10.jsonl is created,
        the reader should detect and process the new file via directory polling,
        even without watchdog firing on_created (simulates NFS)."""
        import time

        received_events: list[str] = []

        def on_event(entity_id: str, event: EventBase):
            if isinstance(event, JobStateChangedEvent):
                received_events.append(event.job_id)

        reader = EventReader(
            WatchedDirectory(
                path=events_dir,
                glob_pattern="events-*.jsonl",
                on_event=on_event,
                events_count_resolver=lambda _: 0,
            )
        )

        reader.start_watching()
        try:
            # Register entity
            reader._followed_entities["test"] = reader.directories[0]

            # Create initial event file (events-9.jsonl)
            event_file_9 = events_dir / "events-9.jsonl"
            with event_file_9.open("w") as f:
                for i in range(3):
                    event = JobStateChangedEvent(
                        job_id=f"file9-job-{i}", state="running"
                    )
                    f.write(event.to_json() + "\n")

            # Process file 9 (simulates watchdog detecting it)
            reader._process_file_change(event_file_9)
            assert len(received_events) == 3

            # Disable watchdog handler to simulate NFS (no on_created fires)
            for watch in reader._all_watchers:
                if watch._handler:
                    watch._handler._watch_ref = weakref.ref(type("Dead", (), {})())
                # Enable fast directory polling
                watch._dir_poller.min_interval = 0.1
                watch._dir_poller.poll_interval = 0.1
                watch._dir_poller.schedule_next()

            # Create events-10.jsonl (watchdog won't fire since handler disabled)
            event_file_10 = events_dir / "events-10.jsonl"
            with event_file_10.open("w") as f:
                for i in range(3):
                    event = JobStateChangedEvent(
                        job_id=f"file10-job-{i}", state="running"
                    )
                    f.write(event.to_json() + "\n")

            # Wait for directory polling to detect the new file
            deadline = time.time() + 5.0
            while len(received_events) < 6 and time.time() < deadline:
                time.sleep(0.1)

            assert len(received_events) == 6, (
                f"Expected 6 events (3 from file9, 3 from file10), "
                f"got {len(received_events)}: {received_events}"
            )

            # Verify events from file10 were received
            file10_events = [e for e in received_events if e.startswith("file10-")]
            assert len(file10_events) == 3

        finally:
            reader.stop_watching()

    def test_detects_multiple_rotations_via_poll(self, events_dir):
        """Multiple file rotations should all be detected via polling."""
        import time

        received_events: list[str] = []

        def on_event(entity_id: str, event: EventBase):
            if isinstance(event, JobStateChangedEvent):
                received_events.append(event.job_id)

        reader = EventReader(
            WatchedDirectory(
                path=events_dir,
                glob_pattern="events-*.jsonl",
                on_event=on_event,
                events_count_resolver=lambda _: 0,
            )
        )

        reader.start_watching()
        try:
            reader._followed_entities["test"] = reader.directories[0]

            # Create and process initial file
            event_file_0 = events_dir / "events-0.jsonl"
            with event_file_0.open("w") as f:
                event = JobStateChangedEvent(job_id="file0-job", state="running")
                f.write(event.to_json() + "\n")
            reader._process_file_change(event_file_0)
            assert len(received_events) == 1

            # Disable watchdog handler to simulate NFS
            for watch in reader._all_watchers:
                if watch._handler:
                    watch._handler._watch_ref = weakref.ref(type("Dead", (), {})())
                watch._dir_poller.min_interval = 0.1
                watch._dir_poller.poll_interval = 0.1
                watch._dir_poller.schedule_next()

            # Create files 1, 2, 3 without explicit notifications
            for n in range(1, 4):
                event_file = events_dir / f"events-{n}.jsonl"
                with event_file.open("w") as f:
                    event = JobStateChangedEvent(job_id=f"file{n}-job", state="running")
                    f.write(event.to_json() + "\n")

            # Wait for polling to pick up all new files
            deadline = time.time() + 5.0
            while len(received_events) < 4 and time.time() < deadline:
                time.sleep(0.1)

            assert len(received_events) == 4, (
                f"Expected 4 events, got {len(received_events)}: {received_events}"
            )

        finally:
            reader.stop_watching()


# =============================================================================
# Tests: EventReader with TailedFilePool integration
# =============================================================================


class TestEventReaderTailedPool:
    """Tests for EventReader using TailedFilePool via DirectoryWatch"""

    def test_event_reader_uses_tailed_pool(self, events_dir):
        """Verify EventReader creates watches with tailing enabled"""
        received_events: list[str] = []

        def on_event(entity_id: str, event: EventBase):
            if isinstance(event, JobStateChangedEvent):
                received_events.append(event.job_id)

        reader = EventReader(
            WatchedDirectory(
                path=events_dir, glob_pattern="events-*.jsonl", on_event=on_event
            )
        )

        reader.start_watching()
        try:
            # Verify watches have tailing enabled
            assert reader._file_watcher is not None
            assert reader._file_watcher._tailed_pool is not None

            # Write events and process them via the tailed pool
            event_file = events_dir / "events-0.jsonl"
            with event_file.open("w") as f:
                for i in range(3):
                    event = JobStateChangedEvent(job_id=f"job-{i}", state="running")
                    f.write(event.to_json() + "\n")

            # Register entity and process
            reader._followed_entities["test"] = reader.directories[0]
            reader._process_file_change(event_file)

            assert len(received_events) == 3
        finally:
            reader.stop_watching()

    def test_event_reader_fd_limit_low(self, events_dir):
        """With max_open_files=2, many event files still work correctly"""
        received_events: list[str] = []

        def on_event(entity_id: str, event: EventBase):
            if isinstance(event, JobStateChangedEvent):
                received_events.append(event.job_id)

        reader = EventReader(
            WatchedDirectory(
                path=events_dir,
                glob_pattern="events-*.jsonl",
                on_event=on_event,
                events_count_resolver=lambda _: 0,
            ),
            max_open_files=2,
        )

        # Create many event files
        num_files = 5
        for file_num in range(num_files):
            event_file = events_dir / f"events-{file_num}.jsonl"
            with event_file.open("w") as f:
                for i in range(3):
                    event = JobStateChangedEvent(
                        job_id=f"f{file_num}-job-{i}", state="running"
                    )
                    f.write(event.to_json() + "\n")

        reader.start_watching()
        try:
            reader._followed_entities["test"] = reader.directories[0]

            # Process last file (triggers processing all earlier files)
            reader._process_file_change(events_dir / f"events-{num_files - 1}.jsonl")

            # All events should be received despite low FD limit
            assert len(received_events) == num_files * 3

            # FD count should be at most 2
            assert reader._file_watcher._tailed_pool.open_count <= 2
        finally:
            reader.stop_watching()

    def test_event_reader_incremental_with_tailing(self, events_dir):
        """Incremental reads via tailed pool work correctly"""
        received_events: list[str] = []

        def on_event(entity_id: str, event: EventBase):
            if isinstance(event, JobStateChangedEvent):
                received_events.append(event.job_id)

        reader = EventReader(
            WatchedDirectory(
                path=events_dir, glob_pattern="events-*.jsonl", on_event=on_event
            )
        )

        reader.start_watching()
        try:
            reader._followed_entities["test"] = reader.directories[0]

            # Write first batch
            event_file = events_dir / "events-0.jsonl"
            with event_file.open("w") as f:
                for i in range(3):
                    event = JobStateChangedEvent(job_id=f"batch1-{i}", state="running")
                    f.write(event.to_json() + "\n")

            reader._process_file_change(event_file)
            assert len(received_events) == 3

            # Write second batch (append)
            with event_file.open("a") as f:
                for i in range(3):
                    event = JobStateChangedEvent(job_id=f"batch2-{i}", state="done")
                    f.write(event.to_json() + "\n")

            reader._process_file_change(event_file)
            assert len(received_events) == 6

            # Verify only batch2 events in the second read
            batch2 = [e for e in received_events if e.startswith("batch2-")]
            assert len(batch2) == 3
        finally:
            reader.stop_watching()
