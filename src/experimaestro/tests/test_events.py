"""Tests for event writing and reading system

This module tests the core event system:
- EventWriter: Writing events to JSONL files with rotation and status tracking
- EventReader: Reading events from JSONL files with ordering guarantees
"""

import json
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

        # Create status.json with events_count=0 (start from file 0)
        status_file = events_dir / "status.json"
        status_file.write_text(json.dumps({"events_count": 0}))

        # Track received events
        received_events: list[str] = []

        def on_event(entity_id: str, event: EventBase):
            if isinstance(event, JobStateChangedEvent):
                received_events.append(event.job_id)

        reader = EventReader(
            WatchedDirectory(
                path=events_dir, glob_pattern="events-*.jsonl", on_event=on_event
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
        """Files below events_count in status.json should be skipped"""
        # Create event files
        for file_num in range(3):
            event_file = events_dir / f"events-{file_num}.jsonl"
            with event_file.open("w") as f:
                for i in range(3):
                    event = JobStateChangedEvent(
                        job_id=f"file{file_num}-job-{i}", state="running"
                    )
                    f.write(event.to_json() + "\n")

        # Create status.json with events_count=1 (skip file 0)
        status_file = events_dir / "status.json"
        status_file.write_text(json.dumps({"events_count": 1}))

        # Track received events
        received_events: list[str] = []

        def on_event(entity_id: str, event: EventBase):
            if isinstance(event, JobStateChangedEvent):
                received_events.append(event.job_id)

        reader = EventReader(
            WatchedDirectory(
                path=events_dir, glob_pattern="events-*.jsonl", on_event=on_event
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

        # Create status.json with events_count=0
        status_file = events_dir / "status.json"
        status_file.write_text(json.dumps({"events_count": 0}))

        # Track received events
        received_events: list[str] = []

        def on_event(entity_id: str, event: EventBase):
            if isinstance(event, JobStateChangedEvent):
                received_events.append(event.job_id)

        reader = EventReader(
            WatchedDirectory(
                path=events_dir, glob_pattern="events-*.jsonl", on_event=on_event
            )
        )

        # Register entity to be followed
        reader._followed_entities["test"] = reader.directories[0]

        # Process file 2 (files 0 and 1 don't exist)
        reader._process_file_change(events_dir / "events-2.jsonl")

        # Should still get events from file 2
        assert len(received_events) == 3


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

        # Create status.json
        status_file = events_dir / "status.json"
        status_file.write_text(json.dumps({"events_count": 0}))

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
                path=events_dir, glob_pattern="events-*.jsonl", on_event=on_event
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

    def test_random_file_access_ordering(self, events_dir):
        """Process files in random order, verify events still ordered"""
        import random

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

        # Create status.json
        status_file = events_dir / "status.json"
        status_file.write_text(json.dumps({"events_count": 0}))

        received_events: list[str] = []

        def on_event(entity_id: str, event: EventBase):
            if isinstance(event, JobStateChangedEvent):
                received_events.append(event.job_id)

        reader = EventReader(
            WatchedDirectory(
                path=events_dir, glob_pattern="events-*.jsonl", on_event=on_event
            )
        )

        # Register entity to be followed
        reader._followed_entities["test"] = reader.directories[0]

        # Process files in random order
        file_order = list(range(num_files))
        random.shuffle(file_order)

        for file_num in file_order:
            reader._process_file_change(events_dir / f"events-{file_num}.jsonl")

        # Extract file numbers from received events
        def get_file_num(job_id: str) -> int:
            return int(job_id.split("-")[0][1:])  # "f{num}-{i}" -> num

        # Verify events from each file are contiguous and in order
        # (all events from file N appear before any event from file N+1)
        seen_files = set()
        current_file = -1

        for job_id in received_events:
            file_num = get_file_num(job_id)
            if file_num != current_file:
                # Switching to new file
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

        def on_created(entity_id: str) -> bool:
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

        received_events: list[str] = []

        def on_created(entity_id: str) -> bool:
            # Only follow entity1, ignore entity2
            return entity_id == "entity1"

        def on_event(entity_id: str, event: EventBase):
            if isinstance(event, JobStateChangedEvent):
                received_events.append(event.job_id)

        reader = EventReader(
            WatchedDirectory(
                path=events_dir,
                glob_pattern="*/events-*.jsonl",
                on_created=on_created,
                on_event=on_event,
            )
        )

        reader.start_watching()

        # Should only have events from entity1
        assert "job-1" in received_events
        assert "job-2" not in received_events

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

        def on_created(entity_id: str) -> bool:
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

        # Now explicitly follow the entity
        result = reader.follow("new_entity", dir_config)
        assert result is True

        # Events should have been replayed
        assert len(received_events) == 3

        reader.stop_watching()

    def test_follow_bypasses_on_created(self, events_dir):
        """follow() should bypass on_created and always succeed"""
        created_entities: list[str] = []

        def on_created(entity_id: str) -> bool:
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
        assert result is True
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
        reader.follow("entity1", dir_config)
        reader.follow("entity1", dir_config)

        # Events should only be replayed once (second follow() is a no-op)
        assert len(received_events) == 1


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

        # Reset events_count to 0 to read all events (simulates fresh reader)
        with status_file.open() as f:
            status = json.load(f)
        status["events_count"] = 0
        with status_file.open("w") as f:
            json.dump(status, f)

        # Read events
        received_events: list[str] = []

        def on_event(entity_id: str, event: EventBase):
            if isinstance(event, JobStateChangedEvent):
                received_events.append(event.job_id)

        reader = EventReader(
            WatchedDirectory(
                path=events_dir, glob_pattern="events-*.jsonl", on_event=on_event
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
        """EventReader should resume from events_count in status.json"""

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

        # Create reader that respects events_count
        received_events: list[str] = []

        def on_event(entity_id: str, event: EventBase):
            if isinstance(event, JobStateChangedEvent):
                received_events.append(event.job_id)

        reader = EventReader(
            WatchedDirectory(
                path=events_dir, glob_pattern="events-*.jsonl", on_event=on_event
            )
        )

        # Register entity to be followed
        reader._followed_entities["test"] = reader.directories[0]

        # Process file 3 (doesn't exist, so nothing happens)
        # This simulates what happens when monitoring resumes after a crash
        reader._process_file_change(events_dir / "events-3.jsonl")

        # Should have no events (file 3 doesn't exist, and files 0-2 are below events_count)
        assert len(received_events) == 0
