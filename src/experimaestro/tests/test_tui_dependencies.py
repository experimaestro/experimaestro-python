"""Tests for the dependency list of the TUI job detail view"""

import asyncio

from textual.app import App, ComposeResult

from experimaestro.scheduler.interfaces import JobState
from experimaestro.tui.messages import JobSelected
from experimaestro.tui.widgets.jobs import DependencyList


class DependencyApp(App):
    def __init__(self):
        super().__init__()
        self.selected: list[tuple[str, str, str]] = []

    def compose(self) -> ComposeResult:
        yield DependencyList(id="deps")

    def on_job_selected(self, message: JobSelected) -> None:
        self.selected.append((message.job_id, message.task_id, message.experiment_id))


def test_dependency_list_shows_state_and_jumps():
    async def run():
        app = DependencyApp()
        async with app.run_test() as pilot:
            deps = app.query_one(DependencyList)
            deps.set_dependencies(
                "xp-a",
                ["aaaa1111bbbb", "cccc2222dddd", "eeee3333ffff"],
                {
                    "aaaa1111bbbb": ("pkg.Learn", JobState.DONE),
                    "cccc2222dddd": ("pkg.[Index]", JobState.RUNNING),
                },
            )
            await pilot.pause()
            text = str(deps.render())
            assert "done" in text and "Learn" in text
            assert "running" in text and "[Index]" in text
            # Unknown dependency (not in the experiment): no link, unknown state
            assert "unknown" in text and "eeee3333..." in text
            assert set(deps.task_ids) == {"aaaa1111bbbb", "cccc2222dddd"}

            await deps.run_action("select_dependency('aaaa1111bbbb')")
            await deps.run_action("select_dependency('eeee3333ffff')")
            await pilot.pause()
        assert app.selected == [("aaaa1111bbbb", "pkg.Learn", "xp-a")]

    asyncio.run(run())


def test_dependency_list_empty():
    async def run():
        app = DependencyApp()
        async with app.run_test() as pilot:
            deps = app.query_one(DependencyList)
            deps.set_dependencies("xp-a", [], {})
            await pilot.pause()
            assert "(no dependencies)" in str(deps.render())

    asyncio.run(run())
