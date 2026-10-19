# Contributing to experimaestro

## Setting up

```bash
git clone https://github.com/experimaestro/experimaestro-python.git
cd experimaestro-python
uv sync --all-extras
```

## Pre-commit hooks (required)

All contributions must pass the [pre-commit](https://pre-commit.com) hooks
defined in `.pre-commit-config.yaml`. Install them once after cloning:

```bash
uv tool install pre-commit   # or: pipx install pre-commit
pre-commit install           # installs both the pre-commit and commit-msg hooks
```

The hooks then run on every `git commit`:

- **ruff** (lint with `--fix`) and **ruff-format**
- YAML check, end-of-file and trailing-whitespace fixers
- no blanket `# noqa`, no `eval()`, no `FIXME:`/`TODO:` markers
- **web UI build sync**: if you change `app/`, rebuild with
  `cd app && npm run build` and stage `src/experimaestro/webui/data/` in the
  same commit
- **Conventional Commits** for the commit message (see below)

When a hook modifies files (e.g. formatting), the commit is aborted: stage the
changes and commit again. A convenient pattern is:

```bash
git add <files> && pre-commit; git add -u && git commit -m "fix: ..."
```

Before opening a pull request, run the hooks on every file you touched:

```bash
pre-commit run --all-files
```

Do not bypass the hooks with `--no-verify`, except for the documented web UI
case where a change under `app/` does not affect the build output.

## Commit messages

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org)
— they are used to generate the changelog with git-cliff:

```
feat(scheduler): add per-task retry limit
fix: release the lock when a job is cancelled
docs: document workspace triggers
```

Common types: `feat`, `fix`, `docs`, `perf`, `refactor`, `test`, `chore`,
`ci`, `build`.

## Tests

```bash
uv run pytest                                   # all tests
uv run pytest src/experimaestro/tests/test_identifier.py   # one file
```

Add tests for bug fixes and new features. Test data must not contain personal
information (real names, emails, …): use made-up values.

## Pull requests

- Keep each pull request focused on one change.
- Make sure `pre-commit run --all-files` and `uv run pytest` pass locally.
- Update the documentation (`docs/source/`) when you change the CLI or a
  public API.
- If you change the frontend (`app/`), commit the rebuilt assets in
  `src/experimaestro/webui/data/`.
