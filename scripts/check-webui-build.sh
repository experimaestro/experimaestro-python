#!/usr/bin/env sh
# Fail the commit when the frontend source (app/) changed but the packaged
# build output (src/experimaestro/webui/data/) was not rebuilt and staged.
#
# The built web UI is committed so the Python package builds without Node. This
# guard keeps the committed assets from drifting away from their source: edit
# app/, rebuild, and stage the result in the same commit.
#
# Bypass intentionally with `git commit --no-verify` when a change under app/
# genuinely does not affect the build output (e.g. tests, README).
set -eu

staged=$(git diff --cached --name-only --diff-filter=ACMR)

app_changed=$(printf '%s\n' "$staged" \
    | grep -E '^app/' \
    | grep -vE '^app/(node_modules|dist)/' || true)
data_changed=$(printf '%s\n' "$staged" \
    | grep -E '^src/experimaestro/webui/data/' || true)

if [ -n "$app_changed" ] && [ -z "$data_changed" ]; then
    echo "error: frontend source changed but the built web UI was not updated." >&2
    echo >&2
    echo "Changed under app/:" >&2
    printf '  %s\n' $app_changed >&2
    echo >&2
    echo "Rebuild and stage the packaged assets:" >&2
    echo "  (cd app && npm run build) && git add src/experimaestro/webui/data" >&2
    echo >&2
    echo "Or bypass if the change does not affect the build: git commit --no-verify" >&2
    exit 1
fi
