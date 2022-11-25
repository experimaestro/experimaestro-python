"""Test snippets of code"""

import re
import pytest
from pathlib import Path
import tempfile
import subprocess
import sys
import shlex
import os
import logging

DIR = Path(__file__).parents[2]
RE_SNIPPET_START = re.compile(
    r"<!-- SNIPPET: (?P<id>\S+)(?:\s+ARGS\[(?P<args>[^\]]+)\])?(?:\s+ENV\[(?P<env>[^\]]+)\])? -->"
)
RE_SNIPPET_START2 = re.compile(r"```python")
RE_SNIPPET_END = re.compile(r"```")
RE_SNIPPET_VAR = re.compile(r"%([_\w\d]+)%")


class Methods:
    def __iter__(self):
        yield DIR / "README.md", "MAIN"


@pytest.fixture(scope="session")
def snippetpath(tmp_path_factory):
    path = tmp_path_factory.mktemp("snippets")
    return path


def argsub(snippetpath):
    workdir = tempfile.mkdtemp(dir=snippetpath)

    def fn(match):
        varname = match.group(1)
        if varname == "WORKDIR":
            return workdir
        else:
            raise Exception("Unknown variable %s", varname)

    return fn


def get_snippet(snippetpath, path, id):
    with path.open("r") as fp:
        snippet = ""
        args = []
        env = {}
        inside = 0
        for line in fp:
            # logging.info("Line: %s", line.strip())
            m = RE_SNIPPET_START.match(line)
            if m and m.group(1) == id:
                inside = 1
                args = (
                    [
                        RE_SNIPPET_VAR.sub(argsub(snippetpath), arg)
                        for arg in shlex.split(m.group("args"))
                    ]
                    if m.group(2)
                    else []
                )
                env = (
                    {
                        name: val
                        for name, val in (
                            nameval.split("=") for nameval in m.group("env").split(",")
                        )
                    }
                    if m.group("env")
                    else {}
                )
            if inside == 1:
                if RE_SNIPPET_START2.match(line):
                    inside = 2
            elif inside == 2:
                if RE_SNIPPET_END.match(line):
                    inside = False
                else:
                    snippet += line

        return snippet, args, env


@pytest.mark.parametrize("path,id", Methods())
def test_snippet(snippetpath, path, id):
    snippet, args, env = get_snippet(snippetpath, path, id)
    assert snippet != ""

    WAIT_TIME = 10

    fullenv = {name: value for name, value in os.environ.items()}
    fullenv.update(
        {"XPM_ENABLEFAULTHANDLER": "1", "XPM_LOCK_TIMEOUT": str(WAIT_TIME / 2)}
    )
    fullenv.update(env)
    with tempfile.NamedTemporaryFile(
        "wt", suffix=".py", dir=snippetpath, delete=False
    ) as fp:
        fp.write(snippet)

    cmd = [sys.executable, fp.name] + args
    logging.info("Running %s with env=%s", cmd, env)
    p = subprocess.Popen(cmd, env=fullenv)

    try:
        p.wait(WAIT_TIME)
    finally:
        if p.poll() is None:
            logging.error("Killing process PID %s", p.pid)
            p.kill()

    assert p.returncode == 0
