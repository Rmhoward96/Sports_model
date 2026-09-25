"""Structural checks of the props-ML GitHub Actions wiring (Props-2 Task 11).

* .github/workflows/train-props-ml.yml -- weekly retrain: features -> quick
  gate -> final fit -> Release publish (props-ml-latest, previous assets kept
  in props-ml-prev); first-Tuesday fixed-config re-check / dispatch full B
  ladder in a separate job that never publishes.
* generate-sim-nfl.yml and injury-watch.yml (nfl job): SIM_ML_MODE from the
  repo variable (default off) + Release download only when the mode is not off.
* desk-auto-nfl.yml never runs generate_sim_nfl.py, so it carries no wiring.

pyyaml is not a project dependency, so the workflows are read with a tiny
block-YAML parser covering the subset the workflows use (block maps and
sequences, ``|`` block scalars, quoted / plain scalars, ``{}`` / ``[]``).
Scalars stay strings. Nothing here touches the network; the re-check plan
step's shell is executed locally against a stubbed ``date``.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / ".github" / "workflows"
SIM_ML_ENV = "${{ vars.SIM_ML_MODE || 'off' }}"
DOWNLOAD_RUN = 'gh release download props-ml-latest -D data/props_ml/models --clobber || echo "no props-ml release"'


# ---- minimal block-YAML reader -----------------------------------------------------------

_KEY = re.compile(r"""^(?P<key>"[^"]*"|'[^']*'|[A-Za-z0-9_.\-/$]+):(?:\s+(?P<rest>.*))?$""")


def _strip_comment(s: str) -> str:
    """Drop a trailing `` #...`` comment outside quotes."""
    quote = None
    for i, ch in enumerate(s):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'" and (i == 0 or s[i - 1] in " [,{:"):
            quote = ch
        elif ch == "#" and (i == 0 or s[i - 1] == " "):
            return s[:i].rstrip()
    return s.rstrip()


def _scalar(raw: str):
    s = _strip_comment(raw.strip())
    if s.startswith('"'):
        m = re.fullmatch(r'"((?:[^"\\]|\\.)*)"', s)
        if not m:
            raise ValueError(f"bad double-quoted scalar: {raw!r}")
        return m.group(1).replace('\\"', '"').replace("\\\\", "\\")
    if s.startswith("'"):
        m = re.fullmatch(r"'((?:[^']|'')*)'", s)
        if not m:
            raise ValueError(f"bad single-quoted scalar: {raw!r}")
        return m.group(1).replace("''", "'")
    if s == "{}":
        return {}
    if s.startswith("["):
        if not s.endswith("]"):
            raise ValueError(f"bad flow sequence: {raw!r}")
        inner = s[1:-1].strip()
        return [] if not inner else [_scalar(x) for x in inner.split(",")]
    if s.startswith("{"):
        raise ValueError(f"flow mappings are not supported: {raw!r}")
    return s


class _Reader:
    def __init__(self, text: str):
        if "\t" in text:
            raise ValueError("tab in workflow YAML")
        self.lines = text.splitlines()
        self.i = 0

    def _next_indent(self) -> int:
        """Indent of the next content line (skipping blanks / comments); -1 at EOF."""
        while self.i < len(self.lines):
            s = self.lines[self.i]
            if s.strip() and not s.strip().startswith("#"):
                return len(s) - len(s.lstrip(" "))
            self.i += 1
        return -1

    def _line(self) -> str:
        return self.lines[self.i].strip()

    def parse(self):
        ind = self._next_indent()
        out = self._block(ind)
        if self._next_indent() != -1:
            raise ValueError(f"unparsed content at line {self.i + 1}: {self.lines[self.i]!r}")
        return out

    def _block(self, ind: int):
        return self._seq(ind) if self._is_dash(self._line()) else self._map(ind)

    @staticmethod
    def _is_dash(line: str) -> bool:
        return line == "-" or line.startswith("- ")

    def _map(self, ind: int) -> dict:
        out: dict = {}
        while True:
            cur = self._next_indent()
            if cur == -1 or cur < ind:
                return out
            line = self._line()
            if cur > ind:
                raise ValueError(f"bad indent at line {self.i + 1}: {self.lines[self.i]!r}")
            if self._is_dash(line):
                return out  # a same-indent sequence belongs to the caller
            m = _KEY.match(line)
            if not m:
                raise ValueError(f"not a mapping entry at line {self.i + 1}: {line!r}")
            key = m.group("key").strip("\"'")
            if key in out:
                raise ValueError(f"duplicate key {key!r} at line {self.i + 1}")
            self.i += 1
            out[key] = self._value(m.group("rest") or "", ind)

    def _value(self, rest: str, ind: int):
        rest = _strip_comment(rest)
        if rest in ("|", "|-", "|+", ">", ">-", ">+"):
            return self._block_scalar(ind, rest)
        if rest:
            return _scalar(rest)
        nxt = self._next_indent()
        if nxt > ind:
            return self._block(nxt)
        if nxt == ind and self._is_dash(self._line()):
            return self._seq(ind)
        return None

    def _block_scalar(self, ind: int, style: str) -> str:
        body: list[str] = []
        block_ind = None
        while self.i < len(self.lines):
            s = self.lines[self.i]
            if not s.strip():
                body.append("")
                self.i += 1
                continue
            cur = len(s) - len(s.lstrip(" "))
            if cur <= ind:
                break
            if block_ind is None:
                block_ind = cur
            if cur < block_ind:
                raise ValueError(f"block scalar dedent at line {self.i + 1}")
            body.append(s[block_ind:])
            self.i += 1
        while body and body[-1] == "":
            body.pop()
        sep = "\n" if style.startswith("|") else " "
        return sep.join(body) + ("" if style.endswith("-") else "\n")

    def _seq(self, ind: int) -> list:
        out: list = []
        while True:
            cur = self._next_indent()
            if cur == -1 or cur < ind:
                return out
            line = self._line()
            if not self._is_dash(line):
                if cur == ind:
                    return out
                raise ValueError(f"bad sequence item at line {self.i + 1}: {line!r}")
            if cur > ind:
                raise ValueError(f"bad indent at line {self.i + 1}: {self.lines[self.i]!r}")
            item = line[1:].strip()
            raw = self.lines[self.i]
            if not item:
                self.i += 1
                nxt = self._next_indent()
                out.append(self._block(nxt) if nxt > cur else None)
            elif _KEY.match(_strip_comment(item)):
                item_ind = cur + 1 + (len(raw[cur + 1:]) - len(raw[cur + 1:].lstrip(" ")))
                self.lines[self.i] = " " * item_ind + item
                out.append(self._map(item_ind))
            else:
                self.i += 1
                out.append(_scalar(item))


def load(name: str) -> dict:
    return _Reader((WF / name).read_text()).parse()


def steps_of(job: dict) -> list[dict]:
    return job["steps"]


def step_index(steps: list[dict], pred) -> int:
    hits = [i for i, s in enumerate(steps) if pred(s)]
    assert len(hits) == 1, f"expected exactly one matching step, got {hits}"
    return hits[0]


def runs(script: str):
    return lambda s: script in (s.get("run") or "")


# ---- the reader itself --------------------------------------------------------------------

def test_reader_parses_the_workflow_subset():
    doc = _Reader(
        'name: x\n'
        'on:\n'
        '  schedule:\n'
        '    - cron: "0 14 * * 2"  # Tue\n'
        '  workflow_dispatch: {}\n'
        'jobs:\n'
        '  a:\n'
        '    steps:\n'
        '    - uses: actions/checkout@v5\n'
        '    - name: multi\n'
        '      if: ${{ a || \'b\' }}\n'
        '      run: |\n'
        '        # a shell comment, not YAML\n'
        '        echo "x: y"\n'
        '\n'
        '        echo done\n'
        '    - [1, 2]\n'
    ).parse()
    assert doc["on"] == {"schedule": [{"cron": "0 14 * * 2"}], "workflow_dispatch": {}}
    steps = doc["jobs"]["a"]["steps"]
    assert steps[0] == {"uses": "actions/checkout@v5"}
    assert steps[1]["if"] == "${{ a || 'b' }}"
    assert steps[1]["run"] == '# a shell comment, not YAML\necho "x: y"\n\necho done\n'
    assert steps[2] == ["1", "2"]


def test_reader_rejects_bad_indent():
    with pytest.raises(ValueError):
        _Reader("a:\n  b: 1\n   c: 2\n").parse()


@pytest.mark.parametrize("path", sorted(WF.glob("*.yml")), ids=lambda p: p.name)
def test_every_workflow_parses(path):
    doc = _Reader(path.read_text()).parse()
    assert "on" in doc and isinstance(doc["jobs"], dict) and doc["jobs"]
    for job in doc["jobs"].values():
        assert isinstance(job["steps"], list) and job["steps"]


# ---- train-props-ml.yml -------------------------------------------------------------------

@pytest.fixture(scope="module")
def train_wf() -> dict:
    return load("train-props-ml.yml")


def test_train_triggers_permissions_concurrency(train_wf):
    assert train_wf["on"]["schedule"] == [{"cron": "0 14 * * 2"}]
    ladder = train_wf["on"]["workflow_dispatch"]["inputs"]["full_ladder"]
    assert ladder["type"] == "boolean" and ladder["default"] == "false"
    assert train_wf["permissions"] == {"contents": "write"}
    assert train_wf["concurrency"]["group"] == "props-ml-train"
    assert train_wf["concurrency"]["cancel-in-progress"] == "false"


def test_train_job_order_gate_before_publish(train_wf):
    job = train_wf["jobs"]["train"]
    assert job["timeout-minutes"] == "120"
    assert job.get("permissions", {"contents": "write"}) == {"contents": "write"}
    steps = steps_of(job)
    assert steps[0]["uses"] == "actions/checkout@v5"
    assert steps[1]["uses"] == "astral-sh/setup-uv@v10.0.1"
    assert steps[2]["run"].strip() == "uv sync"
    build = step_index(steps, runs("scripts/build_player_features.py"))
    gate = step_index(steps, runs("scripts/fit_props_ml_final.py --holdout-weeks 2"))
    final = step_index(steps, lambda s: (s.get("run") or "").strip()
                       == "uv run python scripts/fit_props_ml_final.py")
    publish = step_index(steps, runs("gh release upload props-ml-latest"))
    assert build < gate < final < publish
    # a failed quick gate (exit 1) must stop the job before the fit / publish:
    # nothing may soften a step failure or force the publish to run anyway.
    for s in steps:
        assert "continue-on-error" not in s, s.get("name")
    for s in steps[gate:]:
        cond = str(s.get("if", ""))
        assert not re.search(r"always\(\)|failure\(\)|cancelled\(\)", cond), (s.get("name"), cond)
    assert "continue-on-error" not in job


def test_train_publish_step(train_wf):
    steps = steps_of(train_wf["jobs"]["train"])
    pub = steps[step_index(steps, runs("gh release upload props-ml-latest"))]
    assert "if" not in pub
    assert pub["env"]["GH_TOKEN"] == "${{ github.token }}"
    script = pub["run"]
    assert "set -euo pipefail" in script
    assert "gh release view props-ml-latest" in script
    # latest created (never marked the repo's latest release) when missing
    assert re.search(r"gh release create props-ml-latest[^\n]*--latest=false", script)
    # previous assets -> props-ml-prev (delete + recreate) BEFORE the upload
    i_dl = script.index("gh release download props-ml-latest")
    i_del = script.index("gh release delete props-ml-prev -y --cleanup-tag")
    i_prev = script.index("gh release create props-ml-prev")
    i_up = script.index("gh release upload props-ml-latest")
    assert i_dl < i_del < i_prev < i_up
    assert re.search(r"gh release create props-ml-prev[^\n]*--latest=false", script)
    assert re.search(r"gh release upload props-ml-latest[^\n]*--clobber", script)
    assert "data/props_ml/models" in script


def test_train_only_the_publish_step_touches_releases_and_nothing_commits(train_wf):
    text = (WF / "train-props-ml.yml").read_text()
    assert "git push" not in text and "git commit" not in text
    for name, job in train_wf["jobs"].items():
        for s in steps_of(job):
            if "gh release" in (s.get("run") or ""):
                assert name == "train" and "gh release upload props-ml-latest" in s["run"], s.get("name")


def test_recheck_job(train_wf):
    job = train_wf["jobs"]["recheck"]
    assert job["permissions"] == {"contents": "read"}
    steps = steps_of(job)
    plan = steps[0]
    assert plan["id"] == "plan"
    assert "date -u +%d" in plan["run"]
    assert plan["env"]["EVENT"] == "${{ github.event_name }}"
    assert plan["env"]["FULL_LADDER"] == "${{ inputs.full_ladder }}"
    for s in steps[1:]:
        assert "steps.plan.outputs.mode" in s["if"], s.get("name")
    recheck = steps[step_index(steps, runs("scripts/train_props_ml.py"))]
    assert recheck["if"] == "steps.plan.outputs.mode == 'recheck'"
    assert recheck["env"]["PROPS_ML_SEASONS"] == "${{ steps.plan.outputs.seasons }}"
    assert "PROPS_ML_TOGGLES" in recheck["run"]
    assert ".kept" in recheck["run"] and "assets/nfl/props_ml/a_gate.json" in recheck["run"]
    ladder = steps[step_index(steps, runs("scripts/train_props_ml_b.py"))]
    assert ladder["if"] == "steps.plan.outputs.mode == 'ladder'"
    upload = steps[step_index(steps, lambda s: str(s.get("uses", "")).startswith("actions/upload-artifact@"))]
    assert upload["with"]["if-no-files-found"] == "error"
    assert steps.index(upload) > max(steps.index(recheck), steps.index(ladder))


def _run_plan(train_wf, tmp_path, fake_date: str, event: str, full_ladder: str = "") -> dict:
    """Execute the plan step's shell with a stubbed `date` printing fake_date."""
    plan = train_wf["jobs"]["recheck"]["steps"][0]
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    stub = bindir / "date"
    stub.write_text(
        "#!/bin/bash\n"
        'for a in "$@"; do case "$a" in +*) fmt="${a#+}";; esac; done\n'
        'y=${FAKE_DATE:0:4}; m=${FAKE_DATE:5:2}; d=${FAKE_DATE:8:2}\n'
        'out=${fmt//%Y/$y}; out=${out//%m/$m}; out=${out//%d/$d}; echo "$out"\n')
    stub.chmod(0o755)
    out_file = tmp_path / "github_output"
    out_file.write_text("")
    env = {"PATH": f"{bindir}:{os.environ['PATH']}", "FAKE_DATE": fake_date,
           "EVENT": event, "FULL_LADDER": full_ladder, "GITHUB_OUTPUT": str(out_file)}
    subprocess.run(["bash", "-e", "-o", "pipefail", "-c", plan["run"]], env=env, check=True,
                   capture_output=True, text=True)
    return dict(line.split("=", 1) for line in out_file.read_text().splitlines() if line)


@pytest.mark.parametrize("fake_date,event,ladder,mode,seasons", [
    ("2026-10-06", "schedule", "", "recheck", "2024,2025,2026"),        # 1st Tue, in season
    ("2026-10-13", "schedule", "", "none", "2024,2025,2026"),           # 2nd Tue
    ("2026-09-01", "schedule", "", "recheck", "2023,2024,2025"),        # pre-kickoff: 2026 not started
    ("2027-01-05", "schedule", "", "recheck", "2024,2025,2026"),        # Jan belongs to the 2026 season
    ("2027-05-04", "schedule", "", "recheck", "2024,2025,2026"),        # offseason
    ("2026-10-06", "workflow_dispatch", "false", "none", "2024,2025,2026"),
    ("2026-10-06", "workflow_dispatch", "true", "ladder", "2024,2025,2026"),
    ("2026-10-20", "workflow_dispatch", "true", "ladder", "2024,2025,2026"),
])
def test_recheck_plan_step(train_wf, tmp_path, fake_date, event, ladder, mode, seasons):
    out = _run_plan(train_wf, tmp_path, fake_date, event, ladder)
    assert out["mode"] == mode
    assert out["seasons"] == seasons


# ---- SIM_ML_MODE wiring in the sim workflows ----------------------------------------------

def _check_sim_wiring(job: dict, extra_if: str | None = None) -> None:
    assert job["env"]["SIM_ML_MODE"] == SIM_ML_ENV
    steps = steps_of(job)
    dl = step_index(steps, runs("gh release download props-ml-latest"))
    sim = step_index(steps, runs("scripts/generate_sim_nfl.py"))
    assert dl < sim
    step = steps[dl]
    assert step["run"].strip() == DOWNLOAD_RUN
    assert step["env"] == {"GH_TOKEN": "${{ github.token }}"}
    assert "env.SIM_ML_MODE != 'off'" in step["if"]
    if extra_if:
        assert extra_if in step["if"]


def test_generate_sim_nfl_wiring():
    wf = load("generate-sim-nfl.yml")
    assert wf["permissions"] == {"contents": "read"}
    job = wf["jobs"]["generate"]
    assert job["timeout-minutes"] == "35"
    _check_sim_wiring(job)


def test_injury_watch_nfl_wiring():
    wf = load("injury-watch.yml")
    assert wf["permissions"] == {"contents": "read"}
    nfl, cfb = wf["jobs"]["nfl"], wf["jobs"]["cfb"]
    assert nfl["timeout-minutes"] == "60"
    _check_sim_wiring(nfl, extra_if="steps.check.outputs.changed == 'true'")
    # the cfb job has no sim step and is untouched
    assert cfb["timeout-minutes"] == "40"
    assert "SIM_ML_MODE" not in cfb["env"]
    assert not any("props-ml" in (s.get("run") or "") for s in steps_of(cfb))


def test_desk_auto_nfl_untouched():
    """desk-auto-nfl.yml never runs the sim, so it gets no props-ML wiring."""
    text = (WF / "desk-auto-nfl.yml").read_text()
    assert "generate_sim_nfl.py" not in text
    assert "SIM_ML_MODE" not in text and "props-ml" not in text
    job = load("desk-auto-nfl.yml")["jobs"]["desk"]
    assert job["timeout-minutes"] == "20"
