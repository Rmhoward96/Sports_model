"""Structural checks of the props-ML GitHub Actions wiring (Props-2 Task 11).

* .github/workflows/train-props-ml.yml -- weekly retrain, one matrix leg per
  served version: features -> quick gate -> final fit -> Release publish
  (v1: props-ml-latest / props-ml-prev; v2: props-ml-v2 / props-ml-v2-prev);
  first-Tuesday fixed-config re-check / dispatch full B ladder in a separate
  job that never publishes.
* generate-sim-nfl.yml and injury-watch.yml (nfl job): SIM_ML_MODE from the
  repo variable (default off) + Release downloads only when the mode is not
  off: props-ml-latest -> models/nfl-sim-ml-v1, props-ml-v2 ->
  models/nfl-sim-ml-v2, each manifest-verified.
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
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / ".github" / "workflows"
SIM_ML_ENV = "${{ vars.SIM_ML_MODE || 'off' }}"
DOWNLOAD_CALLS = ("fetch_verified props-ml-latest nfl-sim-ml-v1",
                  "fetch_verified props-ml-v2 nfl-sim-ml-v2 notice")
VERIFY_WARNING = "::warning::props-ml: release verification failed"
MATRIX = [
    {"version": "nfl-sim-ml-v1", "gate_name": "", "qb_params": "gate", "release": "props-ml-latest",
     "prev_release": "props-ml-prev"},
    {"version": "nfl-sim-ml-v2", "gate_name": "_v2", "qb_params": "serving", "release": "props-ml-v2",
     "prev_release": "props-ml-v2-prev"},
]
FINAL_FIT = 'uv run python scripts/fit_props_ml_final.py --gate-name "$GATE_NAME"'
PUBLISH = 'gh release upload "$RELEASE"'


def leg_env(leg: dict) -> dict:
    """The train job's env for one matrix leg (the job-level `env:` mapping)."""
    return {"MODEL_VERSION": leg["version"], "GATE_NAME": leg["gate_name"], "QB_PARAMS": leg["qb_params"],
            "RELEASE": leg["release"], "PREV_RELEASE": leg["prev_release"],
            "MODELS_DIR": f"data/props_ml/models/{leg['version']}"}
needs_sha256sum = pytest.mark.skipif(shutil.which("sha256sum") is None, reason="sha256sum not installed")


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
    # a full_ladder dispatch must never publish: the train job is skipped
    assert job["if"] == "${{ !inputs.full_ladder }}"
    assert job["timeout-minutes"] == "120"
    assert job.get("permissions", {"contents": "write"}) == {"contents": "write"}
    steps = steps_of(job)
    assert steps[0]["uses"] == "actions/checkout@v5"
    assert steps[1]["id"] == "guard"             # the pipeline.json guard runs first
    assert steps[2]["uses"] == "astral-sh/setup-uv@v10.0.1"
    assert steps[3]["run"].strip() == "uv sync"
    build = step_index(steps, runs("scripts/build_player_features.py"))
    # v1 legs build on the gate QB params, v2 on the serving ones (Ruling S1)
    assert steps[build]["run"].strip() == 'uv run python scripts/build_player_features.py --qb-params "$QB_PARAMS"'
    gate = step_index(steps, runs("scripts/fit_props_ml_final.py --holdout-weeks 2"))
    assert steps[gate]["run"].strip() == ('uv run python scripts/fit_props_ml_final.py '
                                          '--holdout-weeks 2 --gate-name "$GATE_NAME"')
    final = step_index(steps, lambda s: (s.get("run") or "").strip() == FINAL_FIT)
    publish = step_index(steps, runs(PUBLISH))
    assert build < gate < final < publish
    # a failed quick gate (exit 1) must stop the job before the fit / publish:
    # nothing may soften a step failure or force the publish to run anyway.
    for s in steps:
        assert "continue-on-error" not in s, s.get("name")
    for s in steps[gate:]:
        cond = str(s.get("if", ""))
        assert not re.search(r"always\(\)|failure\(\)|cancelled\(\)", cond), (s.get("name"), cond)
    assert "continue-on-error" not in job


def test_train_matrix_one_leg_per_served_version(train_wf):
    job = train_wf["jobs"]["train"]
    assert job["strategy"]["fail-fast"] == "false"          # a v2 failure never blocks v1
    assert job["strategy"]["matrix"]["include"] == MATRIX
    assert job["env"] == {"MODEL_VERSION": "${{ matrix.version }}",
                          "GATE_NAME": "${{ matrix.gate_name }}",
                          "QB_PARAMS": "${{ matrix.qb_params }}",
                          "RELEASE": "${{ matrix.release }}",
                          "PREV_RELEASE": "${{ matrix.prev_release }}",
                          "MODELS_DIR": "data/props_ml/models/${{ matrix.version }}"}


def test_train_publish_step(train_wf):
    steps = steps_of(train_wf["jobs"]["train"])
    pub = steps[step_index(steps, runs(PUBLISH))]
    # only the two retrain guards gate it (no status function: success() is implied)
    assert pub["if"] == "steps.guard.outputs.skip != 'true' && steps.fresh.outputs.skip != 'true'"
    assert pub["env"]["GH_TOKEN"] == "${{ github.token }}"
    script = pub["run"]
    assert "set -euo pipefail" in script
    assert 'gh release view "$RELEASE"' in script
    # latest created (never marked the repo's latest release) when missing
    assert re.search(r'gh release create "\$RELEASE"[^\n]*--latest=false', script)
    # previous assets -> $PREV_RELEASE (delete + recreate) BEFORE the upload
    i_dl = script.index('gh release download "$RELEASE"')
    i_del = script.index('gh release delete "$PREV_RELEASE" -y --cleanup-tag')
    i_prev = script.index('gh release create "$PREV_RELEASE"')
    i_up = script.index(PUBLISH)
    assert i_dl < i_del < i_prev < i_up
    assert re.search(r'gh release create "\$PREV_RELEASE"[^\n]*--latest=false', script)
    assert re.search(r'gh release upload "\$RELEASE"[^\n]*--clobber', script)
    assert 'models="$MODELS_DIR"' in script
    # MANIFEST.sha256 is the LAST upload, after the models and the stale-asset cleanup
    uploads = [m.start() for m in re.finditer(re.escape(PUBLISH), script)]
    manifest_up = [u for u in uploads if "MANIFEST.sha256" in script[u:script.index("\n", u)]]
    assert len(manifest_up) == 1 and manifest_up[0] == max(uploads)
    assert script.index('gh release delete-asset "$RELEASE"') < manifest_up[0]
    # a failure after the upload phase starts restores the saved previous assets
    assert re.search(r"trap restore_latest ERR", script)
    assert script.index("trap restore_latest ERR") < script.index('gh release upload "$RELEASE" "${files[@]}"')
    body = script[script.index("restore_latest() {"):]
    assert re.search(r'gh release upload "\$RELEASE" "\$\{prev\[@\]\}" --clobber', body)
    assert "exit 1" in body
    # no release name is hard-coded: every leg publishes to its own pair
    assert "props-ml-latest" not in script and "props-ml-prev" not in script


def test_train_only_the_publish_step_touches_releases_and_nothing_commits(train_wf):
    text = (WF / "train-props-ml.yml").read_text()
    assert "git push" not in text and "git commit" not in text
    for name, job in train_wf["jobs"].items():
        for s in steps_of(job):
            run = s.get("run") or ""
            if "gh release" not in run:
                continue
            assert name == "train", s.get("name")
            if s.get("id") == "fresh":   # the freshness guard only READS the published config
                assert re.findall(r"gh release \w+", run) == ["gh release download"]
            else:
                assert PUBLISH in run, s.get("name")


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
    assert re.search(r'if \[ -z "\$PROPS_ML_TOGGLES" \]; then\s+echo "::error::[^"]*kept', recheck["run"])
    build = steps[step_index(steps, runs("scripts/build_player_features.py"))]
    collect = steps[step_index(steps, lambda s: s.get("id") == "collect")]
    assert build["id"] == "build"
    assert "steps.build.outcome == 'success'" in collect["if"]
    ladder = steps[step_index(steps, runs("scripts/train_props_ml_b.py"))]
    assert ladder["if"] == "steps.plan.outputs.mode == 'ladder'"
    upload = steps[step_index(steps, lambda s: str(s.get("uses", "")).startswith("actions/upload-artifact@"))]
    assert upload["with"]["if-no-files-found"] == "error"
    assert "steps.collect.outcome == 'success'" in upload["if"]
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
    dl = step_index(steps, runs(DOWNLOAD_CALLS[0]))
    sim = step_index(steps, runs("scripts/generate_sim_nfl.py"))
    assert dl < sim
    step = steps[dl]
    # both versions' Releases, each into its own dir (the last two lines)
    assert [ln.strip() for ln in step["run"].strip().splitlines()[-2:]] == list(DOWNLOAD_CALLS)
    assert 'gh release download "$rel" -D "$dir" --clobber' in step["run"]
    assert 'dir="data/props_ml/models/$2"' in step["run"]
    # verify the downloaded set against the Release's MANIFEST.sha256; any
    # failure removes that version's dir (-> the sim's "no artifacts" ML path)
    assert "sha256sum -c" in step["run"] and "MANIFEST.sha256" in step["run"]
    # --strict: an improperly formatted MANIFEST line is a failure, not a warning
    assert "sha256sum -c --strict" in step["run"]
    assert 'rm -rf "$dir"' in step["run"]
    assert VERIFY_WARNING in step["run"]
    assert "continue-on-error" not in step
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


# ---- executed shell: sim-side Release verification ---------------------------------------

def _download_step_run(name: str, job: str) -> str:
    steps = steps_of(load(name)["jobs"][job])
    return steps[step_index(steps, runs(DOWNLOAD_CALLS[0]))]["run"]


def test_sim_download_steps_are_identical():
    assert _download_step_run("generate-sim-nfl.yml", "generate") == \
        _download_step_run("injury-watch.yml", "nfl")


_FAKE_DL_GH = r'''
import shutil, sys
from pathlib import Path
state = Path(__import__("os").environ["GH_STATE"])
args = sys.argv[1:]
assert args[:2] == ["release", "download"], args
rel = state / args[2]
if not rel.is_dir():
    sys.exit(1)                      # no such release
dest = Path(args[args.index("-D") + 1])
dest.mkdir(parents=True, exist_ok=True)
for f in rel.iterdir():
    shutil.copy(f, dest / f.name)
'''


def _release_state(tmp_path: Path, releases: dict[str, dict[str, str] | None],
                   manifest: dict[str, bool] | None = None) -> Path:
    """Fake Releases on disk: {tag: {file: body}} (+ a MANIFEST.sha256 each
    unless manifest[tag] is False); a None body set = no such release."""
    state = tmp_path / "gh_state"
    state.mkdir()
    for tag, files in releases.items():
        if files is None:
            continue
        d = state / tag
        d.mkdir()
        for n, body in files.items():
            (d / n).write_text(body)
        if (manifest or {}).get(tag, True):
            out = subprocess.run(["sha256sum", *sorted(files)], cwd=d, check=True,
                                 capture_output=True, text=True).stdout
            (d / "MANIFEST.sha256").write_text(out)
    return state


def _run_verify(tmp_path: Path, state: Path, tamper=None) -> subprocess.CompletedProcess:
    """Run the download step against the fake Releases; `tamper(tag_dir)` edits
    a Release's files AFTER its manifest was written (a corrupted download)."""
    script = _download_step_run("generate-sim-nfl.yml", "generate")
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    (bindir / "fake_gh.py").write_text(_FAKE_DL_GH)
    gh = bindir / "gh"
    gh.write_text(f'#!/bin/bash\nexec "{sys.executable}" "{bindir / "fake_gh.py"}" "$@"\n')
    gh.chmod(0o755)
    if tamper is not None:
        tamper(state)
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    env = {"PATH": f"{bindir}:{os.environ['PATH']}", "GH_STATE": str(state)}
    return subprocess.run(["bash", "-e", "-o", "pipefail", "-c", script], cwd=ws, env=env,
                          capture_output=True, text=True)


def _dir(tmp_path: Path, version: str) -> Path:
    return tmp_path / "ws" / "data" / "props_ml" / "models" / version


MODEL_FILES = {"learned.joblib": "A", "b_receptions.joblib": "B", "calibration.json": "{}",
               "props_ml_config.json": '{"x": 1}'}
V2_FILES = {"learned.joblib": "A2", "b_pass_yds.joblib": "B2", "calibration.json": "{2}",
            "props_ml_config.json": '{"model_version": "nfl-sim-ml-v2"}'}


@needs_sha256sum
def test_verify_both_releases_land_in_their_version_dirs(tmp_path):
    state = _release_state(tmp_path, {"props-ml-latest": MODEL_FILES, "props-ml-v2": V2_FILES})
    r = _run_verify(tmp_path, state)
    assert r.returncode == 0, r.stderr
    assert VERIFY_WARNING not in r.stdout and "::warning::" not in r.stdout
    for version, files in (("nfl-sim-ml-v1", MODEL_FILES), ("nfl-sim-ml-v2", V2_FILES)):
        d = _dir(tmp_path, version)
        assert sorted(p.name for p in d.iterdir()) == sorted([*files, "MANIFEST.sha256"])
        assert all((d / n).read_text() == body for n, body in files.items())
    assert not any(p.is_file() for p in (tmp_path / "ws" / "data" / "props_ml" / "models").iterdir())


@needs_sha256sum
def test_verify_missing_v2_release_is_a_notice_and_keeps_v1(tmp_path):
    """M4: a missing props-ml-v2 Release is expected while v1 is served -- a
    ::notice::, not a ::warning:: (if v2 IS served, the sim step's ML path
    fails with its own ::warning::props-ml: artifacts-missing line)."""
    state = _release_state(tmp_path, {"props-ml-latest": MODEL_FILES, "props-ml-v2": None})
    r = _run_verify(tmp_path, state)
    assert r.returncode == 0, r.stderr
    assert "::notice::props-ml: no props-ml-v2 release; nfl-sim-ml-v2 disabled" in r.stdout
    assert "::warning::" not in r.stdout
    assert not _dir(tmp_path, "nfl-sim-ml-v2").exists()
    assert sorted(p.name for p in _dir(tmp_path, "nfl-sim-ml-v1").iterdir()) == sorted(
        [*MODEL_FILES, "MANIFEST.sha256"])


@needs_sha256sum
@pytest.mark.parametrize("bad_tag,bad_version,good_version", [
    ("props-ml-latest", "nfl-sim-ml-v1", "nfl-sim-ml-v2"),
    ("props-ml-v2", "nfl-sim-ml-v2", "nfl-sim-ml-v1"),
])
@pytest.mark.parametrize("case,reason", [
    ("tampered", "checksum"), ("no_manifest", "MANIFEST.sha256 missing"),
    ("extra", "not in MANIFEST.sha256"), ("missing_file", "checksum"),
])
def test_verify_failure_removes_only_that_version(tmp_path, bad_tag, bad_version, good_version,
                                                  case, reason):
    state = _release_state(tmp_path, {"props-ml-latest": MODEL_FILES, "props-ml-v2": V2_FILES},
                           manifest={bad_tag: case != "no_manifest"})

    def tamper(st):
        d = st / bad_tag
        if case == "tampered":
            (d / "calibration.json").write_text('{"tampered": true}')
        elif case == "extra":
            (d / "b_stale.joblib").write_text("old")
        elif case == "missing_file":
            (d / "learned.joblib").unlink()

    r = _run_verify(tmp_path, state, tamper)
    assert r.returncode == 0, r.stderr          # the step never fails the job
    (warn,) = [ln for ln in r.stdout.splitlines() if ln.startswith("::warning::")]
    assert warn.startswith(VERIFY_WARNING) and bad_tag in warn and reason in warn
    assert not _dir(tmp_path, bad_version).exists()
    assert (_dir(tmp_path, good_version) / "props_ml_config.json").is_file()


def test_verify_no_releases_is_a_noop(tmp_path):
    state = _release_state(tmp_path, {"props-ml-latest": None, "props-ml-v2": None})
    r = _run_verify(tmp_path, state)
    assert r.returncode == 0, r.stderr
    # a missing v1 Release stays a warning; the v2 one is a notice (M4)
    assert "::warning::props-ml: no props-ml-latest release; nfl-sim-ml-v1 disabled" in r.stdout
    assert "::notice::props-ml: no props-ml-v2 release; nfl-sim-ml-v2 disabled" in r.stdout
    assert not _dir(tmp_path, "nfl-sim-ml-v1").exists() and not _dir(tmp_path, "nfl-sim-ml-v2").exists()


# ---- executed shell: publish against a stateful fake gh -----------------------------------

_FAKE_GH = r'''
import shutil, sys
from pathlib import Path
state = Path(__import__("os").environ["GH_STATE"])
log = state / "log"
args = sys.argv[1:]
with log.open("a") as fh:
    fh.write(" ".join(Path(a).name if "/" in a else a for a in args) + "\n")
VALUED = {"-D", "--target", "--title", "--notes", "--json", "--jq"}
pos, flags, i = [], {}, 0
while i < len(args):
    if args[i] in VALUED:
        flags[args[i]] = args[i + 1]; i += 2
    else:
        pos.append(args[i]); i += 1
assert pos[0] == "release", args
sub, tag, rest = pos[1], pos[2], pos[3:]
rel = state / tag
if sub == "view":
    if not rel.is_dir():
        sys.exit(1)
    if "--json" in flags:
        print("\n".join(sorted(p.name for p in rel.iterdir())))
elif sub == "download":
    if not rel.is_dir() or not any(rel.iterdir()):
        sys.exit(1)
    shutil.copytree(rel, flags["-D"])
elif sub == "delete":
    if not rel.is_dir():
        sys.exit(1)
    shutil.rmtree(rel)
elif sub == "create":
    rel.mkdir()
    for f in rest:
        if not f.startswith("-"):
            shutil.copy(f, rel / Path(f).name)
elif sub == "upload":
    fail_on = __import__("os").environ.get("FAIL_ON", "")
    for f in rest:
        if f.startswith("-"):
            continue
        if Path(f).name == fail_on and "-prev" not in f:  # restores succeed
            sys.exit(1)
        shutil.copy(f, rel / Path(f).name)
elif sub == "delete-asset":
    (rel / rest[0]).unlink()
else:
    sys.exit(2)
'''

NEW_MODELS = {"learned.joblib": "A2", "b_receptions.joblib": "B2", "calibration.json": "{2}",
              "props_ml_config.json": "{cfg2}", "quick_gate.json": "{pass}"}
OLD_LATEST = {"learned.joblib": "A1", "b_stale.joblib": "S1", "calibration.json": "{1}",
              "props_ml_config.json": "{cfg1}"}


def _publish_env(tmp_path: Path, latest: dict | None, fail_on: str = "",
                 leg: dict = MATRIX[0]) -> tuple[dict, Path, Path]:
    state = tmp_path / "gh_state"
    state.mkdir()
    if latest is not None:
        rel = state / leg["release"]
        rel.mkdir()
        for n, body in latest.items():
            (rel / n).write_text(body)
        man = subprocess.run(["sha256sum", *sorted(latest)], cwd=rel, check=True,
                             capture_output=True, text=True).stdout
        (rel / "MANIFEST.sha256").write_text(man)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "fake_gh.py").write_text(_FAKE_GH)
    gh = bindir / "gh"   # bash wrapper: a shebang can't hold a path with spaces
    gh.write_text(f'#!/bin/bash\nexec "{sys.executable}" "{bindir / "fake_gh.py"}" "$@"\n')
    gh.chmod(0o755)
    ws = tmp_path / "ws"
    models = ws / leg_env(leg)["MODELS_DIR"]
    models.mkdir(parents=True)
    for n, body in NEW_MODELS.items():
        (models / n).write_text(body)
    # the OTHER version's dir must never be published by this leg
    other = ws / "data" / "props_ml" / "models" / "nfl-sim-ml-other"
    other.mkdir(parents=True)
    (other / "props_ml_config.json").write_text("{other}")
    runner_temp = tmp_path / "runner_temp"
    runner_temp.mkdir()
    env = {"PATH": f"{bindir}:{os.environ['PATH']}", "GH_STATE": str(state), "FAIL_ON": fail_on,
           "RUNNER_TEMP": str(runner_temp), "GITHUB_SHA": "abc123", "GITHUB_RUN_ID": "42",
           **leg_env(leg)}
    return env, ws, state


def _run_publish(train_wf, env, ws) -> subprocess.CompletedProcess:
    steps = steps_of(train_wf["jobs"]["train"])
    script = steps[step_index(steps, runs(PUBLISH))]["run"]
    return subprocess.run(["bash", "-e", "-o", "pipefail", "-c", script], cwd=ws, env=env,
                          capture_output=True, text=True)


def _release(state: Path, tag: str) -> dict[str, str]:
    return {p.name: p.read_text() for p in (state / tag).iterdir()}


@needs_sha256sum
@pytest.mark.parametrize("leg", MATRIX, ids=[leg["version"] for leg in MATRIX])
def test_publish_rotates_prev_and_uploads_manifest_last(train_wf, tmp_path, leg):
    env, ws, state = _publish_env(tmp_path, OLD_LATEST, leg=leg)
    rel, prev = leg["release"], leg["prev_release"]
    old = _release(state, rel)
    r = _run_publish(train_wf, env, ws)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _release(state, prev) == old
    latest = _release(state, rel)
    assert set(latest) == {*NEW_MODELS, "MANIFEST.sha256"}      # b_stale.joblib removed
    assert all(latest[n] == body for n, body in NEW_MODELS.items())
    chk = subprocess.run(["sha256sum", "-c", "MANIFEST.sha256"], cwd=state / rel,
                         capture_output=True, text=True)
    assert chk.returncode == 0, chk.stdout
    uploads = [ln for ln in (state / "log").read_text().splitlines()
               if ln.startswith(f"release upload {rel}")]
    assert uploads[-1].split()[3:] == ["MANIFEST.sha256", "--clobber"]
    # the leg touched only its own Release pair
    assert sorted(p.name for p in state.iterdir() if p.is_dir()) == sorted([rel, prev])


@needs_sha256sum
@pytest.mark.parametrize("leg", MATRIX, ids=[leg["version"] for leg in MATRIX])
def test_publish_creates_latest_when_missing(train_wf, tmp_path, leg):
    env, ws, state = _publish_env(tmp_path, None, leg=leg)
    r = _run_publish(train_wf, env, ws)
    assert r.returncode == 0, r.stdout + r.stderr
    assert not (state / leg["prev_release"]).exists()
    assert set(_release(state, leg["release"])) == {*NEW_MODELS, "MANIFEST.sha256"}
    log = (state / "log").read_text()
    assert re.search(rf"release create {leg['release']} .*--latest=false", log)


@needs_sha256sum
@pytest.mark.parametrize("leg", MATRIX, ids=[leg["version"] for leg in MATRIX])
@pytest.mark.parametrize("fail_on", ["calibration.json", "MANIFEST.sha256"])
def test_publish_failure_restores_previous_latest(train_wf, tmp_path, fail_on, leg):
    env, ws, state = _publish_env(tmp_path, OLD_LATEST, fail_on=fail_on, leg=leg)
    old = _release(state, leg["release"])
    r = _run_publish(train_wf, env, ws)
    assert r.returncode != 0
    assert "restoring" in r.stdout
    assert _release(state, leg["release"]) == old


def test_train_rollback_text_serving_table_first():
    text = (WF / "train-props-ml.yml").read_text()
    assert ("Rollback: UPDATE nfl_sim_serving SET model_version = 'sim-nfl-v1' first; then\n"
            "# (optionally) set SIM_ML_MODE=off") in text
    assert "or set the repo variable SIM_ML_MODE=off" not in text



# ---- I2: weekly retrain guards -------------------------------------------------------------

GUARD_NOTICE = "::notice::props-ml: no passing pipeline.json — skipping retrain"
FRESH_NOTICE = "::notice::props-ml: no labelled week after the published trained_through"
GUARD_IF = "steps.guard.outputs.skip != 'true'"
FRESH_IF = "steps.fresh.outputs.skip != 'true'"


def test_train_guards_structure(train_wf):
    steps = steps_of(train_wf["jobs"]["train"])
    guard, fresh = steps[1], steps[step_index(steps, lambda s: s.get("id") == "fresh")]
    assert "if" not in guard and 'f="assets/nfl/props_ml/pipeline${GATE_NAME}.json"' in guard["run"]
    assert GUARD_NOTICE in guard["run"]
    build = step_index(steps, runs("scripts/build_player_features.py"))
    gate = step_index(steps, runs("scripts/fit_props_ml_final.py --holdout-weeks 2"))
    assert build < steps.index(fresh) < gate
    assert fresh["if"] == GUARD_IF
    assert 'gh release download "$RELEASE" -p props_ml_config.json' in fresh["run"]
    assert "--check-new-weeks" in fresh["run"] and FRESH_NOTICE in fresh["run"]
    assert fresh["env"] == {"GH_TOKEN": "${{ github.token }}"}
    # every step after the guard is skipped by it; the fit / publish also by the freshness guard
    for s in steps[2:]:
        assert GUARD_IF in str(s.get("if", "")), s.get("name")
    for s in steps[gate:]:
        assert FRESH_IF in str(s.get("if", "")), s.get("name")


def _run_step(script: str, tmp_path: Path, env_extra: dict | None = None) -> tuple[dict, str]:
    out_file = tmp_path / "github_output"
    out_file.write_text("")
    env = {"PATH": os.environ["PATH"], "GITHUB_OUTPUT": str(out_file),
           "RUNNER_TEMP": str(tmp_path / "runner_temp"), **(env_extra or {})}
    (tmp_path / "runner_temp").mkdir(exist_ok=True)
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    r = subprocess.run(["bash", "-e", "-o", "pipefail", "-c", script], cwd=ws, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return dict(line.split("=", 1) for line in out_file.read_text().splitlines() if line), r.stdout


needs_jq = pytest.mark.skipif(shutil.which("jq") is None, reason="jq not installed")


@needs_jq
@pytest.mark.parametrize("content,skip", [
    (None, "true"),                                        # absent
    ('{"final_pass": false, "markets": {}}', "true"),
    ('{"markets": {}}', "true"),                            # predates final_pass
    ('{"final_pass": "true"}', "true"),                     # only a JSON true counts
    ("not json", "true"),
    ('{"final_pass": true, "markets": {}}', "false"),
])
@pytest.mark.parametrize("leg", MATRIX, ids=[leg["version"] for leg in MATRIX])
def test_train_guard_step_executes(train_wf, tmp_path, content, skip, leg):
    guard = steps_of(train_wf["jobs"]["train"])[1]
    if content is not None:
        f = tmp_path / "ws" / "assets" / "nfl" / "props_ml" / f"pipeline{leg['gate_name']}.json"
        f.parent.mkdir(parents=True)
        f.write_text(content)
        # the OTHER version's pipeline never decides this leg
        other = "pipeline_v2.json" if leg["gate_name"] == "" else "pipeline.json"
        (f.parent / other).write_text('{"final_pass": true, "markets": {}}' if skip == "true"
                                      else '{"final_pass": false}')
    if leg["gate_name"] == "_v2":   # the v2-only guards pass here (tested separately below)
        _v2_guard_files(tmp_path, gate='{"pass": true}', qb='{"gate": {}, "serving": {"H": 2, "k": 50}}',
                        pipeline=None)
    out, stdout = _run_step(guard["run"], tmp_path, leg_env(leg))
    assert out == {"skip": skip}
    assert (GUARD_NOTICE in stdout) is (skip == "true")


V2_GATE_NOTICE = "::notice::props-ml: no passing v2_gate.json (v2 vs v1) — skipping retrain"
SERVING_NOTICE = "::notice::props-ml: no serving block in assets/nfl/props_ml/qb_profile_params.json"


def _v2_guard_files(tmp_path: Path, *, gate: str | None, qb: str | None,
                    pipeline: str | None = '{"final_pass": true}') -> None:
    """The v2 leg's guard inputs (None = absent; `pipeline` None = leave as is)."""
    d = tmp_path / "ws" / "assets" / "nfl" / "props_ml"
    d.mkdir(parents=True, exist_ok=True)
    if pipeline is not None:
        (d / "pipeline_v2.json").write_text(pipeline)
    for name, body in (("v2_gate.json", gate), ("qb_profile_params.json", qb)):
        if body is None:
            (d / name).unlink(missing_ok=True)
        else:
            (d / name).write_text(body)


SERVING_OK = '{"gate": {"H": 1, "k": 100}, "serving": {"H": 2.0, "k": 50.0}}'


@needs_jq
@pytest.mark.parametrize("gate,qb,skip,notice", [
    ('{"pass": true, "verdict": {}}', SERVING_OK, "false", None),
    (None, SERVING_OK, "true", V2_GATE_NOTICE),                           # gate not run / committed
    ('{"pass": false}', SERVING_OK, "true", V2_GATE_NOTICE),
    ('{"pass": "true"}', SERVING_OK, "true", V2_GATE_NOTICE),             # only a JSON true counts
    ("not json", SERVING_OK, "true", V2_GATE_NOTICE),
    ('{"pass": true}', None, "true", SERVING_NOTICE),                     # M3: no params file
    ('{"pass": true}', '{"gate": {"H": 1, "k": 100}}', "true", SERVING_NOTICE),   # no serving block
    ('{"pass": true}', '{"serving": {"H": 2}}', "true", SERVING_NOTICE),  # incomplete block
    ('{"pass": true}', "not json", "true", SERVING_NOTICE),
])
def test_train_guard_v2_leg_needs_the_v2_gate_and_a_serving_block(train_wf, tmp_path, gate, qb, skip,
                                                                    notice):
    """M2: the v2 leg also needs v2_gate.json pass == true (v2 vs v1; pipeline_v2's
    final_pass is only v2 vs the current sim). M3: a missing serving block is a
    ::notice:: skip, not a red build failure."""
    guard = steps_of(train_wf["jobs"]["train"])[1]
    _v2_guard_files(tmp_path, gate=gate, qb=qb)
    out, stdout = _run_step(guard["run"], tmp_path, leg_env(MATRIX[1]))
    assert out == {"skip": skip}
    assert "::warning::" not in stdout and "::error::" not in stdout
    for n in (V2_GATE_NOTICE, SERVING_NOTICE, GUARD_NOTICE):
        assert (n in stdout) is (n == notice), n


@needs_jq
def test_train_guard_v1_leg_ignores_the_v2_only_guards(train_wf, tmp_path):
    guard = steps_of(train_wf["jobs"]["train"])[1]
    d = tmp_path / "ws" / "assets" / "nfl" / "props_ml"
    d.mkdir(parents=True)
    (d / "pipeline.json").write_text('{"final_pass": true}')
    (d / "v2_gate.json").write_text('{"pass": false}')            # a failed v2 gate never blocks v1
    (d / "qb_profile_params.json").write_text('{"gate": {"H": 1, "k": 100}}')   # no serving block
    out, stdout = _run_step(guard["run"], tmp_path, leg_env(MATRIX[0]))
    assert out == {"skip": "false"} and "::notice::" not in stdout


def _fresh_bins(tmp_path: Path, *, release: bool, new_weeks: str) -> dict:
    """Fake `gh` (download -> a props_ml_config.json, or rc 1 when no release) and
    `uv` (prints the --check-new-weeks verdict and records its argv)."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    gh = bindir / "gh"
    gh.write_text("#!/bin/bash\n"
                  f'echo "$@" > "{tmp_path}/gh_argv"\n'
                  f'[ "{int(release)}" = 1 ] || exit 1\n'
                  'while [ $# -gt 0 ]; do [ "$1" = -D ] && d="$2"; shift; done\n'
                  'echo \'{"trained_through": [2026, 3]}\' > "$d/props_ml_config.json"\n')
    uv = bindir / "uv"
    uv.write_text("#!/bin/bash\n"
                  f'echo "$@" > "{tmp_path}/uv_argv"\n'
                  'echo "props-ML: labelled weeks after trained_through ..."\n'
                  f'echo "new_weeks={new_weeks}"\n')
    for f in (gh, uv):
        f.chmod(0o755)
    return {"PATH": f"{bindir}:{os.environ['PATH']}", "GH_TOKEN": "x"}


@pytest.mark.parametrize("release,new_weeks,skip", [
    (True, "false", "true"), (True, "true", "false"), (False, "false", "false"),
])
@pytest.mark.parametrize("leg", MATRIX, ids=[leg["version"] for leg in MATRIX])
def test_train_fresh_step_executes(train_wf, tmp_path, release, new_weeks, skip, leg):
    steps = steps_of(train_wf["jobs"]["train"])
    fresh = steps[step_index(steps, lambda s: s.get("id") == "fresh")]
    out, stdout = _run_step(fresh["run"], tmp_path, {**_fresh_bins(tmp_path, release=release,
                                                                   new_weeks=new_weeks),
                                                     **leg_env(leg)})
    # the published config is read from THIS leg's Release
    assert (tmp_path / "gh_argv").read_text().split()[:3] == ["release", "download", leg["release"]]
    assert out == {"skip": skip}
    assert (FRESH_NOTICE in stdout) is (skip == "true")
    argv = (tmp_path / "uv_argv")
    if release:
        assert "scripts/fit_props_ml_final.py --check-new-weeks" in argv.read_text()
        assert argv.read_text().split()[-1].endswith("props_ml_config.json")
    else:
        assert not argv.exists()     # no release -> proceed without checking
