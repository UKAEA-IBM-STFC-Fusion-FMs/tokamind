"""
gpo_pipeline.py — Portable LSF orchestration for the complete, verified CGPO workflow.

Stages: optional base fine-tuning, official train/validation pair collection, immutable calibration,
base test evaluation, CGPO training/test evaluation, and strict comparison. The small Bash entrypoints
in the external cluster folder select a mode; all scheduler decisions live here and are exercised with simulated LSF.

Cluster paths, Python environment and LSF resources come from exported variables or CGPO_CLUSTER_ENV
loaded by the Bash entrypoint. No user paths, queue names or per-task resource assumptions are embedded.
Dry-run prints every planned stage without invoking subprocesses, submitting jobs or writing files.
Only an explicit DONE (or matching successful bhist record) counts as scheduler success. Missing and
unknown jobs, incomplete artifacts and omitted requested tasks fail the invocation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_TASKS = " ".join(f"task_{g}-{n}" for g, size in ((1, 3), (2, 3), (3, 3), (4, 5)) for n in range(1, size + 1))


# ----------------------------------------------------------------------------------------------------------------------
def job_status(job_id: str) -> str:
    """
    Query a numeric LSF job ID and classify success, active work, failure or unknown state.

    Parameters
    ----------
    job_id : str
        Validated positive LSF identifier.

    Returns
    -------
    str
        DONE for success, an active LSF state, or EXIT for failure.

    Raises
    ------
    RuntimeError
        If status is unknown, output is ambiguous or neither bjobs nor bhist identifies the job.
    """
    if not re.fullmatch(r"[1-9][0-9]*", job_id):
        raise RuntimeError(f"Invalid job ID: {job_id!r}")
    result = subprocess.run(["bjobs", "-a", "-noheader", "-o", "jobid stat", job_id], capture_output=True, text=True)
    rows = [line.split() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode == 0 and len(rows) == 1 and len(rows[0]) == 2 and rows[0][0] == job_id:
        state = rows[0][1]
        if state in {"DONE", "EXIT", "RUN", "PEND", "WAIT", "PROV", "PSUSP", "USUSP", "SSUSP"}:
            return state
        raise RuntimeError(f"UNKNOWN job={job_id} state={state}")
    # Accounting may retain finished jobs after bjobs purges them. Never interpret absence as success.
    try:
        history = subprocess.run(["bhist", "-n", "1", "-l", job_id], capture_output=True, text=True)
    except FileNotFoundError:
        history = None
    if history and history.returncode == 0 and re.search(rf"Job\s*<{job_id}>", history.stdout):
        if "Done successfully" in history.stdout:
            return "DONE"
        if re.search(r"Exited (with|by)|Completed <exit>", history.stdout):
            return "EXIT"
    raise RuntimeError(f"UNKNOWN job={job_id}: bjobs/bhist did not establish a terminal or active state")


# ----------------------------------------------------------------------------------------------------------------------
class Pipeline:
    """
    Coordinate stages while keeping cluster configuration outside the repository.

    Parameters
    ----------
    dry_run : bool
        Print commands and stage transitions only; do not execute or write anything.
    no_plots : bool
        Omit optional calibration figures.
    """

    def __init__(self, dry_run=False, no_plots=True):
        """
        Read and validate exported paths, task names and experiment settings without writes.

        Parameters
        ----------
        dry_run : bool
            Print the plan without subprocesses or writes.
        no_plots : bool
            Disable optional calibration figures.
        """
        self.root = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parent.parent)).resolve()
        self.runs = Path(os.environ.get("RUNS_DIR", self.root / "runs")).resolve()
        self.tag = os.environ.get("GPO_TAG", "cgpo-controlled")
        self.base_tag = os.environ.get("BASE_TAG", "dct3d-embed-mse")
        self.pair_tag = os.environ.get("GPO_PAIR_TAG", self.tag)
        self.eval_tag = os.environ.get("EVAL_TAG", self.tag)
        self.tasks = os.environ.get(
            "TASKS",
            os.environ.get("GPO_TASK", os.environ.get("FINETUNE_TASK", os.environ.get("EVAL_TASK", DEFAULT_TASKS))),
        ).split()
        for legacy in (
            "GPO_MODEL",
            "FINETUNE_MODEL",
            "EVAL_MODEL",
            "GPO_DIR",
            "FINETUNE_INIT",
            "FINETUNE_TAG",
            "RUN_CALIBRATE",
        ):
            if os.environ.get(legacy):
                raise ValueError(
                    f"Legacy variable {legacy} is unsupported; use BASE_RUN/BASE_TAG and the documented launcher modes."
                )
        self.model = os.environ.get("MODEL_PROFILE", "mmt")
        self.embedding = os.environ.get("EMB_PROFILE", "dct3d")
        self.artifacts = Path(os.environ.get("ARTIFACT_DIR", self.root / "cgpo_artifacts" / self.tag)).resolve()
        self.recipe = Path(
            os.environ.get("GPO_TASKS_YAML", self.root / "scripts_mast/configs/mmt/tasks/gpo_tasks.yaml")
        ).resolve()
        self.python = os.environ.get("PYTHON_BIN", sys.executable)
        for name in (
            "DRY_RUN",
            "RUN_FINETUNE",
            "RUN_COLLECT",
            "RUN_GPO",
            "SUBMIT_EVAL",
            "GPO_RESUME",
            "GPO_OVERWRITE",
            "NO_PLOTS",
        ):
            if name in os.environ and os.environ[name] not in {"0", "1"}:
                raise ValueError(f"{name} must be 0 or 1.")
        if os.environ.get("GPO_SPLIT", "both") != "both" or float(os.environ.get("GPO_VAL_FRACTION", "0")) != 0:
            raise ValueError("CGPO requires GPO_SPLIT=both and GPO_VAL_FRACTION=0.")
        self.dry_run = dry_run
        self.no_plots = no_plots
        if not self.tasks or len(self.tasks) != len(set(self.tasks)):
            raise ValueError("TASKS must be nonempty and contain no duplicates.")
        if self.model != "mmt":
            raise ValueError("This CGPO branch supports MODEL_PROFILE=mmt only.")
        for value in [*self.tasks, self.tag, self.base_tag, self.pair_tag, self.eval_tag]:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in {".", ".."}:
                raise ValueError(f"Invalid task/tag: {value!r}")
        if not self.eval_tag:
            raise ValueError("Use a dedicated evaluation tag for the controlled comparison.")
        self.env = {
            **os.environ,
            "MMT_RUNS_DIR": str(self.runs),
            "PYTHONPATH": os.pathsep.join(
                [str(self.root / "src"), str(self.root / "scripts_mast"), os.environ.get("PYTHONPATH", "")]
            ),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

    def base(self, task):
        """
        Return the configured base run path for a task.

        Parameters
        ----------
        task : str
            Requested benchmark task.

        Returns
        -------
        pathlib.Path
            Existing or planned base run directory.
        """
        if os.environ.get("BASE_RUN"):
            if len(self.tasks) != 1:
                raise ValueError("BASE_RUN requires exactly one task.")
            return Path(os.environ["BASE_RUN"]).resolve()
        return self.runs / f"ft-{task}-scratch-{self.model}-{self.base_tag}"

    def gpo(self, task):
        """
        Return the exact warmstart run path generated by the training entrypoint.

        Parameters
        ----------
        task : str
            Requested benchmark task.

        Returns
        -------
        pathlib.Path
            Generated warmstart output directory.
        """
        return self.runs / f"ft-{task}-ws-{self.base(task).name}-{self.model}-{self.tag}"

    def pairs(self, task):
        """
        Return the tagged pair directory under the source run.

        Parameters
        ----------
        task : str
            Requested benchmark task.

        Returns
        -------
        pathlib.Path
            Tagged pair collection directory.
        """
        return self.base(task) / f"gpo_pairs_{self.pair_tag}"

    def calibration(self, task):
        """
        Return the immutable task recipe used by both calibration and training.

        Parameters
        ----------
        task : str
            Requested benchmark task.

        Returns
        -------
        pathlib.Path
            Run-specific frozen YAML recipe.
        """
        return self.artifacts / task / "gpo_tasks.yaml"

    def command(self, script, *args):
        """
        Build a Python entrypoint command with explicit argument boundaries.

        Parameters
        ----------
        script : str
            Entry point filename under scripts_mast.
        *args : Any
            Positional command arguments, converted to strings.

        Returns
        -------
        list[str]
            Python executable, entrypoint and explicit arguments.
        """
        return [self.python, str(self.root / "scripts_mast" / script), *map(str, args)]

    def run(self, command):
        """
        Print a command and execute it only outside dry-run, propagating failures.

        Parameters
        ----------
        command : list[str]
            Command argv to execute with the project environment.

        Returns
        -------
        None
        """
        print("  + " + shlex.join(command), flush=True)
        if not self.dry_run:
            subprocess.run(command, cwd=self.root, env=self.env, check=True)

    def submit(self, stage, task, command):
        """
        Submit one explicit command, rejecting missing or ambiguous scheduler identifiers.

        Parameters
        ----------
        stage, task : str
            Labels for job names and progress output.
        command : list[str]
            Worker argv. Environment and argv are shell-quoted for the LSF command interpreter.

        Returns
        -------
        str | None
            Valid positive job ID, or None for dry-run.
        """
        resources = []
        for key, option in (("LSF_QUEUE", "-q"), ("LSF_NCPUS", "-n"), ("LSF_WALLTIME", "-W")):
            value = os.environ.get(f"{stage.upper()}_{key}", os.environ.get(key))
            if not value and not self.dry_run:
                raise ValueError(f"Set {key} in the external cluster environment before submission.")
            resources += [option, value or f"<{key}>"]
        mem = os.environ.get(f"{stage.upper()}_LSF_MEM_GB", os.environ.get("LSF_MEM_GB"))
        if not mem and not self.dry_run:
            raise ValueError("Set LSF_MEM_GB before submission.")
        resources += ["-R", f"span[hosts=1] rusage[mem={mem or '<LSF_MEM_GB>'}GB]", "-M", f"{mem or '<LSF_MEM_GB>'}GB"]
        gpu = os.environ.get(f"{stage.upper()}_LSF_GPU", os.environ.get("LSF_GPU"))
        if gpu:
            resources += ["-gpu", gpu]
        log_dir = Path(os.environ.get("LOG_DIR", self.artifacts / "logs"))
        name = f"{stage}-{task}-{self.tag}"
        # Force the frozen recipe into the worker environment; no dependence on shell export subtleties.
        worker_env = [
            f"{key}={self.env[key]}"
            for key in (
                "MMT_RUNS_DIR",
                "PYTHONPATH",
                "PYTHONDONTWRITEBYTECODE",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        ]
        if stage == "gpo":
            worker_env += [
                f"GPO_TASKS_YAML={self.calibration(task)}",
                f"GPO_RESUME={os.environ.get('GPO_RESUME', '0')}",
            ]
        command = ["env", *worker_env, *command]
        submit_cmd = [
            "bsub",
            *resources,
            "-cwd",
            str(self.root),
            "-J",
            name,
            "-o",
            str(log_dir / f"{name}-%J.out"),
            "-e",
            str(log_dir / f"{name}-%J.err"),
            shlex.join(command),
        ]
        print("  + " + shlex.join(submit_cmd), flush=True)
        if self.dry_run:
            return None
        log_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(submit_cmd, cwd=self.root, env=self.env, capture_output=True, text=True)
        ids = re.findall(r"Job <([1-9][0-9]*)>", result.stdout)
        if result.returncode or len(ids) != 1:
            raise RuntimeError(f"SUBMISSION FAILED/UNKNOWN {stage} {task}: {result.stdout} {result.stderr}")
        print(f"SUBMITTED stage={stage} task={task} job={ids[0]}", flush=True)
        return ids[0]

    def wait(self, jobs):
        """
        Wait for explicit success of every job; fail on EXIT, unknown status or polling timeout.

        Parameters
        ----------
        jobs : Iterable[str | None]
            Submitted job IDs; dry-run identifiers may be None.

        Returns
        -------
        None
        """
        if self.dry_run:
            return
        pending = {job for job in jobs if job is not None}
        for _ in range(int(os.environ.get("MAX_POLLS", "1440"))):
            for job in sorted(pending):
                state = job_status(job)
                print(f"STATUS job={job} state={state}", flush=True)
                if state == "DONE":
                    pending.remove(job)
                elif state == "EXIT":
                    raise RuntimeError(f"FAILED job={job}: downstream stages were not submitted.")
            if not pending:
                return
            time.sleep(float(os.environ.get("POLL_INTERVAL", "120")))
        raise RuntimeError(f"UNKNOWN/TIMEOUT unfinished jobs={sorted(pending)}")

    def require_base(self, task):
        """
        Check a source run has saved configuration and model blocks before submitting dependent work.

        Parameters
        ----------
        task : str
            Task whose source configuration and model blocks are required.

        Returns
        -------
        None
        """
        if self.dry_run:
            return
        base = self.base(task)
        checkpoint = base / "checkpoints" / "best"
        if not checkpoint.is_dir():
            checkpoint = base / "checkpoints" / "latest"
        required = [
            base / f"{base.name}.yaml",
            *[
                checkpoint / f"{block}.pt"
                for block in ("token_encoder", "backbone", "modality_heads", "output_adapters")
            ],
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"MISSING base artifacts task={task}: {missing}")

    def validate_pairs(self, task):
        """
        Validate complete pair artifacts, including their official split and shard hashes.

        Parameters
        ----------
        task : str
            Task whose collected pair directory must pass validation.

        Returns
        -------
        None
        """
        self.run(self.command("validate_gpo_pairs.py", self.pairs(task)))

    def finetune(self):
        """
        Submit and await base training for every requested task.

        Returns
        -------
        None
        """
        jobs = []
        for task in self.tasks:
            if os.environ.get("BASE_RUN"):
                raise ValueError("BASE_RUN selects an existing run; do not combine it with RUN_FINETUNE=1.")
            jobs.append(
                self.submit(
                    "finetune",
                    task,
                    self.command(
                        "run_finetune.py",
                        "--task",
                        task,
                        "--init",
                        "scratch",
                        "--model_profile",
                        self.model,
                        "--emb_profile",
                        self.embedding,
                        "--tag",
                        self.base_tag,
                    ),
                )
            )
        self.wait(jobs)
        for task in self.tasks:
            self.require_base(task)

    def collect(self):
        """
        Collect official train/validation pairs or explicitly reuse verified existing pairs.

        Returns
        -------
        None
        """
        jobs = []
        for task in self.tasks:
            self.require_base(task)
            if not self.dry_run and self.pairs(task).exists() and os.environ.get("GPO_OVERWRITE", "0") != "1":
                print(f"REUSE collection task={task}: {self.pairs(task)}", flush=True)
                continue
            command = self.command(
                "run_collect_gpo_pairs.py",
                "--task",
                task,
                "--model_source",
                self.base(task),
                "--split",
                "both",
                "--val_fraction",
                "0",
                "--tag",
                self.pair_tag,
                "--train_fraction",
                os.environ.get("GPO_TRAIN_FRACTION", "1"),
                "--shard_size",
                os.environ.get("GPO_SHARD_SIZE", "2048"),
            )
            if os.environ.get("GPO_OVERWRITE", "0") == "1":
                command.append("--overwrite")
            jobs.append(self.submit("collect", task, command))
        self.wait(jobs)
        for task in self.tasks:
            self.validate_pairs(task)

    def calibrate(self):
        """
        Publish or verify each frozen recipe, forwarding the same selected input task configuration.

        Returns
        -------
        None
        """
        for task in self.tasks:
            self.require_base(task)
            self.validate_pairs(task)
            if not self.dry_run and self.calibration(task).exists():
                self.require_calibration(task)
                print(f"REUSE calibration task={task}: {self.calibration(task)}", flush=True)
                continue
            command = self.command(
                "calibrate_gpo_task.py",
                self.pairs(task),
                "--gpo_tasks_yaml",
                self.recipe,
                "--output_dir",
                self.artifacts,
                "--run_tag",
                self.tag,
            )
            if self.no_plots:
                command.append("--no_plots")
            self.run(command)
            self.require_calibration(task)

    def require_calibration(self, task):
        """
        Verify recipe identity, current config inputs, collection identity and the intended run tag.

        Parameters
        ----------
        task : str
            Task whose immutable recipe and summary must match current inputs.

        Returns
        -------
        None
        """
        if self.dry_run:
            return
        import yaml
        from mast_utils.gpo.protocol import config_fingerprint, digest, file_digest

        artifact = yaml.safe_load(self.calibration(task).read_text())
        summary = json.loads((self.calibration(task).parent / "calibration_summary.json").read_text())
        meta = artifact["calibration"]
        if summary.get("calibration") != meta:
            raise ValueError(f"Missing/inconsistent calibration summary for {task}.")
        collection = json.loads((self.pairs(task) / "collection_config.json").read_text())
        if (
            meta.get("format") != "cgpo-calibration-v1"
            or meta["task"] != task
            or meta["run_tag"] != self.tag
            or meta["input_recipe"] != str(self.recipe)
            or meta["input_recipe_digest"] != file_digest(self.recipe)
            or meta["recipe_digest"] != digest(artifact["tasks"][task])
            or meta["inputs"] != config_fingerprint(self.root / "scripts_mast/configs")
            or meta["collection_digest"] != digest(collection)
        ):
            raise ValueError(f"Calibration identity/configuration mismatch for {task}; use a new run tag.")

    def evaluate(self, which):
        """
        Run fresh base or CGPO evaluation on the collection's verified official test split.

        Parameters
        ----------
        which : str
            Model selection: base or gpo.

        Returns
        -------
        None
        """
        if which not in {"base", "gpo"}:
            raise ValueError("EVAL_KIND must be base or gpo.")
        jobs = []
        for task in self.tasks:
            run = self.base(task) if which == "base" else self.gpo(task)
            evaluation = run / f"eval_{self.eval_tag}"
            if not self.dry_run and evaluation.exists() and any(evaluation.iterdir()):
                from mast_utils.gpo.protocol import digest, validate_evaluation_manifest

                proof = validate_evaluation_manifest(run, evaluation, task)
                cc = json.loads((self.pairs(task) / "collection_config.json").read_text())
                if proof["collection_digest"] != digest(cc):
                    raise ValueError(f"Stale evaluation for {task}; select a fresh EVAL_TAG.")
                print(f"REUSE verified {which} evaluation task={task}: {evaluation}", flush=True)
                continue
            command = self.command(
                "run_eval.py",
                "--task",
                task,
                "--model_source",
                run,
                "--tag",
                self.eval_tag,
                "--cgpo_protocol",
                self.pairs(task),
            )
            jobs.append(self.submit(f"{which}_eval", task, command))
        self.wait(jobs)
        if not self.dry_run:
            from mast_utils.gpo.protocol import validate_evaluation_manifest

            for task in self.tasks:
                run = self.base(task) if which == "base" else self.gpo(task)
                validate_evaluation_manifest(run, run / f"eval_{self.eval_tag}", task)

    def train_gpo(self):
        """
        Train every requested task using its immutable calibration recipe; await all jobs.

        Returns
        -------
        None
        """
        jobs = []
        for task in self.tasks:
            self.require_base(task)
            self.require_calibration(task)
            command = self.command(
                "run_gpo_finetune.py",
                "--task",
                task,
                "--model_source",
                self.base(task),
                "--gpo_dir",
                self.pairs(task),
                "--tag",
                self.tag,
                "--model_profile",
                self.model,
                "--emb_profile",
                self.embedding,
            )
            jobs.append(self.submit("gpo", task, command))
        self.wait(jobs)

    def compare(self):
        """
        Compare all requested tasks explicitly; no calibration, plots or recipe writes occur here.

        Returns
        -------
        None
        """
        failures = []
        for task in self.tasks:
            command = self.command(
                "compare_gpo_eval.py",
                "--task",
                task,
                "--base_run",
                self.base(task),
                "--gpo_run",
                self.gpo(task),
                "--runs_dir",
                self.runs,
                "--eval_tag",
                self.eval_tag,
                "--require_verified_test",
            )
            try:
                self.run(command)
            except subprocess.CalledProcessError:
                failures.append(task)
        if failures:
            raise RuntimeError(f"INCOMPLETE comparison: failed/missing tasks={failures}; requested={self.tasks}")

    def complete(self):
        """
        Execute the full staged workflow, with explicit reuse and no implicit task omission.

        Returns
        -------
        None
        """
        print("=== Stage 0: base training ===", flush=True)
        if os.environ.get("RUN_FINETUNE", "0") == "1":
            self.finetune()
        else:
            print("SKIP base training: RUN_FINETUNE=0; verify existing sources", flush=True)
            for task in self.tasks:
                self.require_base(task)
        print("=== Stage 1: pair collection ===", flush=True)
        if os.environ.get("RUN_COLLECT", "1") == "1":
            self.collect()
        else:
            print("SKIP collection: RUN_COLLECT=0; require verified existing pairs", flush=True)
            for task in self.tasks:
                self.validate_pairs(task)
        print("=== Stage 2: immutable calibration ===", flush=True)
        self.calibrate()
        print("=== Stage 3: base evaluation, CGPO training and evaluation ===", flush=True)
        self.evaluate("base")
        if os.environ.get("RUN_GPO", "1") == "1":
            self.train_gpo()
            self.evaluate("gpo")
        else:
            print(
                "SKIP CGPO training/evaluation: RUN_GPO=0; comparison still requires complete verified outputs",
                flush=True,
            )
        print("=== Stage 4: comparison only ===", flush=True)
        self.compare()
        print(
            "DRY RUN COMPLETE (no execution or writes)"
            if self.dry_run
            else "PIPELINE COMPLETE: every requested task compared",
            flush=True,
        )


# ----------------------------------------------------------------------------------------------------------------------
def main():
    """
    Dispatch the selected launcher mode; report any incomplete operation with a nonzero exit code.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--mode", choices=["pipeline", "finetune", "collect", "calibrate", "gpo", "eval"], default="pipeline"
    )
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--compare_only", action="store_true")
    args = parser.parse_args()
    try:
        pipeline = Pipeline(
            args.dry_run or os.environ.get("DRY_RUN", "0") == "1",
            args.no_plots or os.environ.get("NO_PLOTS", "1") == "1",
        )
        if args.compare_only:
            pipeline.compare()
        elif args.mode == "pipeline":
            pipeline.complete()
        elif args.mode == "gpo":
            pipeline.train_gpo()
            if os.environ.get("SUBMIT_EVAL", "1") == "1":
                pipeline.evaluate("gpo")
            else:
                print("SKIP CGPO evaluation: SUBMIT_EVAL=0", flush=True)
        elif args.mode == "eval":
            pipeline.evaluate(os.environ.get("EVAL_KIND", "gpo"))
        else:
            getattr(pipeline, args.mode)()
            if args.compare:
                pipeline.compare()
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"INCOMPLETE: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
