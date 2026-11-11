"""One-command agent entry point."""

from __future__ import annotations

import argparse
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

from .controller import PLATFORMS
from .coverage import run_coverage
from .experiment import run_bench, run_study
from .model import LanguageModel
from .simulation import CYCLES, TICKS


@contextmanager
def local_server(args):
    if not args.gguf:
        yield
        return
    url = urlsplit(args.model_url)
    if url.hostname not in {"127.0.0.1", "localhost"} or not url.port or args.api != "chat":
        raise ValueError("Local GGUF runs require a loopback URL with an explicit port and the chat API")
    args.output.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.llama_server.resolve()), "-m", str(args.gguf.resolve()),
        "--alias", args.model, "--host", "127.0.0.1", "--port", str(url.port),
        "--ctx-size", "8192", "--threads", str(args.threads),
        "--threads-batch", str(args.threads), "--parallel", "1", "--n-gpu-layers", "0",
        "--reasoning-budget", "0",
    ]
    with (args.output / "runtime.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, stdout=log, stderr=log) as process:
            try:
                started = time.monotonic()
                while time.monotonic() - started < 90:
                    if process.poll() is not None:
                        raise RuntimeError(f"Local model exited with {process.returncode}. See {log.name}")
                    try:
                        with urlopen(f"http://127.0.0.1:{url.port}/health", timeout=2) as response:
                            if response.status == 200:
                                break
                    except (URLError, TimeoutError):
                        time.sleep(0.5)
                else:
                    raise RuntimeError(f"Local model did not become ready. See {log.name}")
                yield
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run PropelLab's validation and diagnosis agent")
    commands = parser.add_subparsers(dest="command", required=True)
    study = commands.add_parser("study", help="Generate the plan, run the benchmark, and write the report")
    study.add_argument("--output", type=Path, default=Path("local-results/study"))
    study.add_argument("--seed", type=int, default=42)
    study.add_argument("--replicates", type=int, default=5)
    study.add_argument("--development-replicates", type=int, default=2)
    study.add_argument("--evidence-budget", type=int, default=18)
    study.add_argument("--model-cases", type=int, default=0)
    study.add_argument("--model")
    study.add_argument("--model-url", default="http://127.0.0.1:8157/v1")
    study.add_argument("--model-source")
    study.add_argument("--api", choices=("chat", "responses"), default="chat")
    study.add_argument("--timeout", type=float, default=150)
    study.add_argument("--llama-server", type=Path)
    study.add_argument("--gguf", type=Path)
    study.add_argument("--threads", type=int, default=6)
    bench = commands.add_parser("bench", help="Run and compare the host and serial bench controllers")
    bench.add_argument("--output", type=Path, default=Path("local-results/bench"))
    bench.add_argument("--port", required=True)
    bench.add_argument("--platform", choices=PLATFORMS, default="BEV")
    bench.add_argument("--cycle", choices=CYCLES, default="udds")
    bench.add_argument("--seed", type=int, default=42)
    coverage = commands.add_parser("coverage", help="Retest the timing of the range-extender stimulus")
    coverage.add_argument("--output", type=Path, default=Path("local-results/coverage"))
    coverage.add_argument("--seed", type=int, default=42)
    coverage.add_argument("--replicates", type=int, default=5)
    args = parser.parse_args(argv)
    if args.command == "coverage":
        if args.replicates < 1:
            parser.error("replicates must be positive")
        result = run_coverage(args.output, args.seed, args.replicates)
        print(f"Saved range-extender follow-up for {result['paired_scenarios']} scenarios to {args.output}")
        return
    if args.command == "bench":
        result = run_bench(args.output, args.port, args.platform, args.cycle, args.seed)
        print(f"Saved {result['physical_hil_runs']} hardware runs to {args.output}")
        return
    if not 0 <= args.development_replicates < args.replicates:
        parser.error("development replicates must be nonnegative and smaller than total replicates")
    if not 1 <= args.evidence_budget < TICKS:
        parser.error(f"evidence budget must be between 1 and {TICKS - 1}")
    if args.model_cases < 0 or (args.model_cases and not args.model):
        parser.error("model cases must be nonnegative and require --model when positive")
    if bool(args.gguf) != bool(args.llama_server) or (args.gguf and not args.model):
        parser.error("local inference requires --gguf, --llama-server and --model")
    if args.timeout <= 0 or args.threads < 1:
        parser.error("timeout and thread count must be positive")
    model = LanguageModel(args.model_url, args.model, args.api, args.timeout) if args.model else None
    with local_server(args):
        summary = run_study(
            args.output, seed=args.seed, replicates=args.replicates,
            development=args.development_replicates, budget=args.evidence_budget,
            model_cases=args.model_cases, model=model, model_source=args.model_source,
        )
    print(f"Saved {summary['runs']} runs and {summary['successful_llm_pairs']} model pairs to {args.output}")
