"""Autonomous execution, artifact persistence, and matched evaluation."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import random
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .controller import PLATFORMS, build_host, host_link, serial_link
from .evidence import paired_deltas, payload, rules_ranking
from .model import DIAGNOSIS_INSTRUCTIONS, DIAGNOSIS_SCHEMA, LanguageModel, diagnose, draft_plan
from .simulation import Case, Run, load_cycles, load_requirements, simulate, test_matrix

ARMS = ("rules", "random_forest", "llm_full", "llm_selected")


def save_json(path: Path, data) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as file:
        temporary = Path(file.name)
        json.dump(data, file, indent=2, allow_nan=False)
        file.write("\n")
    temporary.replace(path)


def save_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def features(run: Run, reference: Run) -> np.ndarray:
    data = np.asarray(paired_deltas(run, reference))
    absolute = np.abs(data)
    onsets = np.where(absolute > 0, np.arange(len(data))[:, None], len(data)).min(axis=0)
    return np.concatenate([data.mean(axis=0), absolute.max(axis=0),
                           data.std(axis=0), onsets / len(data)])


def sample_failures(runs: list[Run], count: int, seed: int) -> list[Run]:
    """Round-robin by powertrain, favoring fault families not yet represented."""
    randomizer = random.Random(seed)
    pools = {platform: [r for r in runs if r.case.scenario.platform == platform]
             for platform in PLATFORMS}
    for pool in pools.values():
        randomizer.shuffle(pool)
    selected = []
    frequency = Counter()
    while len(selected) < min(count, len(runs)):
        for pool in pools.values():
            if not pool or len(selected) >= count:
                continue
            index = min(range(len(pool)), key=lambda i: frequency[pool[i].case.fault])
            run = pool.pop(index)
            frequency[run.case.fault] += 1
            selected.append(run)
    return selected


def prediction(run: Run, method: str, ranking: list[str], **metadata) -> dict:
    gold = run.case.fault
    return {
        "case_id": run.case.id, "platform": run.case.scenario.platform,
        "cycle": run.case.scenario.cycle, "gold": gold, "method": method,
        "ranking": ranking, "rank": ranking.index(gold) + 1 if gold in ranking else None,
        "status": "ok", **metadata,
    }


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    hit1 = sum(row["rank"] == 1 for row in rows)
    hit3 = sum(row["rank"] is not None and row["rank"] <= 3 for row in rows)
    return {
        "cases": n, "valid": sum(row["status"] == "ok" for row in rows),
        "top1_hits": hit1, "top3_hits": hit3,
        "top1": hit1 / n if n else None, "top3": hit3 / n if n else None,
        "mrr": sum(1 / row["rank"] for row in rows if row["rank"]) / n if n else None,
        "input_tokens": sum(row.get("input_tokens", 0) for row in rows),
        "output_tokens": sum(row.get("output_tokens", 0) for row in rows),
        "median_seconds": statistics.median(row["seconds"] for row in rows if "seconds" in row)
        if any("seconds" in row for row in rows) else None,
    }


def _verify_plan(output: Path, requirements: list[dict], model: LanguageModel | None,
                 reuse_plan: bool) -> dict:
    plan_file = output / "verification_plan.json"
    if reuse_plan and model is not None and plan_file.is_file():
        previous = json.loads(plan_file.read_text(encoding="utf-8"))
        requested = {req["id"]: req["text"] for req in requirements}
        existing = {test["requirement"]: test["pass_criterion"] for test in previous["tests"]}
        if (previous.get("source") == "language_model" and previous.get("model") == model.model
                and existing == requested and len(previous["tests"]) == len(requirements)):
            return previous
    proposals = []
    metadata = {"source": "requirement_catalog"}
    if model is not None:
        try:
            proposals, usage = draft_plan(model, requirements)
            metadata = {"source": "language_model", "model": model.model, **usage}
        except (ValueError, RuntimeError, KeyError, TypeError) as exc:
            metadata["model_error"] = str(exc)
    suggestions = {test["requirement"]: test for test in proposals}
    rows = [
        {
            "test_id": f"TEST-{req['id'][4:]}", "requirement": req["id"],
            "description": req["text"], "stimulus": req["stimulus"],
            "severity": req["severity"], "platforms": req["platforms"],
            "pass_criterion": req["text"], "preferred_cycle": suggestions.get(req["id"], {}).get("cycle", "udds"),
            "rationale": suggestions.get(req["id"], {}).get("rationale", "Use the original requirement as the test oracle."),
            "status": "schema_checked_for_simulation",
        }
        for req in sorted(requirements, key=lambda item: (item["severity"] != "high", item["id"]))
    ]
    plan = {**metadata, "tests": rows}
    save_json(plan_file, plan)
    return plan


def _llm_record(model: LanguageModel, input_data: dict, cache: Path) -> dict:
    material = [model.url, model.model, model.api, DIAGNOSIS_INSTRUCTIONS, DIAGNOSIS_SCHEMA, input_data]
    digest = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
    file = cache / f"{digest}.json"
    if file.exists():
        return json.loads(file.read_text(encoding="utf-8"))
    response = diagnose(model, input_data)
    save_json(file, response)
    return response


def _model_predictions(model, study_sample, references, requirements, budget, cache):
    cache.mkdir(parents=True, exist_ok=True)
    output = []
    for index, run in enumerate(study_sample, start=1):
        reference = references[run.case.scenario.id]
        for method, row_budget in (("llm_full", None), ("llm_selected", budget)):
            evidence = payload(run, reference, requirements, row_budget)
            try:
                result = _llm_record(model, evidence, cache)
                ranking = result.pop("ranking")
                row = prediction(run, method, ranking, **result)
            except (RuntimeError, ValueError, KeyError, TypeError) as exc:
                row = prediction(run, method, [], status="invalid", error=str(exc))
            row["rows_sent"] = len(evidence["paired_trace"])
            output.append(row)
            print(f"Model case {index}/{len(study_sample)} {run.case.id} {method}: {row['status']}", flush=True)
    return output


def _save_observations(output: Path, runs: list[Run]) -> None:
    with (output / "telemetry.jsonl.gz").open("wb") as file:
        with gzip.GzipFile(filename="", mode="wb", fileobj=file, mtime=0) as compressed:
            for run in runs:
                compressed.write((json.dumps(run.record(), separators=(",", ":")) + "\n").encode())
    save_csv(output / "runs.csv", [
        {"case_id": r.case.id, "platform": r.case.scenario.platform, "cycle": r.case.scenario.cycle,
         "replicate": r.case.scenario.replicate, "split": r.case.scenario.split,
         "injected_component": r.case.fault, "failed": r.failed,
         "failed_requirements": " ".join(check.requirement for check in r.checks if check.failing_ticks)}
        for r in runs
    ], ["case_id", "platform", "cycle", "replicate", "split", "injected_component", "failed", "failed_requirements"])


def run_study(output: Path, *, seed: int, replicates: int, development: int,
              budget: int, model_cases: int, model: LanguageModel | None,
              model_source: str | None) -> dict:
    from .report import write_report

    output.mkdir(parents=True, exist_ok=True)
    requirements = load_requirements()
    cycles = load_cycles()
    config = {"seed": seed, "replicates": replicates, "development_replicates": development,
              "evidence_budget": budget, "requested_llm_cases": model_cases,
              "llm_model": model.model if model else None, "model_source": model_source}
    config_file = output / "config.json"
    reuse_plan = config_file.is_file() and json.loads(config_file.read_text(encoding="utf-8")) == config
    save_json(config_file, config)
    plan = _verify_plan(output, requirements, model, reuse_plan)
    cases = test_matrix(seed, replicates, development)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="propellab-host-") as temporary:
        with host_link(build_host(Path(temporary))) as controller:
            runs = [simulate(controller, case, cycles[case.scenario.cycle], requirements) for case in cases]
    simulation_seconds = time.monotonic() - started
    _save_observations(output, runs)
    references = {run.case.scenario.id: run for run in runs if run.case.fault == "healthy"}
    failing = [run for run in runs if run.failed and run.case.fault != "healthy"]
    dev = [run for run in failing if run.case.scenario.split == "development"]
    test = [run for run in failing if run.case.scenario.split == "held_out"]
    print(f"Executed {len(runs)} SIL runs, {len(test)} held-out failures eligible for ranking.", flush=True)
    predictions = [prediction(run, "rules", rules_ranking(run, requirements)) for run in test]
    if dev and test:
        classifier = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
        classifier.fit([features(run, references[run.case.scenario.id]) for run in dev],
                       [run.case.fault for run in dev])
        probabilities = classifier.predict_proba([features(run, references[run.case.scenario.id]) for run in test])
        for run, probabilities_for_case in zip(test, probabilities, strict=True):
            ranked = [str(classifier.classes_[i]) for i in np.argsort(-probabilities_for_case, kind="stable")
                      if classifier.classes_[i] != "generator_supervisor" or run.case.scenario.platform == "REEV"]
            predictions.append(prediction(run, "random_forest", ranked[:3]))
    sample = sample_failures(test, model_cases, seed)
    if model and sample:
        predictions.extend(_model_predictions(model, sample, references, requirements, budget, output / ".model-cache"))
    by_arm = {arm: [row for row in predictions if row["method"] == arm] for arm in ARMS}
    sample_ids = {run.case.id for run in sample}
    matched = {arm: metrics([row for row in rows if row["case_id"] in sample_ids])
               for arm, rows in by_arm.items() if sample and rows}
    matched_ids = (
        {row["case_id"] for row in by_arm["llm_full"] if row["status"] == "ok"}
        & {row["case_id"] for row in by_arm["llm_selected"] if row["status"] == "ok"}
    )
    paired_full = metrics([row for row in by_arm["llm_full"] if row["case_id"] in matched_ids])
    paired_selected = metrics([row for row in by_arm["llm_selected"] if row["case_id"] in matched_ids])
    summary = {
        **config, "mode": "sil", "physical_hil_runs": 0, "simulation_seconds": round(simulation_seconds, 3),
        "runs": len(runs), "healthy_runs": len(references),
        "healthy_passes": sum(not run.failed for run in references.values()),
        "seeded_faults": len(runs) - len(references), "detected_faults": len(failing),
        "missed_by_component": dict(Counter(run.case.fault for run in runs
                                            if not run.failed and run.case.fault != "healthy")),
        "development_failures": len(dev), "held_out_failures": len(test),
        "requirements_exercised": {
            req["id"]: sum(any(check.requirement == req["id"] and check.evaluated_ticks for check in run.checks)
                           for run in runs)
            for req in requirements
        },
        "llm_sample": [{"id": run.case.id, "platform": run.case.scenario.platform,
                        "component": run.case.fault} for run in sample],
        "successful_llm_pairs": len(matched_ids),
        "methods": {arm: metrics(rows) for arm, rows in by_arm.items() if rows},
        "matched_subset": matched,
        "paired_input_tokens": {"full": paired_full["input_tokens"], "selected": paired_selected["input_tokens"]},
        "plan_source": plan["source"], "plan_error": plan.get("model_error"),
    }
    if paired_full["input_tokens"]:
        summary["paired_token_reduction"] = 1 - paired_selected["input_tokens"] / paired_full["input_tokens"]
    if summary["missed_by_component"].get("generator_supervisor"):
        from .coverage import run_coverage

        summary["coverage_followup"] = run_coverage(output, seed, replicates)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as file:
        for row in predictions:
            file.write(json.dumps(row, separators=(",", ":")) + "\n")
    save_json(output / "summary.json", summary)
    write_report(output, summary, runs, references, predictions, budget)
    return summary


def run_bench(output: Path, port: str, platform: str, cycle: str, seed: int) -> dict:
    requirements = load_requirements()
    cycles = load_cycles()
    case = next(case for case in test_matrix(seed, 1, 0)
                if case.scenario.platform == platform and case.scenario.cycle == cycle
                and case.fault == "healthy")
    cases = [case, Case(case.scenario, "temperature_monitor")]
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="propellab-host-") as temporary:
        with host_link(build_host(Path(temporary))) as link:
            software = [simulate(link, trial, cycles[cycle], requirements) for trial in cases]
    with serial_link(port) as link:
        hardware = [simulate(link, trial, cycles[cycle], requirements) for trial in cases]
    comparisons = [
        {"case_id": sil.case.id, "component": sil.case.fault,
         "verdict_agreement": [(c.requirement, c.status) for c in sil.checks]
         == [(c.requirement, c.status) for c in hil.checks],
         "deadline_misses": hil.deadline_misses,
         "sil": sil.record(), "hil": hil.record()}
        for sil, hil in zip(software, hardware, strict=True)
    ]
    result = {"physical_hil_runs": len(hardware), "comparisons": comparisons}
    save_json(output / "hil_comparison.json", result)
    return result
