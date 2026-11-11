"""Exploratory follow-up for faults that escaped the initial scenario matrix."""

from __future__ import annotations

import tempfile
from pathlib import Path

from .controller import build_host, host_link
from .experiment import save_csv, save_json
from .simulation import load_cycles, load_requirements, simulate, test_matrix


def run_coverage(output: Path, seed: int = 42, replicates: int = 5) -> dict:
    """Exercise the generator start after the seeded software fault can take effect."""
    from .report import BLUE, NAVY, TEAL, plt

    requirements = load_requirements()
    cycles = load_cycles()
    cases = [
        case for case in test_matrix(seed, replicates, 0)
        if case.scenario.platform == "REEV" and case.fault in {"healthy", "generator_supervisor"}
    ]
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="propellab-coverage-") as temporary:
        with host_link(build_host(Path(temporary))) as controller:
            for case in cases:
                for late in (False, True):
                    run = simulate(controller, case, cycles[case.scenario.cycle], requirements, late_charge=late)
                    check = next(check for check in run.checks if check.requirement == "DVP-05")
                    rows.append({
                        "case_id": case.id, "cycle": case.scenario.cycle, "condition": "late" if late else "original",
                        "component": case.fault, "status": check.status, "evaluated_ticks": check.evaluated_ticks,
                        "first_failure_second": check.failing_ticks[0] if check.failing_ticks else None,
                    })
    summary = {"seed": seed, "paired_scenarios": len(cases) // 2, "conditions": {}}
    for condition in ("original", "late"):
        group = [row for row in rows if row["condition"] == condition]
        summary["conditions"][condition] = {
            "faults_detected": sum(row["component"] != "healthy" and row["status"] == "fail" for row in group),
            "healthy_passes": sum(row["component"] == "healthy" and row["status"] == "pass" for row in group),
        }
    save_csv(output / "coverage_followup.csv", rows, list(rows[0]))
    save_json(output / "coverage_followup.json", summary)
    count = summary["paired_scenarios"]
    original = summary["conditions"]["original"]
    late = summary["conditions"]["late"]
    fig, axis = plt.subplots(figsize=(8, 3.4), layout="constrained")
    bars = axis.barh(
        ["Original charge profile", "Later low-charge stimulus"],
        [original["faults_detected"], late["faults_detected"]], color=[BLUE, TEAL], height=0.52,
    )
    axis.invert_yaxis()
    axis.set(xlim=(0, count * 1.22), xlabel="Seeded generator faults detected")
    axis.bar_label(bars, labels=[f"{x['faults_detected']}/{count}" for x in (original, late)], padding=5)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="x", alpha=0.16)
    axis.set_axisbelow(True)
    fig.suptitle("Range-extender test coverage", color=NAVY, fontsize=15, fontweight="bold")
    fig.savefig(output / "coverage.png", dpi=170, facecolor="white")
    plt.close(fig)
    (output / "COVERAGE.md").write_text(
        "# Range-extender follow-up\n\n"
        "Some original tests reached low charge before the seeded generator defect became active. "
        "Once the generator was running, those tests could miss a problem with starting it.\n\n"
        "The follow-up held charge above the start threshold for the first 30 seconds, then let it fall. "
        f"On the same **{count}** scenario seeds, the original test caught "
        f"**{original['faults_detected']}/{count}** generator faults. The later stimulus caught "
        f"**{late['faults_detected']}/{count}**. The healthy controller passed "
        f"**{late['healthy_passes']}/{count}** later-stimulus runs.\n\n"
        "![Range-extender test coverage](coverage.png)\n\n"
        "This exploratory retest checks one defect family on the simulator. "
        "The primary diagnosis benchmark retains its original runs.\n\n"
        "[Per-run results](coverage_followup.csv) · [Summary](coverage_followup.json)\n",
        encoding="utf-8",
    )
    return summary
