"""Generate the report, plots, and a filterable local evidence dashboard."""

from __future__ import annotations

import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator

from .evidence import CHANNELS, paired_deltas, selected_ticks

LABELS = {
    "rules": "Rules", "random_forest": "Random forest",
    "llm_full": "LLM full telemetry", "llm_selected": "LLM selected telemetry",
}
NAVY = "#112640"
BLUE = "#346cb0"
TEAL = "#0f836f"
SIGNAL_LABELS = {
    "req": "Smoothed request", "energy": "Energy estimate", "temp_flag": "Temperature flag",
    "cap": "Torque limit", "cmd": "Torque command", "actual": "Measured torque",
    "diag": "Diagnostic code", "gen": "Generator status",
}


def _comparison(output: Path, summary: dict) -> None:
    methods = summary["matched_subset"] or summary["methods"]
    if not methods:
        return
    fig, (score_ax, token_ax) = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained")
    positions = np.arange(len(methods))
    top1 = [100 * row["top1"] for row in methods.values()]
    top3 = [100 * row["top3"] for row in methods.values()]
    first = score_ax.barh(positions - 0.15, top1, height=0.28, color=BLUE, label="Top 1")
    first_three = score_ax.barh(positions + 0.15, top3, height=0.28, color=TEAL, label="Top 3")
    for bars, values in ((first, top1), (first_three, top3)):
        score_ax.bar_label(bars, labels=[f"{value:.0f}%" for value in values], padding=3, fontsize=8)
    score_ax.set(yticks=positions, yticklabels=[LABELS[arm] for arm in methods],
                 xlim=(0, 110), xlabel="Correct component (%)", title="Diagnosis accuracy")
    score_ax.invert_yaxis()
    score_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncols=2, frameon=False)
    token_totals = summary["paired_input_tokens"]
    if summary["successful_llm_pairs"]:
        tokens = [token_totals["full"], token_totals["selected"]]
        bars = token_ax.barh(["Full telemetry", "Selected telemetry"], tokens, color=[BLUE, TEAL], height=0.5)
        token_ax.invert_yaxis()
        token_ax.set_xlim(0, max(tokens) * 1.24)
        if max(tokens) >= 10000:
            token_ax.xaxis.set_major_locator(MultipleLocator(10000))
            token_ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}k"))
        token_ax.bar_label(bars, labels=[f"{value:,}" for value in tokens], padding=4)
        token_ax.set(xlabel="Actual input tokens", title=f"Same {summary['successful_llm_pairs']} model pairs")
    else:
        token_ax.text(0.5, 0.5, "Token comparison needs two valid model arms",
                      ha="center", va="center", transform=token_ax.transAxes)
        token_ax.set_axis_off()
    for axis in (score_ax, token_ax):
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.grid(axis="x", alpha=0.18)
        axis.set_axisbelow(True)
    fig.suptitle("PropelLab fault localization", fontsize=15, fontweight="bold", color=NAVY)
    fig.savefig(output / "comparison.png", dpi=170, facecolor="white")
    plt.close(fig)


def _example_plot(output: Path, run, reference, budget: int) -> None:
    deltas = np.asarray(paired_deltas(run, reference))
    retained = selected_ticks(deltas.tolist(), run, budget)
    peak = np.max(np.abs(deltas), axis=0) / np.array([channel[3] for channel in CHANNELS])
    channels = np.argsort(-peak, kind="stable")[:3]
    first_failure = min(check.failing_ticks[0] for check in run.checks if check.failing_ticks)
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.2), sharex=True, layout="constrained")
    for axis, index in zip(axes, channels, strict=True):
        axis.axhline(0, color="#c6cdd7", lw=1)
        axis.axvline(first_failure, color="#657588", linestyle="--", lw=1)
        axis.plot(deltas[:, index], color=BLUE, lw=1.7)
        axis.scatter(retained, deltas[retained, index], s=24, color=TEAL, zorder=3)
        axis.set_ylabel(SIGNAL_LABELS[CHANNELS[index][1]])
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", alpha=0.16)
    axes[-1].set_xlabel("Test time (seconds)")
    axes[0].set_title("Green dots mark selected rows. The dashed line marks the first failed check.",
                      fontsize=10, loc="left")
    fig.supylabel("Change from the healthy run", fontsize=10)
    fig.suptitle(f"Evidence retained for {run.case.id} · {run.case.scenario.platform}",
                 fontsize=14, fontweight="bold", color=NAVY)
    fig.savefig(output / "example.png", dpi=170, facecolor="white")
    plt.close(fig)


def _dashboard(output: Path, summary: dict, runs: list) -> None:
    table_rows = []
    for run in runs:
        case = run.case
        checks = ", ".join(f"{check.requirement} {check.status}" for check in run.checks)
        values = [case.id, case.scenario.platform, case.scenario.cycle.upper(), case.fault,
                  "Fail" if run.failed else "Pass", checks]
        table_rows.append("<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in values) + "</tr>")
    figure = '<p><img src="comparison.png" alt="Accuracy and token comparison"></p>' if (
        output / "comparison.png").exists() else ""
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>PropelLab validation results</title><style>
body{{font:16px/1.5 system-ui,sans-serif;color:#112640;background:#f5f8fa;margin:0}}
main{{max-width:1120px;margin:auto;padding:36px 24px}} h1{{margin-bottom:0}}
img{{width:100%;max-width:900px;border:1px solid #d2dce7;border-radius:10px}}
table{{border-collapse:collapse;background:white;width:100%;font-size:14px}}
td,th{{padding:8px 12px;border-bottom:1px solid #e0e7ef;text-align:left}}
th{{background:#eaf1f6}} input{{margin:0 0 14px 8px;padding:8px;min-width:15em}}
</style></head><body><main>
<h1>PropelLab validation results</h1>
<p>{summary["runs"]} software-in-the-loop runs · {summary["detected_faults"]} detected seeded faults</p>
{figure}
<h2>Requirement results by run</h2>
<label for="filter">Filter by run, vehicle, component, or requirement</label>
<input type="search" id="filter"><table><thead><tr>
<th>Run</th><th>Vehicle</th><th>Drive cycle</th><th>Seeded component</th><th>Result</th><th>Checks</th>
</tr></thead><tbody id="runs">{''.join(table_rows)}</tbody></table></main><script>
const input = document.getElementById('filter');
const rows = Array.from(document.querySelectorAll('#runs tr'));
input.addEventListener('input', () => {{
 const words = input.value.toLowerCase().trim().split(/\\s+/);
 for (const row of rows) {{
  const text = row.textContent.toLowerCase();
  row.hidden = !words.every(word => text.includes(word));
 }}
}});
</script></body></html>
"""
    (output / "dashboard.html").write_text(page, encoding="utf-8")


def write_report(output: Path, summary: dict, runs: list, references: dict,
                 predictions: list[dict], budget: int) -> None:
    _comparison(output, summary)
    valid = [row for row in predictions if row["method"] == "llm_selected" and row["status"] == "ok"]
    example = next((row for row in valid if row["rank"] == 1), valid[0] if valid else None)
    failed = {run.case.id: run for run in runs if run.failed}
    run = failed.get(example["case_id"]) if example else next(iter(failed.values()), None)
    if run:
        _example_plot(output, run, references[run.case.scenario.id], budget)
    _dashboard(output, summary, runs)

    sections = [
        "# PropelLab experiment report", "",
        "PropelLab tests vehicle-control software and ranks the likely faulty component "
        "when a requirement fails. This experiment asks how much telemetry an LLM needs to diagnose a failure.", "",
        "## What ran", "",
        f"The agent ran **{summary['runs']} software-in-the-loop tests** across engine, battery-electric, "
        f"and range-extended electric vehicles. All three used public EPA driving schedules. "
        f"**{summary['healthy_passes']}/{summary['healthy_runs']}** healthy runs passed. "
        f"The checks detected **{summary['detected_faults']}/{summary['seeded_faults']}** seeded software faults.", "",
    ]
    if summary["missed_by_component"]:
        escaped = ", ".join(f"{label.replace('_', ' ')} ({count})"
                            for label, count in summary["missed_by_component"].items())
        sections.extend([f"Faults that escaped the requirement checks came from {escaped}. "
                         "These runs count toward test coverage and are outside the ranking evaluation.", ""])
    if summary["llm_model"]:
        sections.extend([
            f"The agent used **{summary['llm_model']}** for the two language-model arms. "
            f"The verification plan source was **{summary['plan_source'].replace('_', ' ')}**. "
            "The full arm received every paired telemetry row. The selected arm received up to "
            f"{budget} rows centered on failures, early changes, and large deviations.", "",
        ])
    if summary["plan_error"]:
        sections.extend(["The model plan required a catalog fallback. Details are in verification_plan.json.", ""])
    methods = summary["matched_subset"] or summary["methods"]
    if methods:
        sections.extend([
            "## How well it found the fault", "",
            "Top 1 means the correct component was first. Top 3 means it was among the first three.",
            "The table uses the same sampled cases across methods when an LLM comparison was requested.", "",
            "| Method | Cases | Valid outputs | Top 1 | Top 3 | Median model time |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for method, stats in methods.items():
            timing = f"{stats['median_seconds']:.1f} s" if stats["median_seconds"] is not None else "NA"
            sections.append(f"| {LABELS[method]} | {stats['cases']} | {stats['valid']} | "
                            f"{stats['top1_hits']}/{stats['cases']} | {stats['top3_hits']}/{stats['cases']} | {timing} |")
        sections.extend(["", "![Accuracy and actual token use](comparison.png)", ""])
    if summary["successful_llm_pairs"]:
        tokens = summary["paired_input_tokens"]
        sections.extend([
            f"Across **{summary['successful_llm_pairs']}** valid pairs, selected evidence used "
            f"**{tokens['selected']:,}** input tokens. Full telemetry used **{tokens['full']:,}**. "
            f"The input-token reduction was **{summary['paired_token_reduction']:.1%}**.", "",
        ])
    if run:
        sections.extend(["## One failed run", "",
                         f"The seeded cause in **{run.case.id}** was **{run.case.fault.replace('_', ' ')}**.", ""])
        if example:
            reason = " ".join(example["reason"].split())
            sections.extend([f"The selected-evidence model ranked **{example['ranking'][0].replace('_', ' ')}** first."
                             if example["ranking"] else "The model abstained.",
                             f"Evidence IDs were {', '.join(example['evidence'])}. {reason}", ""])
        sections.extend(["![Signals and selected timestamps](example.png)", ""])
    if followup := summary.get("coverage_followup"):
        count = followup["paired_scenarios"]
        original = followup["conditions"]["original"]["faults_detected"]
        late = followup["conditions"]["late"]["faults_detected"]
        sections.extend([
            "## Improving the generator test", "",
            f"Moving the low-charge stimulus later increased detection from **{original}/{count}** to "
            f"**{late}/{count}** on the same scenario seeds. "
            "This follow-up keeps the original diagnosis benchmark intact.", "",
            "![Generator test coverage before and after](coverage.png)", "",
            "[Read the follow-up](COVERAGE.md)", "",
        ])
    sections.extend([
        "## Scope", "",
        "The benchmark uses simplified powertrain models, original requirements, and single software faults. "
        "The LLM sample is a pilot drawn from held-out scenario seeds. Results describe this simulator. "
        "The full arm ran before the selected arm for each case. Timing can reflect runtime caching. "
        "A physical board run is still needed to measure hardware-in-the-loop agreement and timing.", "",
        "[Browse the dashboard](dashboard.html) · [Metrics](summary.json) · [Runs](runs.csv) · "
        "[Predictions](predictions.jsonl) · [Telemetry](telemetry.jsonl.gz) · "
        "[Research references](../../docs/related-work.md)", "",
    ])
    (output / "REPORT.md").write_text("\n".join(sections), encoding="utf-8")
