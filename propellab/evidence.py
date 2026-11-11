"""Telemetry evidence shared by the two language-model conditions."""

from __future__ import annotations

from .simulation import Run

COMPONENTS = {
    "request_filter": "Smooths the driver's torque request.",
    "torque_allocator": "Turns the smoothed request and torque cap into a command.",
    "actuator_control": "Tracks the torque command and provides measured torque.",
    "energy_estimator": "Estimates available energy from the reference sensor.",
    "temperature_monitor": "Raises an internal flag for missing or hot temperature readings.",
    "thermal_limiter": "Uses the temperature flag to set the torque cap.",
    "diagnostic_manager": "Publishes the temperature flag as a diagnostic code.",
    "generator_supervisor": "Starts and stops the range extender based on the energy estimate.",
}

# Deviations are expressed as integer hundredths for continuous normalized
# channels and integer code changes for discrete channels.
CHANNELS = (
    ("request_filtered", "req", 100, 6),
    ("energy_estimate", "energy", 100, 5),
    ("thermal_flag", "temp_flag", 1, 1),
    ("torque_cap", "cap", 100, 8),
    ("torque_command", "cmd", 100, 6),
    ("torque_measured", "actual", 100, 6),
    ("diagnostic", "diag", 1, 1),
    ("generator", "gen", 1, 1),
)


def paired_deltas(run: Run, reference: Run) -> list[tuple[int, ...]]:
    if run.case.scenario != reference.case.scenario or reference.case.fault != "healthy":
        raise ValueError("Paired reference must be the healthy run from the same scenario")
    return [
        tuple(round((observed[key] - healthy[key]) * scale) for key, _, scale, _ in CHANNELS)
        for observed, healthy in zip(run.trace, reference.trace, strict=True)
    ]


def selected_ticks(deltas: list[tuple[int, ...]], run: Run, budget: int) -> list[int]:
    if budget <= 0:
        raise ValueError("Evidence budget must be positive")
    scale = [channel[3] for channel in CHANNELS]
    scores = [sum(abs(value) / scale[i] for i, value in enumerate(row)) for row in deltas]
    anchors = sorted({check.failing_ticks[0] for check in run.checks if check.failing_ticks})
    priorities = list(anchors)
    onsets = []
    peaks = []
    for index, tolerance in enumerate(scale):
        magnitude = [abs(row[index]) for row in deltas]
        peak = max(range(len(magnitude)), key=magnitude.__getitem__)
        if magnitude[peak] >= tolerance:
            first = next(tick for tick, value in enumerate(magnitude) if value >= tolerance)
            weight = magnitude[peak] / tolerance
            onsets.append((weight, first))
            peaks.append((weight, peak))
    priorities.extend(tick for _, tick in sorted(onsets, reverse=True))
    priorities.extend(tick - 1 for tick in anchors if tick > 0)
    priorities.extend(tick for _, tick in sorted(peaks, reverse=True))
    priorities.extend(tick + 1 for tick in anchors if tick + 1 < len(deltas))
    priorities.extend(sorted(range(len(deltas)), key=lambda tick: (-scores[tick], tick)))
    return sorted(list(dict.fromkeys(priorities))[:budget])


def evidence_id(tick: int) -> str:
    return f"E{tick:03d}"


def payload(run: Run, reference: Run, requirements: list[dict], budget: int | None) -> dict:
    deltas = paired_deltas(run, reference)
    ticks = list(range(len(deltas))) if budget is None else selected_ticks(deltas, run, budget)
    catalog = {requirement["id"]: requirement for requirement in requirements}
    return {
        "case_id": run.case.id,
        "powertrain": run.case.scenario.platform,
        "checks": [
            {"id": check.requirement, "status": check.status,
             "description": catalog[check.requirement]["text"],
             "first_failure_second": check.failing_ticks[0] if check.failing_ticks else None}
            for check in run.checks
        ],
        "columns": ["evidence_id", "second", *(channel[1] for channel in CHANNELS)],
        "paired_trace": [
            ",".join(map(str, (evidence_id(tick), tick, *deltas[tick]))) for tick in ticks
        ],
    }


def rules_ranking(run: Run, requirements: list[dict]) -> list[str]:
    catalog = {requirement["id"]: requirement for requirement in requirements}
    failing = [check for check in run.checks if check.failing_ticks]
    failing.sort(key=lambda check: (check.failing_ticks[0], check.requirement))
    candidates = [
        component for check in failing for component in catalog[check.requirement]["candidates"]
        if component != "generator_supervisor" or run.case.scenario.platform == "REEV"
    ]
    return list(dict.fromkeys(candidates))[:3]
