"""Paired propulsion runs, requirement checks, and a deterministic test matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .controller import FAULTS, PLATFORMS, ControllerLink

DATA = Path(__file__).with_name("data")
CYCLES = ("udds", "hwfet", "us06")
TICKS = 72


@dataclass(frozen=True)
class Scenario:
    platform: str
    cycle: str
    replicate: int
    seed: int
    split: str

    @property
    def id(self) -> str:
        return f"{self.platform.lower()}-{self.cycle}-{self.replicate + 1}"


@dataclass(frozen=True)
class Case:
    scenario: Scenario
    fault: str

    @property
    def id(self) -> str:
        material = f"{self.scenario.id}:{self.scenario.seed}:{self.fault}"
        return "case-" + hashlib.sha256(material.encode()).hexdigest()[:10]


@dataclass(frozen=True)
class Check:
    requirement: str
    evaluated_ticks: int
    failing_ticks: tuple[int, ...]

    @property
    def status(self) -> str:
        if not self.evaluated_ticks:
            return "not_exercised"
        return "fail" if self.failing_ticks else "pass"


@dataclass
class Run:
    case: Case
    trace: list[dict]
    checks: list[Check]
    mode: str
    deadline_misses: int

    @property
    def failed(self) -> bool:
        return any(check.failing_ticks for check in self.checks)

    def record(self) -> dict:
        return {
            "id": self.case.id, "scenario": asdict(self.case.scenario),
            "injected_component": self.case.fault, "mode": self.mode,
            "deadline_misses": self.deadline_misses, "failed": self.failed,
            "checks": [{**asdict(check), "status": check.status} for check in self.checks],
            "telemetry": self.trace,
        }


def load_requirements() -> list[dict]:
    return json.loads((DATA / "requirements.json").read_text(encoding="utf-8"))


def load_cycles() -> dict[str, list[float]]:
    cycles = {}
    for name in CYCLES:
        with (DATA / "cycles" / f"{name}.csv").open(newline="", encoding="utf-8") as file:
            cycles[name] = [float(row["speed_mph"]) for row in csv.DictReader(file)]
    return cycles


def test_matrix(seed: int, replicates: int, development: int) -> list[Case]:
    source = random.Random(seed)
    cases = []
    for platform in PLATFORMS:
        for cycle in CYCLES:
            for replicate in range(replicates):
                split = "development" if replicate < development else "held_out"
                scenario = Scenario(platform, cycle, replicate, source.randrange(2**32), split)
                allowed_faults = FAULTS if platform == "REEV" else FAULTS[:-1]
                cases.extend(Case(scenario, fault) for fault in allowed_faults)
    return cases


def _reference_speed(speeds: list[float], randomizer: random.Random) -> list[float]:
    offsets = range(0, len(speeds) - TICKS + 1, 8)
    moving = [offset for offset in offsets if sum(speeds[offset:offset + TICKS]) / TICKS > 14]
    start = randomizer.choice(moving or list(offsets))
    return speeds[start:start + TICKS]


def simulate(link: ControllerLink, case: Case, speeds: list[float], requirements: list[dict],
             *, late_charge: bool = False) -> Run:
    scenario = case.scenario
    if late_charge and scenario.platform != "REEV":
        raise ValueError("The late low-charge stimulus requires a range extender")
    randomizer = random.Random(scenario.seed)
    speed_schedule = _reference_speed(speeds, randomizer)
    onset = randomizer.randint(11, 24)
    severity = randomizer.uniform(0.42, 0.96)
    dropout_start = randomizer.randint(28, 35)
    hot_start = randomizer.randint(46, 51)
    energy = randomizer.uniform(0.327, 0.353) if scenario.platform == "REEV" else randomizer.uniform(0.65, 0.81)
    temperature = randomizer.uniform(67, 78)
    tick_start = time.monotonic()
    deadline_misses = 0
    trace = []
    link.reset(scenario.platform, case.fault, onset, severity)

    for tick, speed in enumerate(speed_schedule):
        if link.mode == "hil":
            remaining = tick_start + tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            elif tick and remaining < -0.1:
                deadline_misses += 1
        acceleration = speed - speed_schedule[max(0, tick - 1)]
        if late_charge and tick < 30:
            energy = max(energy, 0.34)
        grade_load = 0.23 if 12 <= tick < 26 or 45 <= tick < 64 else 0.0
        demand = min(1.0, max(0.0, 0.30 + speed / 126 + acceleration * 0.028 + grade_load
                              + randomizer.uniform(-0.018, 0.018)))
        valid = not dropout_start <= tick < dropout_start + 5
        overheat = hot_start <= tick < hot_start + 8
        measured_temp = (108.0 + randomizer.uniform(0.0, 3.0) if overheat else
                         temperature + randomizer.uniform(-0.35, 0.35))
        output = link.step(demand, energy, valid, measured_temp if valid else 0.0)
        trace.append({
            "t": tick, "speed_mph": speed, "driver_request": round(demand, 6),
            "energy_reference": round(energy, 6), "temperature_valid": int(valid),
            "temperature_c": round(measured_temp, 4) if valid else None,
            **output,
        })
        torque = output["torque_measured"]
        regen = 0.00025 * min(3.0, max(0.0, -acceleration)) if scenario.platform != "ICE" else 0.0
        load = (0.001 + 0.0024 * torque) * (1.08 if scenario.platform == "ICE" else 1.0)
        energy = min(1.0, max(0.0, energy - load + regen + 0.0044 * output["generator"]))
        temperature += 0.24 * torque - 0.038 * (temperature - 71.0)
    return Run(case, trace, check_requirements(trace, scenario.platform, requirements),
               link.mode, deadline_misses)


def check_requirements(trace: list[dict], platform: str, requirements: list[dict]) -> list[Check]:
    active = {req["id"]: req for req in requirements if platform in req["platforms"]}
    observations = {identifier: [] for identifier in active}
    missing = hot = low = recovered = 0
    saw_dropout = False
    filtered = expected_torque = 0.0
    for row in trace:
        valid = bool(row["temperature_valid"])
        is_hot = valid and row["temperature_c"] >= 103.0
        missing = 0 if valid else missing + 1
        hot = hot + 1 if is_hot else 0
        low = low + 1 if row["energy_reference"] < 0.30 else 0
        saw_dropout |= not valid
        recovered = recovered + 1 if saw_dropout and valid and not is_hot else 0
        filtered = 0.38 * filtered + 0.62 * row["driver_request"]
        limit = 0.33 if is_hot else (0.48 if missing >= 2 else 1.0)
        expected_torque = 0.24 * expected_torque + 0.76 * min(filtered, limit)
        checks = {
            "DVP-01": (row["t"] >= 5 and valid and not is_hot and recovered != 1,
                       abs(row["torque_measured"] - expected_torque) <= active["DVP-01"]["limit"]),
            "DVP-02": (missing >= 3, row["diagnostic"] == 1),
            "DVP-03": (hot >= 3, row["torque_measured"] <= active["DVP-03"]["limit"]),
            "DVP-04": (row["t"] >= 3,
                       abs(row["energy_estimate"] - row["energy_reference"]) <= active["DVP-04"]["limit"]),
            "DVP-05": (low >= 3 or (low > 0 and row["generator"] == 1), row["generator"] == 1),
            "DVP-06": (recovered >= 3, row["diagnostic"] == 0),
        }
        for identifier in active:
            exercised, passed = checks[identifier]
            if exercised:
                observations[identifier].append((row["t"], passed))
    return [
        Check(identifier, len(samples), tuple(tick for tick, passed in samples if not passed))
        for identifier, samples in observations.items()
    ]
