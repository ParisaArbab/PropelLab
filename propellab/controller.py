"""Interfaces for the shared native controller and the physical bench."""

from __future__ import annotations

import math
import os
import shlex
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

SIGNALS = (
    "request_filtered", "energy_estimate", "thermal_flag", "torque_cap",
    "torque_command", "torque_measured", "diagnostic", "generator",
)
PLATFORMS = ("ICE", "BEV", "REEV")
FAULTS = (
    "healthy", "request_filter", "torque_allocator", "actuator_control",
    "energy_estimator", "temperature_monitor", "thermal_limiter",
    "diagnostic_manager", "generator_supervisor",
)


def build_host(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "propellab-controller"
    source = Path(__file__).with_name("host.cpp")
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    subprocess.run(
        [*compiler, "-std=c++17", "-O2", "-Wall", "-Wextra",
         str(source), "-o", str(output)],
        check=True, capture_output=True, text=True,
    )
    return output


class ControllerLink:
    def __init__(self, exchange, mode: str):
        self.exchange = exchange
        self.mode = mode

    def reset(self, platform: str, fault: str, onset: int, severity: float) -> None:
        result = self.exchange(
            f"R,{PLATFORMS.index(platform)},{FAULTS.index(fault)},{onset},{severity:.7f}"
        )
        if result != "OK,propellab-controller-1":
            raise RuntimeError(f"Controller reset failed: {result!r}")

    def step(self, demand: float, energy: float, valid: bool, temperature: float) -> dict:
        response = self.exchange(f"S,{demand:.7f},{energy:.7f},{int(valid)},{temperature:.7f}")
        fields = response.split(",")
        if len(fields) != len(SIGNALS) + 1 or fields[0] != "T":
            raise RuntimeError(f"Unexpected controller response: {response!r}")
        measurements = [float(field) for field in fields[1:]]
        if not all(math.isfinite(value) for value in measurements):
            raise RuntimeError("Controller returned a non-finite measurement")
        return dict(zip(SIGNALS, measurements, strict=True))


@contextmanager
def host_link(executable: Path):
    with subprocess.Popen(
        [str(executable.resolve())], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, bufsize=1,
    ) as process:
        def exchange(line: str) -> str:
            process.stdin.write(line + "\n")
            process.stdin.flush()
            return process.stdout.readline().strip()

        try:
            yield ControllerLink(exchange, "sil")
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@contextmanager
def serial_link(port: str, baudrate: int = 115200):
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("Install PropelLab's bench dependency to use a serial controller") from exc

    with serial.Serial(port, baudrate=baudrate, timeout=2, write_timeout=2) as device:
        time.sleep(2)
        device.reset_input_buffer()

        def exchange(line: str) -> str:
            device.write((line + "\n").encode("ascii"))
            return device.readline().decode("ascii").strip()

        yield ControllerLink(exchange, "hil")
