import tempfile
import unittest
from pathlib import Path

from propellab.controller import build_host, host_link
from propellab.evidence import evidence_id, payload
from propellab.simulation import Case, Scenario, load_cycles, load_requirements, simulate, test_matrix


class ValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.native = build_host(Path(cls.temp.name))
        cls.requirements = load_requirements()
        cls.speeds = load_cycles()["udds"]

    def run_case(self, scenario, fault):
        with host_link(self.native) as link:
            return simulate(link, Case(scenario, fault), self.speeds, self.requirements)

    def test_matrix_preserves_grouped_split(self):
        cases = test_matrix(seed=42, replicates=5, development=2)
        self.assertEqual(len(cases), 375)
        self.assertEqual(len({case.id for case in cases}), 375)
        self.assertEqual(len({case.scenario.id for case in cases if case.scenario.split == "held_out"}), 27)
        self.assertEqual(sum(case.fault == "healthy" for case in cases), 45)

    def test_healthy_controller_exercises_diagnostics_on_all_powertrains(self):
        for platform in ("ICE", "BEV", "REEV"):
            with self.subTest(platform=platform):
                scenario = Scenario(platform, "udds", 0, 1234, "held_out")
                run = self.run_case(scenario, "healthy")
                self.assertFalse(run.failed)
                self.assertTrue(all(check.evaluated_ticks > 0 for check in run.checks))

    def test_temperature_defect_has_observable_downstream_failure(self):
        scenario = Scenario("BEV", "udds", 0, 2048, "held_out")
        run = self.run_case(scenario, "temperature_monitor")
        self.assertTrue(run.failed)
        check = next(check for check in run.checks if check.requirement == "DVP-02")
        self.assertEqual(check.status, "fail")
        self.assertFalse(run.trace[check.failing_ticks[0]]["temperature_valid"])
        self.assertEqual(run.trace[check.failing_ticks[0]]["diagnostic"], 0)

    def test_selector_preserves_rows_and_excludes_label(self):
        scenario = Scenario("BEV", "udds", 0, 2048, "held_out")
        healthy = self.run_case(scenario, "healthy")
        faulty = self.run_case(scenario, "temperature_monitor")
        full = payload(faulty, healthy, self.requirements, None)
        selected = payload(faulty, healthy, self.requirements, 18)
        self.assertEqual(len(full["paired_trace"]), 72)
        self.assertEqual(len(selected["paired_trace"]), 18)
        self.assertTrue(set(selected["paired_trace"]).issubset(full["paired_trace"]))
        self.assertNotIn("temperature_monitor", str(selected))
        check = next(check for check in faulty.checks if check.requirement == "DVP-02")
        self.assertIn(evidence_id(check.failing_ticks[0]), {row.split(",")[0] for row in selected["paired_trace"]})


if __name__ == "__main__":
    unittest.main()
