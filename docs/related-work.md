# Related work

PropelLab addresses the publicly described workflow in Stellantis's
[Propulsion Automation Validation Lead posting](https://careers.stellantis.com/job/24108371/propulsion-automation-validation-lead-auburn-hills-mi/).
It covers verification planning, software and hardware test execution, issue
isolation, and release evidence for engine, battery-electric, and range-extended
electric propulsion. The requirements and controllers here are original
research fixtures. Stellantis's private systems would be needed for an
implementation-level comparison.

| Research | How it informs PropelLab |
|---|---|
| [Multi-Step Generation of Test Specifications using Large Language Models for System-Level Requirements](https://aclanthology.org/2025.acl-industry.11/) (ACL Industry 2025) | Supply explicit requirements and ask for structured test specifications. Validate each draft against the requirement catalog before it enters the plan. |
| [Smarter, not Bigger: Fine-Tuned RAG-Enhanced LLMs for Automotive Hardware-in-the-Loop Testing](https://aclanthology.org/2026.acl-industry.2/) (ACL Industry 2026) | Keep requirements, cases and measurements traceable. Retrieve the relevant test artifacts when explaining a failure. |
| [Sensor-Level Fault Diagnosis for Automotive Software Validation Using Large Language Models](https://arxiv.org/abs/2608.05921) (2026 preprint) | Give the model structured time-series evidence and score its diagnosis against a known fault. PropelLab asks which software component caused a failure in a simulated closed-loop test. |
| [SieveFL: Hierarchical Runtime-Aware Pruning for Scalable LLM-Based Fault Localization](https://arxiv.org/abs/2605.13491) (2026 preprint) | Filter runtime evidence before asking the model to rank causes. Compare accuracy and input size using the same model and held-out cases. |
| [ASAM XIL](https://www.asam.net/standards/detail/xil/) | Separate test logic from the bench interface so a test can run against either a host controller or a physical controller. The project uses its own small serial protocol. |

## Research question

Can an AdaptiveContextOpt-style selector reduce the telemetry sent to the
language model while preserving the model's ability to locate a software fault?
The experiment gives both model arms the same requirement results, candidate
components and paired telemetry encoding. The selected-evidence arm sees fewer
rows. Both arms are scored against the same withheld injected component.

The [EPA drive schedules](https://www.epa.gov/vehicle-and-fuel-emissions-testing/dynamometer-drive-schedules)
provide the speed inputs. The simulator generates sensor measurements and fault
labels. See the data manifest for the source URLs and checksums.
