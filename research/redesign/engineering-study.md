# Quantized thermal control reference study

## Scope

This study is a small, deterministic reference problem for derivative-free
optimization. It does not model a real building, equipment schedule, weather
record, or field performance.

The plant has one zone temperature state. The actuator supplies heating or
cooling power. The simulator uses an explicit discrete heat balance:

```
T[k+1] = T[k] + dt/C * ((T_out[k] - T[k]) / R + q_internal[k] + q_act[k])
```

This follows the usual lumped RC control model. ASHRAE describes thermal
networks as algebraic and differential equations, and describes transient
room-temperature models as functions of weather, gains, control actions, and
past temperatures. EnergyPlus also calculates a zone air heat balance at each
time step before it decides whether cooling or heating is required.

Sources:

- [ASHRAE, Chapter 19: Energy estimating and modeling methods](https://handbook.ashrae.org/Handbooks/F25/IP/F25_Ch19/f25_ch19_ip.aspx)
- [ASHRAE, Chapter 42: Supervisory control strategies and optimization](https://handbook.ashrae.org/Handbooks/A15/SI/A15_CH42/a15_ch42_si.aspx)
- [EnergyPlus engineering reference](https://energyplus.readthedocs.io/en/latest/guides/engineering-reference/index.html)

The model has deliberate limits. It omits humidity, radiation, occupancy
models, equipment dynamics, measurement noise, and parameter uncertainty.
Use a validated building simulator before any engineering decision.

## Fixed protocol

`examples/06_engineering_control.py` defines eight train scenarios and five
held-out scenarios. They are deterministic trajectories of outdoor temperature,
internal gain, and initial temperature. Their hashes identify the exact split.

The command plan has 16 controls at 15-minute intervals. Each command is
clamped to [-1, 1], then quantized to 0.25. Saturation and quantization make
the objective nonsmooth. The cost penalizes comfort-band violations, mean
command magnitude, command switching, and the worst scenario comfort cost.

The evaluator is frozen before optimizer tuning. Its reference feasibility
gate is `score <= 3.50`. This is an audit threshold for this synthetic task.
It is not a comfort, safety, or energy-performance limit.

Version `engineering-control-v1` failed its evaluator audit. The zero-control
plan passed its provisional gate while the PI baseline failed it. Version
`engineering-control-v2` raises the warm-side scenario load before any DFO
run. It also records the actual random pass rate instead of treating a
nonzero rate as a failed audit.

Each DFO seed receives 96 train candidate evaluations. Full population methods
may use fewer when a final partial generation cannot update their state. The
report records the actual number. It also evaluates the final train-selected
candidate once on all held-out scenarios. Reports include candidate counts and
scenario-rollout counts, including validation.

## Evaluator audit

Run the four probes on the train split before any optimizer comparison. The
held-out split remains untouched until final candidates are selected.

1. Record the all-zero plan score and gate result.
2. Record the fixed-gain mean-scenario PI score and gate result.
3. Record a constant -0.5 cooling plan as the cheap exploit.
4. Record the fixed-seed uniform-random score distribution and pass rate.

The constant-cooling exploit can pass because it uses one scalar value and has
no switching. The benchmark therefore includes a fixed constant-control grid
as a real baseline. It does not hide that result behind a random comparison.
The separate runaway all-heating check shows that energy and comfort costs
reject an unsafe sign error. Uniform random search should have a much larger
mean score. A random pass is evidence about this small search space, not a
reason to change the evaluator after tuning begins.

The PI controller is a reachable domain baseline. It uses the known RC model
and the mean disturbance trajectory to produce one quantized open-loop plan.
It does not receive an objective-evaluation budget. It still pays the recorded
held-out validation rollouts.

## Comparison and interpretation

Before the five-seed report, the study runs three train-only CMA-ES trials:
`sigma0 = 0.15`, `0.30`, and `0.55`. Each uses seed 10,001 and 96 requested
candidate evaluations. The lower final train score wins. Ties select the
smaller sigma. The ledger retains every trial. The held-out split is not read
during this selection.

The benchmark then runs random search, CMA-ES, and SHADE for five fixed seeds.
It reports every seed. It does not select a favorable seed or report only a
training score. It compares each final train-selected plan on the held-out
split. A low score only shows behavior on these fixed synthetic scenarios.

`benchmarks/engineering_control.py --evotorch --evotorch-src PATH` also runs
upstream EvoTorch CMA-ES. It uses a 12-member population, a zero center,
inactive covariance updates, and the same absolute initial standard deviation
as torch-dfo. EvoTorch's CMA-ES implementation calls `problem.ensure_unbounded`
in `work/evotorch/src/evotorch/algorithms/cmaes.py`. The adapter therefore
gives EvoTorch an unbounded latent problem. The physical simulator clamps each
latent command to [-1, 1] before it scores the plan. The adapter clamps the
reported incumbent too. EvoTorch therefore adapts unbounded proposals, while
torch-dfo samples a bounded space. The report records this difference. The
flag stays optional so the reference example has no EvoTorch dependency.

## CPU reference result

On CPU with Torch 2.10.0 and Python 3.14.4, the train-only trial selected
`sigma0 = 0.30`. The trial train scores were 2.187, 2.049, and 2.153 for
`sigma0 = 0.15`, `0.30`, and `0.55`. The final five-seed run used 1,738 train
candidate evaluations and 17 held-out candidate evaluations. The evaluator
audit used another 68 train candidates.

The PI plan scored 3.241 on held-out scenarios and passed the 3.50 reference
gate. The constant grid scored 4.078. CMA-ES and SHADE both had a mean
held-out score of about 4.845, and each had zero gate passes across five seeds.
Their train scores were near 2.0. This is a generalization failure for this
small synthetic split. The study does not claim an optimization win.

The unit test compares the simulator with an independent unrolled recurrence.
It also checks saturation, quantization, split identity, evaluator stability,
and the four probes.
