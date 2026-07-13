# Interference-aware placement design

## Scientific question

For the mostly single-threaded tools in the current traces, placement is not a
choice between multi-thread layouts such as `compact_l3` and `spread_numa`.
The action is a concrete logical/physical core candidate under the machine
state observed immediately before launch:

\[
a_t^*=\arg\min_{a\in\mathcal A_t}
\mathbb E[L(x_t,a\mid s_t)+\lambda I(x_t,a\mid s_t)].
\]

Here \(L\) is the new tool's latency and \(I\) is interference imposed on
co-runners. A single thread can still conflict with its SMT sibling, other
cores in the shared LLC cluster, and tasks using the same memory controller.

## Required normalized input

Each row used for a real placement decision must contain a pre-launch snapshot:

```json
{
  "resources": {
    "cpu_parallelism": 1.0,
    "placement_candidates": [
      {
        "candidate_id": "cpu-2",
        "core_id": 1,
        "cluster_id": 0,
        "numa_node": 0,
        "core_util": 0.10,
        "smt_sibling_util": 0.75,
        "cluster_util": 0.55,
        "llc_pressure": 0.40,
        "memory_bw_pressure": 0.35,
        "run_queue": 1,
        "frequency_ratio": 0.95
      }
    ]
  },
  "labels": {
    "placement_costs": {
      "cpu-2": {"tool_latency_ms": 1520.0, "peer_slowdown_ms": 120.0},
      "cpu-6": {"tool_latency_ms": 1180.0, "peer_slowdown_ms": 360.0}
    }
  }
}
```

Utilization and pressure fields are normalized to \([0,1]\). `run_queue` is a
non-negative count. `frequency_ratio` is current frequency divided by the
machine's reference frequency. Candidate state must be sampled before launch;
post-outcome counters must not enter the decision features.

For compatibility, `placement_costs[candidate_id]` may be a scalar latency.
The preferred form records both the new tool latency and co-runner slowdown;
the current evaluator uses

\[
C=L_{tool}+0.20L_{peer\ slowdown}.
\]

The weight must be fixed before looking at test outcomes and should be varied
in a sensitivity analysis for a paper.

## Policy score

The current policy infers a demand vector

\[
d(x)=(d_{cpu},d_{mem},d_{cache},d_{io},parallelism)
\]

from online-available tool metadata. Missing parallelism defaults to one rather
than assuming hidden parallelism. For candidate \(c\), the policy estimates

\[
\widehat C(x,c)=\widehat L_{90}(x)
\left[1+J_{self}(d(x),s_c)+\lambda J_{peer}(d(x),s_c)\right].
\]

`J_self` contains explicit interaction terms such as

\[
d_{cpu}u_{core},\quad d_{cpu}u_{SMT},\quad
d_{cache}p_{LLC},\quad d_{mem}p_{BW},
\]

plus run-queue and frequency penalties. `J_peer` penalizes selecting a core
whose SMT/LLC/memory domain is already occupied. The selected action is

\[
\hat c=\arg\min_c\widehat C(x,c).
\]

The hand-written coefficients are a cold-start policy, not a learned hardware
model. They should later be replaced by an action-conditioned model trained on
controlled replay data.

## Evaluation strata

Real and synthetic evidence are never aggregated.

## End-to-end Linux collector

`prepare-placement-study` partitions cases by a stable hash and constructs five
real ToolSched workload families per shard: latency logistic regression,
latency random forest, next-tool logistic regression, remaining-time random
forest, and grouped quantile fitting. These are actual repository computations;
only the co-runners are controlled stressors.

For each invocation \(i\), interference scenario \(s\), candidate CPU \(c\),
and repeat \(r\), the collector records a raw row

\[
Z_{i,s,c,r}=(x_i,s_{pre},c,L_{tool},T_{peer},exit,status).
\]

Candidate order is randomly permuted inside every repeat. `cluster_a_busy` and
`cluster_b_busy` refer to two fixed LLC domains, not to a candidate-relative
oracle label; whether a candidate shares the busy cluster is visible in its
measured state. Co-runner placement
is fixed for the whole `(invocation, scenario)` replay block, so all candidate
actions are compared against the same interference intervention. Candidate
state is sampled before tool launch while co-runners are active.

Linux affinity is applied with `sched_setaffinity` to the shell process and is
inherited by children. Common hidden thread pools are constrained through
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`NUMEXPR_NUM_THREADS`, and `LOKY_MAX_CPU_COUNT` in the study manifest.

CPU utilization comes from a `/proc/stat` delta. Current/reference frequency
comes from cpufreq sysfs. In `--perf-mode auto|required`, the collector tries
per-LLC-domain `perf stat` measurements and defines the portable proxies

\[
p_{cache}=\frac{cache\ misses}{cache\ references},\qquad
p_{memory\ stall}=\frac{backend\ stalled\ cycles}{cycles}.
\]

Generic perf events are not guaranteed to be literal last-level-cache misses or
memory-bandwidth bytes on every microarchitecture; raw provenance therefore
calls them pressure proxies. A machine-specific paper can replace them with
validated uncore/IMC events. If `auto` cannot access perf counters, it records a controlled co-runner proxy
and marks the provenance on every candidate. `required` aborts instead; it is
the stronger setting for publishable hardware claims.

Peer interference is estimated from a short solo-throughput calibration and
the concurrent co-runner throughput:

\[
\Delta L_{peer}
=W\max\left(0,
\frac{Throughput_{solo}}{Throughput_{concurrent}}-1
\right),
\]

where \(W\) is the tool overlap window. Multiple co-runners contribute
additively. The experiment retains both rates, so alternative interference
objectives can be recomputed from raw data.

Aggregation requires at least `min_success_per_candidate` successful repeats
for every action. It uses medians:

\[
\widetilde L_{i,s,c}=median_r L_{i,s,c,r},\qquad
\widetilde I_{i,s,c}=median_r\Delta L_{peer,i,s,c,r}.
\]

The normalized evaluator row then contains

\[
C_{i,s}(c)=\widetilde L_{i,s,c}+0.20\widetilde I_{i,s,c}.
\]

The raw long table, not the aggregated sample, is the primary scientific
artifact. Repeats must remain grouped by invocation/case during splitting and
bootstrap analysis.

The collector reports minimum design gates: at least 20 independent
invocations, 5 repeats, 3 candidate cores, 2 supported interference scenarios,
failure rate at most 10%, hardware perf pressure availability, and at least one
complete counterfactual aggregate. These gates catch toy or broken executions;
passing them does not establish cross-machine external validity.

### Real counterfactual replay

For the same invocation and comparable initial state, run each allowed action
multiple times in randomized order. Report the median cost per action and
retain dispersion and run order in the raw dataset. The evaluator uses

\[
Regret_i=\frac{C_i(\hat c_i)-\min_c C_i(c)}{\min_c C_i(c)}.
\]

Rows missing either candidate state or counterfactual costs are excluded and
counted. They are never filled with synthetic labels.

Mean regret and paired improvement over the OS proxy include deterministic
percentile-bootstrap 95% intervals. Real studies should additionally bootstrap
at the case or replay-block level rather than treating correlated invocations
as independent.

Recommended experimental controls:

- randomize candidate order and include warm-up runs;
- block by machine, power governor, thermal regime, and background workload;
- record SMT topology, LLC domain, NUMA node, frequency, and co-runner phase;
- repeat enough times to report confidence intervals, not only point regret;
- separate single-threaded tools from tools whose `CPU time / wall time > 1`;
- evaluate both tool latency and the slowdown imposed on existing workloads.

### Synthetic stress test

`--mode synthetic` creates heterogeneous candidate states deterministically.
Its hidden oracle is nonlinear, contains bounded latent variation, and has a
different functional form from the policy score. This prevents the old
tautology in which policy and oracle minimized the same resource-class factors.
Synthetic results validate code paths and ranking behavior only.

## Baselines

The evaluator reports:

- `os_proxy`: lowest conventional core/run-queue pressure;
- `least_core_util`: lowest core utilization;
- `smt_aware`: core utilization plus SMT sibling pressure;
- `least_cluster_load`: cluster, LLC, and memory pressure;
- `random`: deterministic random candidate;
- oracle: lowest measured counterfactual cost.

These baselines isolate where gains come from. In particular,
`smt_aware - least_core_util` measures whether sibling awareness matters, and
the comparison with `least_cluster_load` probes LLC/memory-domain effects.

## Claims that are and are not supported

With current legacy artifacts, real placement accuracy is unavailable because
there is no per-candidate pre-launch state or controlled replay. A synthetic
result must never be described as a system speedup. A publishable placement
claim requires real counterfactual data, uncertainty intervals, held-out cases
and machines, and comparison against the baselines above.
