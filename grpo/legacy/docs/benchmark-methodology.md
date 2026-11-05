# Benchmark Methodology and Workload Semantics

**Status:** Complete

## Correction

The two architectures used the same frozen configuration-level workload, but stochastic generation produced different realized token counts.
Therefore the experiment is classified as a matched-configuration system comparison rather than an exact realized-token workload comparison.

## Realized workload

- Disaggregated measured-token totals: [11465.0, 10979.0, 12191.0]
- Colocated measured-token totals: [9295.0, 8601.0, 8528.0]
- Disaggregated mean measured tokens: 11545.000
- Colocated mean measured tokens: 8808.000
- Disaggregated realized-token workload difference: 31.074%

## Primary normalized result

- Disaggregated median system throughput: 453.790 tokens/s
- Colocated median system throughput: 213.432 tokens/s
- Disaggregated throughput advantage: 112.615%

## Secondary descriptive result

- Disaggregated median step time: 5.060302 s
- Colocated median step time: 7.995051 s
- Descriptive step-time reduction: 36.707%
- Raw step time is not claimed as an exact realized-workload-matched metric.

## Interpretation boundary

- Matched configuration: true
- Exact realized token workload match: false
- Token-normalized system throughput comparison: true
- Single-variable placement causal claim: false
- General performance recommendation: false
- Cross-hardware claim: false
- Statistical-significance claim: false
- Convergence claim: false
