# Phase 4: Custom Format Testing - Context

**Gathered:** 2026-01-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Systematic testing of exotic FP8 formats (E0M7, E1M6, E3M4, E7M0) with E5M2 as control. Document failure modes, especially E7M0 as expected negative result. Implement stability interventions (partition-relative clipping, emergency mantissa shift). This phase runs experiments and documents findings — the manifold-aware optimizer is Phase 5.

</domain>

<decisions>
## Implementation Decisions

### Experiment Sequencing
- Run order: Safest first (E3M4 → E1M6 → E7M0), with E5M2 as control
- Include E0M7 for completeness (test all 5 formats)
- Short runs (500 steps) to maximize format coverage within budget
- Same seed (42) for all experiments for fair comparison
- Stop on first NaN for failing formats (capture exact failure point)

### E1M6 Strategy
- Test everywhere first (uniform), expect likely failure due to narrow range
- If fails, retry with selective application (post-LayerNorm tensors only)

### Failure Documentation
- Standard detail level: step, loss, gradient stats, overflow counts
- Save checkpoint on first NaN for later forensic analysis
- For E7M0: log both gradient sparsity (bit-stall) AND overflow rates per layer
- Output both W&B metrics AND standalone markdown failure analysis report

### Stability Interventions
- Partition-relative gradient clipping: activate on overflow detection (not always-on)
- Overflow threshold to trigger clipping: 1% (moderate)
- Emergency mantissa shift: enabled
- Shift triggers on EITHER persistent NaN (3+ consecutive batches) OR high bit-stall rate (>50%)

### Per-layer Format Assignment
- Initial test: uniform (all layers same format)
- If uniform fails: retry with mixed precision
- Claude's discretion on which layers stay BF16 based on where overflow/stall occurs
- Claude's discretion on embedding/LM head handling
- Claude's discretion on whether to test mixed configs for successful uniform formats (budget dependent)

### Claude's Discretion
- Which layers to keep in BF16 for mixed-precision fallback (based on observed failures)
- Whether embedding and LM head need BF16 protection
- Whether to explore mixed configs for formats that work uniformly
- Exact implementation of partition-relative clipping algorithm
- Mantissa shift mechanics (which direction: E→M or M→E)

</decisions>

<specifics>
## Specific Ideas

- E7M0 is expected to fail — the scientific value is documenting exactly how and when
- Budget constraint ($20 H100) means short runs and no reruns — get it right first time
- Failure checkpoint enables post-mortem analysis without re-running experiment
- Markdown failure report + W&B provides both narrative and data views

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-custom-format-testing*
*Context gathered: 2026-01-21*
