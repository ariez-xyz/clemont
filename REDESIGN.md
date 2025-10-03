# Clemont API Redesign Notes

## Goals
- Introduce a new FRNN abstraction that cleanly separates neighbor search from monitoring logic.
- Allow experimentation with multiple monitoring strategies (binary, quantitative, etc.) built on top of the same FRNN utilities.
- Keep the legacy API running in parallel until the new stack reaches feature/behaviour parity, enabling side-by-side verification.

## FRNN Layer
- Create `clemont/frnn/` to house the new search backends.
- Add `clemont/frnn/base.py` with:
  - `FRNNBackend` abstract base class defining `query(point, radius=None)` and `add(point, point_id)` plus standard metadata (`epsilon`, `metric`, `is_sound`, `is_complete`) and shared helpers (`supported_metrics()`, `emulate_range_query`).
  - `FRNNResult` value object exposing `ids` and optional `distances` (can be `None` for backends like BDD).
- Reimplement each backend under the FRNN contract from scratch, borrowing algorithms and tuning from the legacy code but keeping the new classes decision-agnostic:
  - `clemont/frnn/faiss.BruteForceFRNN` purely manages FAISS indices.
  - Follow-up modules for KDTree, SNN, and BDD mirror the same approach without embedding label logic.
- `query` may accept an optional `radius` override; implementations that cannot change radius at runtime (BDD) should raise a clear error (e.g., `UnsupportedRadiusOverride`).
- Standardize metric names across backends via shared normalization helpers so callers interact with a single canonical vocabulary (e.g., `linf`, `l2`) and must use those names explicitly.

## Monitor Layer
- Implement `Monitor` in `clemont/monitor.py`.
- Responsibilities:
  - Keep FRNN instances organized by decision label (initial strategy mirrors current behaviour with one index per class).
  - On `observe(point, decision, *, point_id)`:
    1. Query FRNN instances belonging to *other* decision labels.
    2. Union returned neighbour IDs.
    3. Insert the new point into the FRNN instance for its own label.
    4. Return the counterexample set (and later, quantitative metrics).
  - Enforce non-optional `point_id`; provide a simple incremental ID helper for callers that do not manage IDs manually.
  - Accept backend factories directly (lambdas, partials, classes) that capture all configuration internally, keeping the monitor agnostic to backend signatures.
- Because decision handling lives here, the FRNN backends stay decision-agnostic. Alternate strategies (single shared index + filtering, custom scoring monitors, etc.) can be added without touching FRNN code.

## BDD Refactor
- Restructure BDD-specific helpers into a `clemont/bdd/` package:
  - Move current `clemont/backends/discretization.py` into `clemont/bdd/discretization.py` (bin creation, valuation helpers).
  - Extract reusable BDD formula builders, valuation hashing, and constants into `clemont/bdd/utils.py`.
  - Keep `clemont/backends/bdd.BDD` as a thin wrapper importing from the new modules so the legacy API remains intact during transition.

## Test Strategy
- Add pytest coverage that instantiates both the legacy backend and the new FRNN implementation + `Monitor`, streaming identical data and asserting identical counterexamples.
- Include edge cases:
  - Optional radius override success (FAISS/KDTree) vs. failure (BDD).
  - Mandatory `point_id` behaviour.
- Expand tests incrementally as additional FRNN backends land.

## Next Steps
1. [X] Land scaffolding (`frnn/base.py`, FAISS FRNN implementation, `Monitor`, initial tests) without changing existing API surface.
2. [ ] Add remaining FRNN backends and parity tests. (WIP: so far, FAISS and KDTree are implemented and tested)
3. [ ] Migrate BDD helpers into `clemont/bdd/`, update imports.
4. [ ] Once parity is proven, deprecate and eventually retire the legacy `.observe` interface.
