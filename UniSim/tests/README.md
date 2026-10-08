# Test layout

Tests are grouped by ownership rather than kept as one flat namespace:

- `core/` — import and package-boundary invariants.
- `contract/` — engine-neutral public contract, conformance, rendering, reset, and randomization behavior.
- `factory/` — adapter manifest and factory dispatch coverage.
- `adapters/<engine>/` — implementation and optional-runtime tests for one adapter.

Numerical adapter tests use `pytest.importorskip` or dedicated fixtures so unavailable optional SDKs do not weaken the base test path. Add tests to the most specific subtree; besides package markers, only repository-level test helpers belong directly under `tests/`; the `__init__.py` files prevent duplicate leaf-module names across adapter directories.
