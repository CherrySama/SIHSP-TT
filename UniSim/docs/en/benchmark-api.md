# Benchmark API Reservation

[English](benchmark-api.md) | [中文](../zh/benchmark-api.md)

The `1.x` line reserves `BenchmarkCase` and `BenchmarkResult` as the stable extension point for a future engine benchmark package. No workload runner, timing loop, result comparison, or performance claim is implemented here.

Future benchmark work must fix scene, control, and state semantics, schema version, artifact digest, synchronization, and hardware/runtime provenance. It must reuse the same `SimBackend` contract used by UniLab adapters.
