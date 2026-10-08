# Benchmark API 预留

[English](../en/benchmark-api.md) | [中文](benchmark-api.md)

`1.x` 版本线将 `BenchmarkCase` 与 `BenchmarkResult` 预留为未来引擎 benchmark 包的稳定扩展点。这里不实现负载运行器、计时循环、结果比较或性能声明。

未来的 benchmark 工作必须固定场景、控制和状态语义、schema 版本、产物摘要、同步方式，以及硬件/运行时来源。它必须复用 UniLab 适配器使用的同一个 `SimBackend` 契约。
