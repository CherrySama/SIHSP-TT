# Contributing

Use `make sync` for the locked development environment, then run `make check` and `make package` before opening a pull request. Keep the core package free of UniLab and engine-SDK imports; optional adapters must remain lazy and declare their dependencies explicitly. Every public contract change needs focused tests, documentation, and a changelog entry. Release tags and the OIDC PyPI workflow are documented in [`docs/en/release.md`](docs/en/release.md).

Documentation is versioned in parallel under `docs/en/` and `docs/zh/`. When adding or changing a document, update both language trees, keep their filenames and structural elements aligned, and avoid hard-wrapped prose: break lines at paragraph, list-item, table-row, or code boundaries rather than mid-sentence.
