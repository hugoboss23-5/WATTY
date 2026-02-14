# Changelog

All notable changes to Watty will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-02-14

### Added

- **HTTP/SSE transport** — `watty-http` serves all 8 tools over HTTP. ChatGPT, Gemini, Grok now fully supported.
- **onnxruntime backend** — same model, ~100MB instead of ~2GB. PyTorch is now optional.
- **Embedding backend selection** — `WATTY_EMBEDDING_BACKEND=auto|onnx|torch`
- **Conversation importers** — `watty-import-chatgpt`, `watty-import-claude`, `watty-import-json`
- **PyPI publishing** — `pip install watty-ai[onnx]`
- **31 tests** covering brain, importers, HTTP transport, and backend auto-detection

### Changed

- Package name on PyPI: `watty-ai` (since `watty` was taken)
- `sentence-transformers` and `torch` moved to optional `[torch]` extra
- Default install is now lightweight: just `mcp` + `numpy`
- All repo URLs fixed to point to `github.com/hugoboss23-5/WATTY`

### Roadmap

- **v1.2:** Approximate nearest neighbor search for 50k+ memories
- **v1.3:** Encrypted brain, backup/restore CLI, multi-user support

## [1.0.0] - 2026-02-14

### Added

- **8 MCP tools:** `watty_recall`, `watty_remember`, `watty_scan`, `watty_cluster`, `watty_forget`, `watty_surface`, `watty_reflect`, `watty_stats`
- **Semantic memory** using sentence-transformers (`all-MiniLM-L6-v2`, 384 dimensions)
- **Local SQLite storage** at `~/.watty/brain.db` — no cloud, no API keys
- **Smart chunking** with sentence boundary detection (1500 chars, 200 overlap)
- **SHA-256 deduplication** across all storage and scan operations
- **Directory scanning** with automatic file type detection and recursive traversal
- **Agglomerative clustering** for self-organizing knowledge graphs (no sklearn dependency)
- **Novelty-weighted surfacing** that finds surprising connections in memory
- **Recency-weighted search** combining semantic similarity with temporal relevance
- **Cross-platform support:** Claude Desktop, Cursor, Windsurf, Claude Code, VS Code + Copilot
- **CI pipeline** across 9 OS/Python combinations (Ubuntu, macOS, Windows × Python 3.10-3.12)
- **Zero-config defaults** — works out of the box, all settings overridable via environment variables

[1.1.0]: https://github.com/hugoboss23-5/WATTY/releases/tag/v1.1.0
[1.0.0]: https://github.com/hugoboss23-5/WATTY/releases/tag/v1.0.0
