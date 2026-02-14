# Changelog

All notable changes to Watty will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- **17 smoke tests** with mock embeddings (no PyTorch download needed to test)
- **CI pipeline** across 9 OS/Python combinations (Ubuntu, macOS, Windows × Python 3.10-3.12)
- **Zero-config defaults** — works out of the box, all settings overridable via environment variables

### Roadmap

- **v1.1:** HTTP/SSE transport for ChatGPT, Gemini, Grok
- **v1.2:** `onnxruntime` backend option (~100MB vs ~2GB PyTorch)
- **v1.3:** Approximate nearest neighbor search for 50k+ memories

[1.0.0]: https://github.com/hugoboss23-5/WATTY/releases/tag/v1.0.0
