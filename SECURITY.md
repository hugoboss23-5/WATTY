# Security Policy

## Threat Model

Watty is a **local-only** MCP server. Your data never leaves your machine.

- All memory is stored in a local SQLite database (`~/.watty/brain.db`)
- Embeddings are computed locally using a downloaded model (no API calls)
- Watty communicates with AI clients over local stdio only
- No network requests are made after the initial model download

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | ✅         |

## Reporting a Vulnerability

If you discover a security issue, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Email: **security@watty.ai** (or open a private security advisory on GitHub)
3. Include steps to reproduce, potential impact, and any suggested fix
4. You will receive a response within 48 hours

## What We Consider In-Scope

- Unauthorized access to `brain.db` contents
- Data exfiltration through the MCP protocol
- Memory injection or poisoning attacks
- Path traversal via `watty_scan`
- Denial of service through malicious inputs

## What We Consider Out-of-Scope

- Physical access to the machine where Watty runs
- Vulnerabilities in upstream dependencies (report those to the respective projects)
- Social engineering attacks against users
