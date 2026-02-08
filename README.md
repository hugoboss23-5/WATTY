# Watty

**Your phone finally knows you.**

Watty is an MCP server that runs as a native iOS/macOS app. It does two things:

1. **The Daily Brief** — One text message every morning at 7am. It knows your day.
2. **The Recall** — "Hey Siri, what was that idea about shapes and economics?" Semantic search across your life.

That's it. Nothing else ships in v1.

## Architecture

```
WattyApp/       → SwiftUI app. One screen. One button. Then invisible.
WattyCore/      → The brain. Models, ingestion, embedding, intelligence, storage.
WattyMCP/       → MCP server. 7 tools. JSON-RPC over stdio.
WattyIntents/   → Siri + Shortcuts integration.
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `watty_recall` | Semantic search across all memories |
| `watty_brief` | Generate/retrieve daily brief |
| `watty_commitments` | List open commitments |
| `watty_store` | Save a memory from any AI conversation |
| `watty_clusters` | View auto-organized knowledge topics |
| `watty_calendar_prep` | Context for upcoming meetings |
| `watty_contact_context` | Relationship context for any contact |

## Data Sources

| Source | Status | Framework |
|--------|--------|-----------|
| Calendar | Working | EventKit |
| Reminders | Working | EventKit |
| Contacts | Working | Contacts |
| Messages | Stub | App Intents (iOS 26.4+) |
| Mail | Stub | App Intents (iOS 26.4+) |
| Notes | Stub | App Intents (iOS 26.4+) |

## How It Works

1. **Ingestion** — Reads your calendar, reminders, contacts (messages/mail/notes when available)
2. **Embedding** — On-device sentence vectors via `NLEmbedding` (512-dim, Neural Engine)
3. **Storage** — SwiftData with Keychain encryption. Everything stays on-device.
4. **Intelligence** — Cross-references sources, extracts commitments, clusters by topic
5. **Delivery** — Daily brief at 7am via notification. Recall via Siri or MCP.

## Privacy

- All data stays on your device. No cloud. No sync. No telemetry.
- Encryption via Keychain + Secure Enclave.
- No network calls ever (except what Apple frameworks require).

## Requirements

- iOS 26+ / macOS 26+
- Xcode 26+
- Swift 6.0

## Setup

```bash
./Scripts/setup.sh
```

## License

MIT License. See [LICENSE](LICENSE).

---

*Ship the daily brief. Ship it perfectly.*
