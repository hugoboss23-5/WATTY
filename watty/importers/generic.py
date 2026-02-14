"""
Import generic JSON conversation files into Watty.
Expects a JSON array of {"role": "...", "content": "..."} objects.
Usage: watty-import-json ~/conversations.json
"""

import json
import sys
from pathlib import Path

from watty.brain import Brain


def import_json(path: str, provider: str = "import") -> dict:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return {"error": f"File not found: {path}", "conversations": 0, "chunks": 0}

    data = json.loads(path.read_text(encoding="utf-8"))
    brain = Brain()

    # Handle both flat message arrays and arrays of conversations
    if isinstance(data, list) and data and isinstance(data[0], dict):
        if "role" in data[0] and "content" in data[0]:
            # Flat array of messages — treat as one conversation
            messages = [m for m in data if m.get("content", "").strip()]
            chunks = brain.store_conversation(messages, provider=provider)
            return {"conversations": 1, "chunks": chunks}

        # Array of conversations, each with a messages field
        total_convs, total_chunks = 0, 0
        for conv in data:
            msgs = conv.get("messages", [])
            messages = [m for m in msgs if m.get("content", "").strip()]
            if messages:
                chunks = brain.store_conversation(
                    messages, provider=provider,
                    conversation_id=conv.get("id"),
                    metadata={"title": conv.get("title", ""), "source": "json_import"},
                )
                total_convs += 1
                total_chunks += chunks
        return {"conversations": total_convs, "chunks": total_chunks}

    return {"error": "Expected a JSON array of messages or conversations", "conversations": 0, "chunks": 0}


def main():
    if len(sys.argv) < 2:
        print("Usage: watty-import-json <path-to-file.json>", file=sys.stderr)
        sys.exit(1)
    provider = sys.argv[2] if len(sys.argv) > 2 else "import"
    result = import_json(sys.argv[1], provider=provider)
    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)
    print(f"Imported {result['conversations']} conversations ({result['chunks']} chunks)")


if __name__ == "__main__":
    main()
