"""
Import Claude conversation exports into Watty.
Usage: watty-import-claude ~/Downloads/claude-export.json
"""

import json
import sys
import zipfile
from pathlib import Path

from watty.brain import Brain


def import_claude(path: str) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return {"error": f"File not found: {path}", "conversations": 0, "chunks": 0}

    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            candidates = [n for n in zf.namelist() if n.endswith(".json")]
            if not candidates:
                return {"error": "No JSON files found in ZIP", "conversations": 0, "chunks": 0}
            data = json.loads(zf.read(candidates[0]))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        data = [data]

    brain = Brain()
    total_convs, total_chunks = 0, 0

    for conv in data:
        messages = []
        for msg in conv.get("chat_messages", conv.get("messages", [])):
            role = msg.get("sender", msg.get("role", "unknown"))
            content = msg.get("text", msg.get("content", ""))
            if isinstance(content, list):
                content = " ".join(str(p) for p in content if isinstance(p, str))
            if role == "system" or not content.strip():
                continue
            messages.append({"role": role, "content": content})

        if messages:
            chunks = brain.store_conversation(
                messages, provider="claude",
                conversation_id=conv.get("uuid", conv.get("id")),
                metadata={"title": conv.get("name", conv.get("title", "")), "source": "claude_export"},
            )
            total_convs += 1
            total_chunks += chunks

    return {"conversations": total_convs, "chunks": total_chunks}


def main():
    if len(sys.argv) < 2:
        print("Usage: watty-import-claude <path-to-export.json>", file=sys.stderr)
        sys.exit(1)
    result = import_claude(sys.argv[1])
    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)
    print(f"Imported {result['conversations']} conversations ({result['chunks']} chunks)")


if __name__ == "__main__":
    main()
