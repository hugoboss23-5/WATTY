"""
Import ChatGPT conversation exports into Watty.
Usage: watty-import-chatgpt ~/Downloads/chatgpt-export.zip
"""

import json
import sys
import zipfile
from pathlib import Path

from watty.brain import Brain


def import_chatgpt(path: str) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return {"error": f"File not found: {path}", "conversations": 0, "chunks": 0}

    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            candidates = [n for n in zf.namelist() if n.endswith(".json") and "conversation" in n.lower()]
            if not candidates:
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
        mapping = conv.get("mapping", {})
        if mapping:
            for node in mapping.values():
                msg = node.get("message")
                if not msg:
                    continue
                role = msg.get("author", {}).get("role", "unknown")
                if role == "system":
                    continue
                parts = msg.get("content", {}).get("parts", [])
                text = " ".join(str(p) for p in parts if isinstance(p, str))
                if text.strip():
                    messages.append({"role": role, "content": text})
        else:
            for msg in conv.get("messages", []):
                role = msg.get("author", {}).get("role", msg.get("role", "unknown"))
                if role == "system":
                    continue
                content = msg.get("content", "")
                if isinstance(content, dict):
                    content = " ".join(str(p) for p in content.get("parts", []) if isinstance(p, str))
                if content.strip():
                    messages.append({"role": role, "content": content})

        if messages:
            chunks = brain.store_conversation(
                messages, provider="chatgpt",
                conversation_id=conv.get("id", conv.get("conversation_id")),
                metadata={"title": conv.get("title", ""), "source": "chatgpt_export"},
            )
            total_convs += 1
            total_chunks += chunks

    return {"conversations": total_convs, "chunks": total_chunks}


def main():
    if len(sys.argv) < 2:
        print("Usage: watty-import-chatgpt <path-to-export.zip>", file=sys.stderr)
        sys.exit(1)
    result = import_chatgpt(sys.argv[1])
    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)
    print(f"Imported {result['conversations']} conversations ({result['chunks']} chunks)")


if __name__ == "__main__":
    main()
