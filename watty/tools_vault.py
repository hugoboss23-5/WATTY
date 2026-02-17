"""
Watty Vault Tools — MCP Interface
===================================
One tool: watty_vault(action=...)
Actions: init, unlock, lock, store, get, list, delete, change_password, status

Zero-knowledge encrypted secret storage.
February 2026
"""

from mcp.types import Tool, TextContent

from watty.vault import Vault


# ── Singleton Vault ────────────────────────────────────────

_vault = Vault()

VAULT_ACTIONS = [
    "init", "unlock", "lock",
    "store", "get", "list", "delete",
    "change_password", "status",
]


# ── Tool Definition ────────────────────────────────────────

TOOLS = [
    Tool(
        name="watty_vault",
        description=(
            "Watty's encrypted vault. Zero-knowledge secret storage.\n"
            "AES-256-GCM encryption, Scrypt key derivation, per-entry unique salts.\n"
            "Even labels are encrypted. No plaintext anywhere.\n\n"
            "Actions:\n"
            "  init — Set master password (first time only)\n"
            "  unlock — Unlock vault with master password\n"
            "  lock — Lock vault (wipe session key)\n"
            "  store — Store a secret (label + value + optional category)\n"
            "  get — Retrieve a secret by label\n"
            "  list — List all secret labels (not values)\n"
            "  delete — Delete a secret by label\n"
            "  change_password — Re-encrypt everything with new password\n"
            "  status — Check vault state (initialized? locked?)\n\n"
            "IMPORTANT: The master password is NEVER stored. Lose it = lose everything.\n"
            "IMPORTANT: Always ask the user for the password — never guess or remember it."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": VAULT_ACTIONS,
                    "description": "Action to perform",
                },
                "password": {
                    "type": "string",
                    "description": "init/unlock/change_password: Master password",
                },
                "new_password": {
                    "type": "string",
                    "description": "change_password: New master password",
                },
                "label": {
                    "type": "string",
                    "description": "store/get/delete: Secret label (e.g. 'github_token')",
                },
                "secret": {
                    "type": "string",
                    "description": "store: The secret value to encrypt",
                },
                "category": {
                    "type": "string",
                    "description": "store: Category (default: 'general'). E.g. 'api_key', 'password', 'token', 'crypto'",
                },
            },
            "required": ["action"],
        },
    ),
]


# ── Handler ────────────────────────────────────────────────

async def handle_vault(arguments: dict) -> list[TextContent]:
    global _vault
    action = arguments.get("action", "")

    if action == "status":
        return [TextContent(type="text", text=(
            f"Vault Status:\n"
            f"  Initialized: {_vault.is_initialized}\n"
            f"  Unlocked: {_vault.is_unlocked}\n"
            f"  Database: {_vault.db_path}"
        ))]

    elif action == "init":
        password = arguments.get("password")
        if not password:
            return [TextContent(type="text", text="Need a master password. Ask the user.")]
        result = _vault.initialize(password)
        if "error" in result:
            return [TextContent(type="text", text=f"Error: {result['error']}")]
        return [TextContent(type="text", text=result["status"])]

    elif action == "unlock":
        password = arguments.get("password")
        if not password:
            return [TextContent(type="text", text="Need the master password. Ask the user.")]
        result = _vault.unlock(password)
        if "error" in result:
            return [TextContent(type="text", text=f"Error: {result['error']}")]
        return [TextContent(type="text", text=result["status"])]

    elif action == "lock":
        result = _vault.lock()
        return [TextContent(type="text", text=result["status"])]

    elif action == "store":
        label = arguments.get("label")
        secret = arguments.get("secret")
        category = arguments.get("category", "general")
        if not label or not secret:
            return [TextContent(type="text", text="Need both 'label' and 'secret'.")]
        result = _vault.store(label, secret, category=category)
        if "error" in result:
            return [TextContent(type="text", text=f"Error: {result['error']}")]
        return [TextContent(type="text", text=f"Stored. Label encrypted. Category: {result['category']}")]

    elif action == "get":
        label = arguments.get("label")
        if not label:
            return [TextContent(type="text", text="Need 'label' to retrieve.")]
        result = _vault.retrieve(label)
        if "error" in result:
            return [TextContent(type="text", text=f"Error: {result['error']}")]
        return [TextContent(type="text", text=(
            f"Secret found:\n"
            f"  Label: {result['label']}\n"
            f"  Value: {result['secret']}\n"
            f"  Category: {result['category']}\n"
            f"  Created: {result['created_at']}"
        ))]

    elif action == "list":
        result = _vault.list_secrets()
        if "error" in result:
            return [TextContent(type="text", text=f"Error: {result['error']}")]
        if result["count"] == 0:
            return [TextContent(type="text", text="Vault is empty.")]
        lines = [f"Vault: {result['count']} secret(s)\n"]
        for e in result["entries"]:
            lines.append(f"  [{e['id']}] {e['label']} ({e['category']}) — {e.get('created_at', '?')}")
        return [TextContent(type="text", text="\n".join(lines))]

    elif action == "delete":
        label = arguments.get("label")
        if not label:
            return [TextContent(type="text", text="Need 'label' to delete.")]
        result = _vault.delete(label)
        if "error" in result:
            return [TextContent(type="text", text=f"Error: {result['error']}")]
        return [TextContent(type="text", text=result["status"])]

    elif action == "change_password":
        old_pw = arguments.get("password")
        new_pw = arguments.get("new_password")
        if not old_pw or not new_pw:
            return [TextContent(type="text", text="Need both 'password' (old) and 'new_password'.")]
        result = _vault.change_password(old_pw, new_pw)
        if "error" in result:
            return [TextContent(type="text", text=f"Error: {result['error']}")]
        return [TextContent(type="text", text=result["status"])]

    else:
        return [TextContent(type="text", text=f"Unknown vault action: {action}. Valid: {', '.join(VAULT_ACTIONS)}")]


HANDLERS = {"watty_vault": handle_vault}
