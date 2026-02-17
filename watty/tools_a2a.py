"""
Watty A2A Protocol — MCP Tool Interface
=========================================
One tool: watty_a2a(action=...)
Actions: my_card, discover_agent, list_agents, send_task, task_status, list_tasks, set_auth_token

Agent-to-agent communication protocol. Publish Agent Cards,
accept tasks from external agents, discover and delegate to other agents.

Hugo & Watty · February 2026
"""

import json

from mcp.types import Tool, TextContent


# ── Actions ──────────────────────────────────────────────────

A2A_ACTIONS = [
    "my_card", "discover_agent", "list_agents",
    "send_task", "task_status", "list_tasks",
    "set_auth_token",
]


# ── Brain Reference ──────────────────────────────────────────

_brain_ref = None


def set_brain(brain):
    """Called by server.py to inject brain reference."""
    global _brain_ref
    _brain_ref = brain


# ── Tool Definition ──────────────────────────────────────────

TOOLS = [
    Tool(
        name="watty_a2a",
        description=(
            "Watty's A2A (Agent-to-Agent) protocol engine.\n"
            "Publish your Agent Card, discover remote agents, "
            "and delegate tasks across the agent network.\n\n"
            "Actions:\n"
            "  my_card — Show this agent's Agent Card (capabilities + skills)\n"
            "  discover_agent — Discover a remote agent by URL\n"
            "  list_agents — List all discovered agents\n"
            "  send_task — Send a task to a remote agent\n"
            "  task_status — Check status of a task by ID\n"
            "  list_tasks — List tasks with optional filters\n"
            "  set_auth_token — Set bearer auth token (in-memory only)\n\n"
            "Use discover_agent before send_task to verify the remote agent's skills."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": A2A_ACTIONS,
                    "description": "Action to perform",
                },
                "agent_url": {
                    "type": "string",
                    "description": "discover_agent/send_task: Remote agent URL (e.g. http://host:port)",
                },
                "skill_id": {
                    "type": "string",
                    "description": "send_task: Skill/tool ID to invoke on the remote agent",
                },
                "input_data": {
                    "type": "object",
                    "description": "send_task: Input data to pass to the remote skill",
                },
                "task_id": {
                    "type": "string",
                    "description": "task_status: Task ID to check",
                },
                "token": {
                    "type": "string",
                    "description": "set_auth_token: Bearer token for authenticated A2A requests",
                },
                "status": {
                    "type": "string",
                    "enum": ["pending", "running", "completed", "failed", "cancelled"],
                    "description": "list_tasks: Filter by status",
                },
                "direction": {
                    "type": "string",
                    "enum": ["incoming", "outgoing"],
                    "description": "list_tasks: Filter by direction",
                },
                "limit": {
                    "type": "integer",
                    "description": "list_tasks: Max results (default: 50)",
                },
            },
            "required": ["action"],
        },
    ),
]


# ── Handler ──────────────────────────────────────────────────

def _get_engine():
    """Get the A2AEngine from the brain, or create a standalone one."""
    if _brain_ref is not None and hasattr(_brain_ref, "_a2a") and _brain_ref._a2a is not None:
        return _brain_ref._a2a
    # Fallback: create standalone engine using brain's db_path
    from watty.a2a import A2AEngine
    if _brain_ref is not None:
        return A2AEngine(db_path=_brain_ref.db_path)
    return A2AEngine()


async def handle_a2a(arguments: dict) -> list[TextContent]:
    """Dispatch A2A protocol actions."""
    action = arguments.get("action", "")

    engine = _get_engine()
    if engine is None:
        return [TextContent(type="text", text="A2A Protocol not available.")]

    # ── my_card ──────────────────────────────────────────
    if action == "my_card":
        # Try to get the tools list from the brain's server context
        tools_list = None
        if _brain_ref is not None and hasattr(_brain_ref, "_all_tools"):
            tools_list = _brain_ref._all_tools

        card = engine.generate_agent_card(tools_list=tools_list)
        card_json = json.dumps(card, indent=2)
        skill_count = len(card.get("skills", []))

        return [TextContent(
            type="text",
            text=(
                f"Agent Card for {card['name']} v{card['version']}:\n"
                f"  Skills: {skill_count}\n"
                f"  Protocol: {card.get('protocol', 'a2a')} v{card.get('protocolVersion', '?')}\n"
                f"  Input modes: {', '.join(card.get('defaultInputModes', []))}\n"
                f"  Output modes: {', '.join(card.get('defaultOutputModes', []))}\n"
                f"  Auth: {'bearer token' if card.get('authentication', {}).get('schemes') else 'none'}\n\n"
                f"Full card JSON:\n{card_json}"
            ),
        )]

    # ── discover_agent ───────────────────────────────────
    elif action == "discover_agent":
        agent_url = arguments.get("agent_url")
        if not agent_url:
            return [TextContent(type="text", text="Need 'agent_url' to discover a remote agent.")]

        try:
            card = engine.discover_agent(agent_url)
        except Exception as e:
            return [TextContent(type="text", text=f"Discovery error: {e}")]

        if "error" in card:
            return [TextContent(type="text", text=f"Discovery failed: {card['error']}")]

        skills = card.get("skills", [])
        skill_names = [s.get("id", s.get("name", "?")) for s in skills]

        lines = [
            f"Discovered agent: {card.get('name', 'unknown')}",
            f"  URL: {agent_url}",
            f"  Version: {card.get('version', '?')}",
            f"  Description: {card.get('description', 'none')[:200]}",
            f"  Skills ({len(skills)}):",
        ]
        for s in skills[:20]:
            sid = s.get("id", s.get("name", "?"))
            sdesc = s.get("description", "")[:80]
            lines.append(f"    - {sid}: {sdesc}")
        if len(skills) > 20:
            lines.append(f"    ... and {len(skills) - 20} more")

        return [TextContent(type="text", text="\n".join(lines))]

    # ── list_agents ──────────────────────────────────────
    elif action == "list_agents":
        agents = engine.list_agents()

        if not agents:
            return [TextContent(type="text", text="No agents discovered yet. Use discover_agent first.")]

        lines = [f"Discovered agents ({len(agents)}):\n"]
        for url, info in agents.items():
            card = info.get("card", {})
            name = card.get("name", "unknown")
            version = card.get("version", "?")
            skill_count = len(card.get("skills", []))
            discovered = info.get("discovered_at", "?")[:19]
            lines.append(f"  [{name} v{version}] {url}")
            lines.append(f"    Skills: {skill_count} | Discovered: {discovered}")

        return [TextContent(type="text", text="\n".join(lines))]

    # ── send_task ────────────────────────────────────────
    elif action == "send_task":
        agent_url = arguments.get("agent_url")
        skill_id = arguments.get("skill_id")
        input_data = arguments.get("input_data", {})

        if not agent_url:
            return [TextContent(type="text", text="Need 'agent_url' to send a task.")]
        if not skill_id:
            return [TextContent(type="text", text="Need 'skill_id' to send a task.")]

        try:
            result = engine.delegate_task(
                agent_url=agent_url,
                skill_id=skill_id,
                input_data=input_data,
            )
        except Exception as e:
            return [TextContent(type="text", text=f"Task delegation error: {e}")]

        if "error" in result:
            return [TextContent(
                type="text",
                text=(
                    f"Task failed: {result['error']}\n"
                    f"  Local task ID: {result.get('task_id', 'n/a')}"
                ),
            )]

        status = result.get("status", "unknown")
        remote_result = result.get("result", {})
        output_preview = ""
        if isinstance(remote_result, dict):
            output = remote_result.get("output", remote_result.get("result", ""))
            if isinstance(output, str):
                output_preview = output[:500]
            else:
                output_preview = json.dumps(output, indent=2)[:500]

        return [TextContent(
            type="text",
            text=(
                f"Task delegated to {agent_url}:\n"
                f"  Local task ID: {result.get('task_id', 'n/a')}\n"
                f"  Remote task ID: {result.get('remote_task_id', 'n/a')}\n"
                f"  Status: {status}\n"
                + (f"  Output: {output_preview}" if output_preview else "")
            ),
        )]

    # ── task_status ──────────────────────────────────────
    elif action == "task_status":
        task_id = arguments.get("task_id")
        if not task_id:
            return [TextContent(type="text", text="Need 'task_id' to check status.")]

        try:
            task = engine.get_task(task_id)
        except Exception as e:
            return [TextContent(type="text", text=f"Error fetching task: {e}")]

        if not task:
            return [TextContent(type="text", text=f"Task {task_id} not found.")]

        lines = [
            f"Task: {task['id']}",
            f"  Direction: {task['direction']}",
            f"  Skill: {task.get('skill_id', 'n/a')}",
            f"  Status: {task['status']}",
            f"  Agent: {task.get('agent_url', 'n/a')}",
            f"  Created: {task['created_at'][:19]}",
            f"  Updated: {task['updated_at'][:19]}",
        ]
        if task.get("completed_at"):
            lines.append(f"  Completed: {task['completed_at'][:19]}")
        if task.get("error"):
            lines.append(f"  Error: {task['error']}")
        if task.get("output_data"):
            output = task["output_data"]
            if isinstance(output, dict):
                preview = json.dumps(output, indent=2)[:300]
            else:
                preview = str(output)[:300]
            lines.append(f"  Output: {preview}")

        return [TextContent(type="text", text="\n".join(lines))]

    # ── list_tasks ───────────────────────────────────────
    elif action == "list_tasks":
        try:
            tasks = engine.list_tasks(
                status=arguments.get("status"),
                direction=arguments.get("direction"),
                limit=arguments.get("limit", 50),
            )
        except Exception as e:
            return [TextContent(type="text", text=f"Error listing tasks: {e}")]

        if not tasks:
            return [TextContent(type="text", text="No tasks found.")]

        lines = [f"A2A Tasks ({len(tasks)}):\n"]
        for t in tasks:
            error_flag = " [ERROR]" if t.get("error") else ""
            lines.append(
                f"  [{t['id'][:8]}...] {t['direction']} | "
                f"{t.get('skill_id', 'n/a')} | {t['status']}{error_flag} | "
                f"{t['created_at'][:19]}"
            )
            if t.get("agent_url"):
                lines.append(f"    Agent: {t['agent_url']}")

        return [TextContent(type="text", text="\n".join(lines))]

    # ── set_auth_token ───────────────────────────────────
    elif action == "set_auth_token":
        token = arguments.get("token")
        if not token:
            return [TextContent(type="text", text="Need 'token' to set auth.")]

        engine.set_auth_token(token)
        masked = engine.get_auth_token()
        return [TextContent(
            type="text",
            text=f"Auth token set: {masked}\nStored in memory only (not persisted to disk).",
        )]

    # ── Unknown action ───────────────────────────────────
    else:
        return [TextContent(
            type="text",
            text=f"Unknown a2a action: {action}. Valid: {', '.join(A2A_ACTIONS)}",
        )]


# ── Router ───────────────────────────────────────────────────

HANDLERS = {"watty_a2a": handle_a2a}
