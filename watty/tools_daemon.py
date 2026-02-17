"""
Watty Daemon MCP Tools
======================
Let Claude check daemon status, queue tasks, read insights, and control autonomy.
"""

import json
from mcp.types import Tool, TextContent

from watty.daemon import (
    daemon_status, daemon_activity, daemon_insights,
    daemon_queue_task, daemon_config, daemon_update_config,
    daemon_stop,
)


TOOLS = [
    Tool(
        name="watty_daemon",
        description=(
            "Watty's autonomous daemon. Runs 24/7 in the background.\n"
            "Actions:\n"
            "  status   — Is the daemon running? Uptime, last heartbeat\n"
            "  activity — Recent daemon actions (scans, dreams, insights)\n"
            "  insights — Proactively surfaced knowledge connections\n"
            "  queue    — Add a task for the daemon to execute\n"
            "  config   — View or update daemon schedule/settings\n"
            "  stop     — Stop the daemon"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "activity", "insights", "queue", "config", "stop"],
                    "description": "Action to perform",
                },
                "n": {
                    "type": "integer",
                    "description": "activity/insights: number of entries (default 20/10)",
                },
                "task_type": {
                    "type": "string",
                    "description": "queue: task type (brain, shell, gpu)",
                },
                "task_action": {
                    "type": "string",
                    "description": "queue: action to perform",
                },
                "task_params": {
                    "type": "object",
                    "description": "queue: parameters for the task",
                },
                "config_updates": {
                    "type": "object",
                    "description": "config: JSON object of settings to update",
                },
            },
            "required": ["action"],
        },
    ),
]


async def handle_daemon(args: dict) -> list[TextContent]:
    action = args.get("action", "")

    if action == "status":
        status = daemon_status()
        running = "RUNNING" if status.get("running") else "STOPPED"
        lines = [f"Daemon: {running}"]
        if status.get("pid"):
            lines.append(f"PID: {status['pid']}")
        if status.get("started_at"):
            lines.append(f"Started: {status['started_at']}")
        if status.get("last_heartbeat"):
            lines.append(f"Heartbeat: {status['last_heartbeat']}")
        return [TextContent(type="text", text="\n".join(lines))]

    elif action == "activity":
        n = args.get("n", 20)
        entries = daemon_activity(n)
        if not entries:
            return [TextContent(type="text", text="No daemon activity yet. Start with: watty daemon start")]
        lines = []
        for e in entries:
            t = e.get("time_local", "?")
            act = e.get("action", "?")
            detail = e.get("detail", "")
            line = f"{t} | {act}"
            if detail:
                line += f" | {detail[:80]}"
            lines.append(line)
        return [TextContent(type="text", text="-- Daemon Activity --\n" + "\n".join(lines))]

    elif action == "insights":
        n = args.get("n", 10)
        insights = daemon_insights(n)
        if not insights:
            return [TextContent(type="text", text="No insights yet. Daemon surfaces them every 2 hours.")]
        lines = []
        for i, ins in enumerate(insights):
            lines.append(f"[{i+1}] {ins.get('timestamp', '?')}")
            lines.append(f"    {ins.get('content', '?')[:200]}")
        return [TextContent(type="text", text="-- Insights --\n" + "\n".join(lines))]

    elif action == "queue":
        task_type = args.get("task_type", "")
        task_action = args.get("task_action", "")
        if not task_type or not task_action:
            return [TextContent(type="text", text="Missing task_type and task_action. Example: task_type='brain', task_action='dream'")]
        params = args.get("task_params", {})
        task_id = daemon_queue_task(task_type, task_action, params)
        return [TextContent(type="text", text=f"Task queued: {task_id} ({task_type}:{task_action})")]

    elif action == "config":
        updates = args.get("config_updates")
        if updates:
            config = daemon_update_config(updates)
            return [TextContent(type="text", text=f"Config updated:\n{json.dumps(config, indent=2)}")]
        else:
            config = daemon_config()
            return [TextContent(type="text", text=f"-- Daemon Config --\n{json.dumps(config, indent=2)}")]

    elif action == "stop":
        result = daemon_stop()
        if result.get("success"):
            return [TextContent(type="text", text=f"Daemon stopped (PID {result['pid']})")]
        return [TextContent(type="text", text=f"Stop failed: {result.get('error', '?')}")]

    return [TextContent(type="text", text=f"Unknown daemon action: {action}")]


HANDLERS = {
    "watty_daemon": handle_daemon,
}
