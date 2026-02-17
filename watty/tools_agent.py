"""
tools_agent.py — OpenClaw Full Control Gateway
================================================
Two modes of control over clawdbot:

1. DIRECT MODE (action: "tool") — Call any of clawdbot's 17 tools instantly.
   No model involved. No latency. No interpretation errors. FULL CONTROL.

2. AGENT MODE (action: "send/dispatch/fanout") — Spawn sub-agent sessions
   for complex multi-step tasks that need reasoning + tool chaining.

Direct tools available:
  message        — WhatsApp send/react/poll/delete/pin/effects
  browser        — open/snapshot/screenshot/actions/tabs/status
  nodes          — camera_snap/camera_clip/screen/location/run/status
  canvas         — present/hide/navigate/eval/snapshot/A2UI
  cron           — add/list/remove/update/run/status
  tts            — text to speech
  web_search     — Brave search (query, count, country)
  web_fetch      — URL to markdown
  memory_search  — semantic search clawdbot memory
  memory_get     — read memory snippets
  sessions_list  — list sessions
  sessions_spawn — spawn sub-agent
  sessions_send  — send to session
  sessions_history — session history
  agents_list    — list agents
  gateway        — restart/config
  session_status — session status card

MCP tool name: watty_agent
"""

import json
import os
import time
import threading
import uuid
import requests

from mcp.types import Tool, TextContent

# ─── CONFIG ──────────────────────────────────────────────────────────────────

OPENCLAW_BASE = "http://127.0.0.1:18789"
OPENCLAW_TOKEN = "aef1b808872fa265c343d9aefd17f273c975ff60130f346c"
HUGO_JID = "17076318006@s.whatsapp.net"
DEFAULT_TIMEOUT = 120
POLL_INTERVAL = 2.0
MAX_POLLS = 60
MAX_CONCURRENT = 8
TASK_TTL_HOURS = 72
TASKS_FILE = os.path.expanduser("~/.watty/agent_tasks.json")

# All tools callable via direct invocation
DIRECT_TOOLS = [
    "message", "browser", "nodes", "canvas", "cron", "tts",
    "web_search", "web_fetch", "memory_search", "memory_get",
    "sessions_list", "sessions_spawn", "sessions_send",
    "sessions_history", "agents_list", "gateway", "session_status",
]

# ─── TEMPLATES (for agent mode) ──────────────────────────────────────────────

TEMPLATES = {
    "research": {
        "preamble": (
            "You are a research agent. Search the web and return structured findings.\n"
            "Use web_search and web_fetch. Return JSON: {\"summary\": str, "
            "\"findings\": [{\"title\": str, \"detail\": str, \"source\": str}], "
            "\"confidence\": 0-1}. Be thorough. No opinions.\n\nTASK: "
        ),
        "timeout": 180,
    },
    "browse": {
        "preamble": (
            "You are a browser agent. Use the browser tool (actions: open, snapshot, "
            "screenshot, actions, tabs, status; profile='chrome' or 'clawd').\n"
            "Return JSON: {\"url\": str, \"data\": any, \"screenshots\": [str], "
            "\"success\": bool}. Execute precisely.\n\nTASK: "
        ),
        "timeout": 180,
    },
    "code": {
        "preamble": (
            "Execute Python code and return the output.\n"
            "Return JSON: {\"code\": str, \"output\": str, \"exit_code\": int, "
            "\"error\": str|null}. No explanations.\n\nTASK: "
        ),
        "timeout": 120,
    },
    "raw": {"preamble": "", "timeout": 120},
}

# ─── TASK PERSISTENCE ────────────────────────────────────────────────────────

_tasks_lock = threading.Lock()
_active_threads = 0
_active_threads_lock = threading.Lock()


def _load_tasks_unlocked() -> dict:
    try:
        with open(TASKS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_tasks_unlocked(tasks: dict):
    os.makedirs(os.path.dirname(TASKS_FILE), exist_ok=True)
    with open(TASKS_FILE, "w") as f:
        json.dump(tasks, f, indent=2)


def _load_tasks() -> dict:
    with _tasks_lock:
        return _load_tasks_unlocked()


def _update_task(task_id: str, updates: dict):
    with _tasks_lock:
        tasks = _load_tasks_unlocked()
        if task_id in tasks:
            tasks[task_id].update(updates)
            _save_tasks_unlocked(tasks)


def _cleanup_old_tasks():
    with _tasks_lock:
        tasks = _load_tasks_unlocked()
        cutoff = time.time() - (TASK_TTL_HOURS * 3600)
        cleaned = {k: v for k, v in tasks.items() if v.get("created", 0) > cutoff}
        if len(cleaned) < len(tasks):
            _save_tasks_unlocked(cleaned)


# ─── HTTP LAYER ──────────────────────────────────────────────────────────────

def _headers():
    return {
        "Authorization": f"Bearer {OPENCLAW_TOKEN}",
        "Content-Type": "application/json",
    }


def _invoke_tool(tool: str, args: dict, timeout: int = 15) -> dict:
    """Call an OpenClaw tool directly via /tools/invoke. No model involved."""
    r = requests.post(
        f"{OPENCLAW_BASE}/tools/invoke",
        headers=_headers(),
        json={"tool": tool, "args": args},
        timeout=timeout,
    )
    data = r.json()
    if not data.get("ok"):
        error = data.get("error", {})
        msg = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        return {"error": msg}
    result = data.get("result", {})
    details = result.get("details")
    if details:
        return details
    content = result.get("content", [])
    if content and content[0].get("text"):
        try:
            return json.loads(content[0]["text"])
        except (json.JSONDecodeError, KeyError):
            return {"raw": content[0]["text"]}
    return result


def _poll_for_response(session_key: str, after_ts: int = 0,
                       timeout: float = DEFAULT_TIMEOUT) -> dict:
    """Poll until an assistant message WITH TEXT appears (skips tool-call-only messages)."""
    deadline = time.time() + timeout
    polls = 0

    while time.time() < deadline and polls < MAX_POLLS:
        time.sleep(POLL_INTERVAL)
        polls += 1

        try:
            result = _invoke_tool("sessions_history", {
                "sessionKey": session_key,
                "limit": 10,
            })

            messages = result.get("messages", [])
            for msg in reversed(messages):
                if msg.get("role") != "assistant":
                    continue
                if msg.get("timestamp", 0) <= after_ts:
                    continue

                content_parts = msg.get("content", [])
                text_parts = []
                for part in content_parts:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)

                text = "\n".join(text_parts).strip()
                # Skip messages that are only tool calls (no text)
                if not text:
                    continue

                return {
                    "response": text,
                    "model": msg.get("model", "unknown"),
                    "provider": msg.get("provider", "unknown"),
                    "usage": msg.get("usage"),
                }
        except Exception:
            pass

    return {"error": f"No text response after {timeout}s ({polls} polls)"}


def _spawn_and_poll(prompt: str, session: str = None,
                    timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Spawn or send to session, poll for TEXT response."""
    now_ts = int(time.time() * 1000)
    start = time.time()

    if session:
        result = _invoke_tool("sessions_send", {
            "sessionKey": session,
            "message": prompt,
        }, timeout=30)
    else:
        result = _invoke_tool("sessions_spawn", {
            "task": prompt,
        }, timeout=30)

    if "error" in result:
        return result

    run_id = result.get("runId", "")
    child_key = result.get("childSessionKey", session or "")

    poll_result = _poll_for_response(child_key, after_ts=now_ts, timeout=timeout)
    elapsed = time.time() - start

    return {
        **poll_result,
        "session": child_key,
        "run_id": run_id,
        "time_seconds": round(elapsed, 2),
    }


def _auto_store(brain, tag: str, prompt: str, response: str):
    if not brain:
        return
    try:
        content = f"[openclaw:{tag}] {prompt}\n\nResult:\n{response[:2000]}"
        brain.store_memory(content, provider="openclaw")
    except Exception:
        pass


# ─── BACKGROUND TASK RUNNER ──────────────────────────────────────────────────

def _run_background_task(task_id: str, prompt: str, template: str,
                         session: str, timeout: int, brain, store_result: bool):
    global _active_threads

    _update_task(task_id, {"status": "running", "started": time.time()})

    try:
        t = TEMPLATES.get(template, TEMPLATES["raw"])
        preamble = t["preamble"]
        full_prompt = (preamble + prompt) if preamble else prompt

        result = _spawn_and_poll(full_prompt, session=session, timeout=timeout)

        _update_task(task_id, {
            "status": "completed" if "error" not in result else "failed",
            "result": result,
            "finished": time.time(),
        })

        if store_result and "error" not in result:
            _auto_store(brain, template, prompt, result.get("response", ""))

    except Exception as e:
        _update_task(task_id, {
            "status": "failed",
            "result": {"error": str(e)},
            "finished": time.time(),
        })
    finally:
        with _active_threads_lock:
            _active_threads -= 1


def _dispatch_task(prompt: str, template: str = "raw", session: str = None,
                   timeout: int = None, brain=None,
                   store_result: bool = True) -> dict:
    global _active_threads

    with _active_threads_lock:
        if _active_threads >= MAX_CONCURRENT:
            return {"error": f"Max concurrent tasks ({MAX_CONCURRENT}) reached."}
        _active_threads += 1

    t = TEMPLATES.get(template, TEMPLATES["raw"])
    if timeout is None:
        timeout = t["timeout"]

    task_id = uuid.uuid4().hex[:12]
    task = {
        "task_id": task_id,
        "prompt": prompt[:200],
        "template": template,
        "status": "pending",
        "created": time.time(),
        "session": session,
    }

    with _tasks_lock:
        tasks = _load_tasks_unlocked()
        tasks[task_id] = task
        _save_tasks_unlocked(tasks)

    thread = threading.Thread(
        target=_run_background_task,
        args=(task_id, prompt, template, session, timeout, brain, store_result),
        daemon=True,
        name=f"agent-task-{task_id}",
    )
    thread.start()

    return {"task_id": task_id, "status": "dispatched", "template": template}


# ─── ACTIONS ─────────────────────────────────────────────────────────────────

def action_tool(params: dict, brain=None) -> str:
    """
    DIRECT TOOL INVOCATION — Call any clawdbot tool instantly. No model.

    params:
      tool_name (str): One of the 17 gateway tools
      tool_args (dict): Arguments to pass to the tool
      store_result (bool): Auto-store in brain (default: false for direct calls)
    """
    tool_name = params.get("tool_name", "")
    if not tool_name:
        return json.dumps({"error": "tool_name is required", "available": DIRECT_TOOLS})

    if tool_name not in DIRECT_TOOLS:
        return json.dumps({"error": f"Unknown tool: {tool_name}", "available": DIRECT_TOOLS})

    tool_args = params.get("tool_args", {})
    timeout = params.get("timeout", 30)

    try:
        result = _invoke_tool(tool_name, tool_args, timeout=timeout)

        if params.get("store_result") and brain:
            _auto_store(brain, f"direct:{tool_name}",
                        json.dumps(tool_args)[:200],
                        json.dumps(result)[:2000])

        return json.dumps(result)

    except requests.ConnectionError:
        return json.dumps({"error": "OpenClaw gateway not running"})
    except requests.Timeout:
        return json.dumps({"error": f"Tool {tool_name} timed out after {timeout}s"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def action_tools(params: dict, brain=None) -> str:
    """List all available direct tools on the gateway."""
    return json.dumps({
        "tools": DIRECT_TOOLS,
        "count": len(DIRECT_TOOLS),
        "usage": "action='tool', tool_name='<name>', tool_args={...}",
        "shortcuts": {
            "whatsapp_send": {"tool_name": "message", "tool_args": {
                "action": "send", "target": HUGO_JID, "message": "<text>"}},
            "web_search": {"tool_name": "web_search", "tool_args": {
                "query": "<search query>"}},
            "web_fetch": {"tool_name": "web_fetch", "tool_args": {
                "url": "<url>"}},
            "browser_open": {"tool_name": "browser", "tool_args": {
                "action": "open", "url": "<url>"}},
            "browser_screenshot": {"tool_name": "browser", "tool_args": {
                "action": "screenshot"}},
            "camera_snap": {"tool_name": "nodes", "tool_args": {
                "action": "camera_snap", "camera": "back"}},
            "screen_capture": {"tool_name": "nodes", "tool_args": {
                "action": "screen"}},
            "location": {"tool_name": "nodes", "tool_args": {
                "action": "location"}},
            "tts": {"tool_name": "tts", "tool_args": {
                "text": "<text to speak>"}},
            "cron_list": {"tool_name": "cron", "tool_args": {
                "action": "list"}},
            "cron_add": {"tool_name": "cron", "tool_args": {
                "action": "add", "schedule": "<cron expr>", "task": "<prompt>"}},
        },
    })


def action_dispatch(params: dict, brain=None) -> str:
    """Fire a background task, get task_id immediately (non-blocking)."""
    prompt = params.get("prompt", "")
    if not prompt:
        return json.dumps({"error": "prompt is required"})

    template = params.get("template", "raw")
    if template not in TEMPLATES:
        return json.dumps({"error": f"Unknown template: {template}", "available": list(TEMPLATES.keys())})

    return json.dumps(_dispatch_task(
        prompt=prompt, template=template,
        session=params.get("session"),
        timeout=params.get("timeout"),
        brain=brain,
        store_result=params.get("store_result", True),
    ))


def action_poll(params: dict, brain=None) -> str:
    """Check status/result of a dispatched task."""
    task_id = params.get("task_id", "")
    if not task_id:
        return json.dumps({"error": "task_id is required"})

    tasks = _load_tasks()
    task = tasks.get(task_id)
    if not task:
        return json.dumps({"error": f"Task {task_id} not found"})

    out = {
        "task_id": task_id,
        "status": task["status"],
        "template": task.get("template", "raw"),
        "prompt": task.get("prompt", ""),
        "created": task.get("created"),
    }

    if task["status"] in ("completed", "failed"):
        out["result"] = task.get("result", {})
        out["finished"] = task.get("finished")
        elapsed = (task.get("finished", 0) or 0) - (task.get("started", 0) or 0)
        out["time_seconds"] = round(elapsed, 2) if elapsed > 0 else None

    return json.dumps(out)


def action_send(params: dict, brain=None) -> str:
    """Synchronous agent task — blocks until clawdbot responds with text."""
    prompt = params.get("prompt", "")
    if not prompt:
        return json.dumps({"error": "prompt is required"})

    template = params.get("template", "raw")
    if template not in TEMPLATES:
        return json.dumps({"error": f"Unknown template: {template}", "available": list(TEMPLATES.keys())})

    t = TEMPLATES[template]
    timeout = params.get("timeout", t["timeout"])
    session = params.get("session")
    store_result = params.get("store_result", True)

    if params.get("use_memory") and brain:
        try:
            results = brain.recall(prompt, top_k=3)
            if results:
                context = "\n".join(
                    f"[Memory {i+1}] {r['content'][:300]}"
                    for i, r in enumerate(results)
                )
                prompt = f"{prompt}\n\nRelevant context from Watty memory:\n{context}"
        except Exception:
            pass

    preamble = t["preamble"]
    full_prompt = (preamble + prompt) if preamble else prompt

    try:
        result = _spawn_and_poll(full_prompt, session=session, timeout=timeout)

        if store_result and "error" not in result:
            _auto_store(brain, template, params.get("prompt", ""), result.get("response", ""))

        return json.dumps(result)

    except requests.ConnectionError:
        return json.dumps({"error": "OpenClaw gateway not running"})
    except requests.Timeout:
        return json.dumps({"error": f"Gateway timed out after {timeout}s"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def action_followup(params: dict, brain=None) -> str:
    """Send follow-up to an existing session (maintains context)."""
    session = params.get("session", "")
    if not session:
        return json.dumps({"error": "session is required"})

    prompt = params.get("prompt", "")
    if not prompt:
        return json.dumps({"error": "prompt is required"})

    timeout = params.get("timeout", DEFAULT_TIMEOUT)

    try:
        return json.dumps(_spawn_and_poll(prompt, session=session, timeout=timeout))
    except requests.ConnectionError:
        return json.dumps({"error": "OpenClaw gateway not running"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def action_fanout(params: dict, brain=None) -> str:
    """Dispatch up to 8 tasks in parallel."""
    task_list = params.get("tasks", [])
    if not task_list:
        return json.dumps({"error": "tasks array is required"})

    if len(task_list) > MAX_CONCURRENT:
        return json.dumps({"error": f"Max {MAX_CONCURRENT} parallel tasks"})

    results = []
    for t in task_list:
        prompt = t.get("prompt", "")
        template = t.get("template", "raw")
        if not prompt:
            results.append({"error": "prompt is required in each task"})
            continue
        results.append(_dispatch_task(
            prompt=prompt, template=template,
            session=t.get("session"), timeout=t.get("timeout"),
            brain=brain, store_result=t.get("store_result", True),
        ))

    return json.dumps({
        "dispatched": len([r for r in results if "task_id" in r]),
        "failed": len([r for r in results if "error" in r]),
        "tasks": results,
    })


def action_tasks(params: dict, brain=None) -> str:
    """List all tracked background tasks."""
    _cleanup_old_tasks()
    tasks = _load_tasks()

    status_filter = params.get("status")
    if status_filter:
        tasks = {k: v for k, v in tasks.items() if v.get("status") == status_filter}

    summary = []
    for tid, t in sorted(tasks.items(), key=lambda x: x[1].get("created", 0), reverse=True):
        summary.append({
            "task_id": tid,
            "status": t.get("status", "unknown"),
            "template": t.get("template", "raw"),
            "prompt": t.get("prompt", "")[:100],
            "created": t.get("created"),
            "finished": t.get("finished"),
        })

    return json.dumps({"count": len(summary), "tasks": summary})


def action_status(params: dict, brain=None) -> str:
    """Gateway health + sessions + task stats."""
    try:
        r = requests.get(OPENCLAW_BASE, timeout=5)
        if r.status_code != 200:
            return json.dumps({"status": "error", "http_code": r.status_code})

        sessions = _invoke_tool("sessions_list", {})
        session_count = sessions.get("count", 0)
        session_list = []
        for s in sessions.get("sessions", []):
            session_list.append({
                "key": s.get("key"),
                "model": s.get("model"),
                "tokens": s.get("totalTokens", 0),
            })

        tasks = _load_tasks()
        task_stats = {"total": len(tasks)}
        for st in ("pending", "running", "completed", "failed"):
            task_stats[st] = len([t for t in tasks.values() if t.get("status") == st])
        with _active_threads_lock:
            task_stats["active_threads"] = _active_threads

        return json.dumps({
            "status": "online",
            "gateway": OPENCLAW_BASE,
            "sessions": session_count,
            "session_list": session_list,
            "task_stats": task_stats,
            "direct_tools": len(DIRECT_TOOLS),
            "templates": list(TEMPLATES.keys()),
            "max_concurrent": MAX_CONCURRENT,
        })
    except requests.ConnectionError:
        return json.dumps({"status": "offline", "gateway": OPENCLAW_BASE})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def action_history(params: dict, brain=None) -> str:
    """Get conversation history for a session."""
    session = params.get("session", "")
    if not session:
        return json.dumps({"error": "session key is required"})

    limit = params.get("limit", 20)

    try:
        result = _invoke_tool("sessions_history", {
            "sessionKey": session, "limit": limit,
        })
        if "error" in result:
            return json.dumps(result)

        messages = result.get("messages", [])
        formatted = []
        for msg in messages:
            role = msg.get("role", "?")
            content_parts = msg.get("content", [])
            text = ""
            for part in content_parts:
                if isinstance(part, dict) and part.get("type") == "text":
                    text += part["text"]
                elif isinstance(part, str):
                    text += part
            formatted.append({"role": role, "text": text[:500]})

        return json.dumps({
            "session": session,
            "message_count": len(formatted),
            "messages": formatted,
        })
    except requests.ConnectionError:
        return json.dumps({"error": "OpenClaw gateway not running"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ─── MCP TOOL REGISTRATION ──────────────────────────────────────────────────

TOOLS = [
    Tool(
        name="watty_agent",
        description=(
            "Full control over clawdbot (OpenClaw). Two modes:\n\n"
            "DIRECT MODE (action='tool') — Call any tool instantly, no model:\n"
            "  message (WhatsApp send/react/poll/effects), browser (open/screenshot/actions),\n"
            "  nodes (camera_snap/screen/location), canvas (present/A2UI/snapshot),\n"
            "  cron (add/list/remove/run), tts, web_search, web_fetch,\n"
            "  memory_search, memory_get, sessions_*, agents_list, gateway\n\n"
            "AGENT MODE — Spawn sub-agent for complex multi-step tasks:\n"
            "  send (sync), dispatch (async), fanout (parallel), followup, poll\n\n"
            "OTHER: tools (list all), tasks (background), status (health), history"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["tool", "tools", "send", "dispatch", "poll",
                             "followup", "fanout", "tasks", "status", "history"],
                    "description": (
                        "tool=direct call (instant), tools=list available, "
                        "send=sync agent, dispatch=async agent, poll=check task, "
                        "followup=continue session, fanout=parallel dispatch, "
                        "tasks=list bg tasks, status=health, history=session log"
                    ),
                },
                "tool_name": {
                    "type": "string",
                    "description": (
                        "tool: Which tool to call directly. Options: "
                        "message, browser, nodes, canvas, cron, tts, web_search, "
                        "web_fetch, memory_search, memory_get, sessions_list, "
                        "sessions_spawn, sessions_send, sessions_history, "
                        "agents_list, gateway, session_status"
                    ),
                },
                "tool_args": {
                    "type": "object",
                    "description": (
                        "tool: Arguments for the direct tool call. Examples:\n"
                        "  message: {action:'send', target:'17076318006@s.whatsapp.net', message:'hello'}\n"
                        "  web_search: {query:'bitcoin price'}\n"
                        "  web_fetch: {url:'https://...'}\n"
                        "  browser: {action:'open', url:'https://...'}\n"
                        "  browser: {action:'screenshot'}\n"
                        "  nodes: {action:'camera_snap', camera:'back'}\n"
                        "  nodes: {action:'screen'}\n"
                        "  nodes: {action:'location'}\n"
                        "  cron: {action:'list'}\n"
                        "  cron: {action:'add', schedule:'0 9 * * *', task:'...'}\n"
                        "  tts: {text:'Hello Hugo'}\n"
                        "  canvas: {action:'present', html:'<h1>Hi</h1>'}"
                    ),
                },
                "prompt": {
                    "type": "string",
                    "description": "send/dispatch/fanout/followup: Task for the agent",
                },
                "template": {
                    "type": "string",
                    "enum": ["research", "browse", "code", "raw"],
                    "description": "Agent mode template (default: raw)",
                },
                "session": {
                    "type": "string",
                    "description": "Session key for followup/history",
                },
                "task_id": {
                    "type": "string",
                    "description": "poll: Task ID to check",
                },
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "template": {"type": "string"},
                        },
                        "required": ["prompt"],
                    },
                    "description": "fanout: Array of tasks to dispatch in parallel",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds",
                },
                "use_memory": {
                    "type": "boolean",
                    "description": "send: Augment with Watty memories (default: false)",
                },
                "store_result": {
                    "type": "boolean",
                    "description": "Store result in Watty brain",
                },
                "limit": {
                    "type": "integer",
                    "description": "history: Max messages (default: 20)",
                },
            },
            "required": ["action"],
        },
    ),
]


async def handle_watty_agent(params: dict, brain=None) -> list[TextContent]:
    action = params.get("action", "status")

    dispatch = {
        "tool": action_tool,
        "tools": action_tools,
        "send": action_send,
        "dispatch": action_dispatch,
        "poll": action_poll,
        "followup": action_followup,
        "fanout": action_fanout,
        "tasks": action_tasks,
        "status": action_status,
        "history": action_history,
    }

    handler = dispatch.get(action)
    if handler is None:
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Unknown action: {action}", "available": list(dispatch.keys())}),
        )]

    result = handler(params, brain=brain)
    return [TextContent(type="text", text=result)]


HANDLERS = {
    "watty_agent": handle_watty_agent,
}
