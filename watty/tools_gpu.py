"""
Watty GPU Tools v2
==================
Vast.ai GPU management via CLI + SSH.
Rebuilt February 2026 — actually works now.
"""

import json
import subprocess
from pathlib import Path

from mcp.types import Tool, TextContent



# ── Config ──────────────────────────────────────────────────

GPU_CONFIG_DIR = Path.home() / ".basho_gpu"


def _gpu_api_key() -> str:
    key_file = GPU_CONFIG_DIR / "api_key"
    return key_file.read_text(encoding="utf-8").strip() if key_file.exists() else ""

def _gpu_instance_id() -> str:
    inst_file = GPU_CONFIG_DIR / "instance"
    return inst_file.read_text(encoding="utf-8").strip() if inst_file.exists() else ""

def _save_instance_id(instance_id: str):
    GPU_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (GPU_CONFIG_DIR / "instance").write_text(str(instance_id), encoding="utf-8")

def _ssh_key_path() -> Path:
    # Prefer watty_gpu (no passphrase), fall back to basho_gpu
    watty_key = Path.home() / ".ssh" / "watty_gpu"
    if watty_key.exists():
        return watty_key
    return Path.home() / ".ssh" / "basho_gpu"

def _run(cmd, timeout=30):
    """Run a command and return (stdout, stderr, returncode)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.stderr.strip(), r.returncode
    except subprocess.TimeoutExpired:
        return "", "timeout", -1
    except FileNotFoundError:
        return "", f"command not found: {cmd[0]}", -1
    except Exception as e:
        return "", str(e), -1

def _vastai(*args, raw=False, timeout=30):
    """Run a vastai CLI command. Returns parsed JSON if raw=True, else stdout string."""
    cmd = ["vastai"] + list(args)
    if raw:
        cmd.append("--raw")
    out, err, rc = _run(cmd, timeout=timeout)
    if rc != 0:
        return {"error": err or out or f"exit code {rc}"}
    if raw and out:
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return {"error": f"JSON parse failed: {out[:300]}"}
    return out

def _ssh_cmd(command, timeout=30):
    """Execute a command on the GPU via SSH. Returns (output, error)."""
    iid = _gpu_instance_id()
    if not iid:
        return "", "No instance configured"

    # Get SSH URL from vastai CLI (most reliable source of host:port)
    url_out = _vastai("ssh-url", iid)
    if isinstance(url_out, dict) and "error" in url_out:
        return "", f"Can't get SSH URL: {url_out['error']}"

    # Parse ssh://root@host:port
    url_out = url_out.strip()
    if "://" in url_out:
        url_out = url_out.split("://")[1]
    if "@" in url_out:
        user_host = url_out.split("@")[1]
    else:
        user_host = url_out
    if ":" in user_host:
        host, port = user_host.rsplit(":", 1)
    else:
        host, port = user_host, "22"

    key = _ssh_key_path()
    if not key.exists():
        return "", f"SSH key missing: {key}"

    ssh_args = [
        "ssh", "-T",
        "-i", str(key),
        "-p", port,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=NUL",
        "-o", "ConnectTimeout=10",
        "-o", "IdentitiesOnly=yes",
        "-o", "BatchMode=yes",
        "-o", "LogLevel=ERROR",
        f"root@{host}",
        command
    ]
    out, err, rc = _run(ssh_args, timeout=timeout)
    # Filter out the vast.ai welcome banner
    lines = out.split("\n")
    filtered = [l for l in lines if not l.startswith("Welcome to vast.ai") and not l.startswith("Have fun")]
    return "\n".join(filtered).strip(), err


# ── Tool Definition ─────────────────────────────────────────

GPU_ACTIONS = [
    "status", "start", "stop", "search", "create", "destroy",
    "instances", "logs", "ssh_key",
    "exec", "rest_exec",
    "jupyter_url", "jupyter_exec",
    "credit",
]

TOOLS = [
    Tool(
        name="watty_gpu",
        description=(
            "Vast.ai GPU management. One tool, many actions.\n"
            "Actions: status, start, stop, search, create, destroy, "
            "instances, logs, ssh_key, exec, rest_exec, jupyter_url, jupyter_exec, credit."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": GPU_ACTIONS,
                           "description": "Action to perform"},
                "gpu_name": {"type": "string", "description": "search: GPU model (default: RTX 4090)"},
                "max_price": {"type": "number", "description": "search: Max $/hr (default: 0.50)"},
                "num_results": {"type": "integer", "description": "search: Results count (default: 5)"},
                "offer_id": {"type": "integer", "description": "create: Offer ID from search"},
                "image": {"type": "string", "description": "create: Docker image (default: pytorch)"},
                "disk": {"type": "number", "description": "create: Disk GB (default: 40)"},
                "label": {"type": "string", "description": "create: Instance label"},
                "instance_id": {"type": "string", "description": "destroy/logs: Instance ID (default: stored)"},
                "tail": {"type": "integer", "description": "logs: Line count (default: 100)"},
                "code": {"type": "string", "description": "exec/jupyter_exec: Code to execute"},
                "command": {"type": "string", "description": "rest_exec: Shell command to execute"},
            },
            "required": ["action"],
        },
    ),
]


# ── Action Handlers ─────────────────────────────────────────

async def _gpu_status(args):
    iid = args.get("instance_id") or _gpu_instance_id()
    if not iid:
        return [TextContent(type="text", text="No instance configured. Store ID in ~/.basho_gpu/instance")]
    data = _vastai("show", "instance", iid, raw=True)
    if isinstance(data, dict) and "error" in data:
        return [TextContent(type="text", text=f"Error: {data['error']}")]
    return [TextContent(type="text", text=(
        f"-- GPU Status --\n"
        f"Instance: {iid}\n"
        f"Status: {data.get('actual_status', '?')} (intended: {data.get('intended_status', '?')})\n"
        f"GPU: {data.get('gpu_name', '?')} | Temp: {data.get('gpu_temp', '?')}C | Util: {data.get('gpu_util', '?')}%\n"
        f"VRAM: {data.get('vmem_usage', 0):.0f}/{data.get('gpu_totalram', 0)}MB\n"
        f"Disk: {data.get('disk_usage', 0):.1f}/{data.get('disk_space', 0):.0f}GB\n"
        f"Cost: ${data.get('dph_total', 0):.4f}/hr\n"
        f"SSH: {data.get('ssh_host', '?')}:{data.get('ssh_port', '?')}\n"
        f"Image: {data.get('image_uuid', '?')}\n"
        f"Label: {data.get('label', 'none')}"
    ))]

async def _gpu_start(args):
    iid = args.get("instance_id") or _gpu_instance_id()
    if not iid:
        return [TextContent(type="text", text="No instance configured.")]
    out = _vastai("start", "instance", iid)
    if isinstance(out, dict) and "error" in out:
        return [TextContent(type="text", text=f"Start failed: {out['error']}")]
    return [TextContent(type="text", text=f"Instance {iid} starting. Wait ~60s.\n{out}")]

async def _gpu_stop(args):
    iid = args.get("instance_id") or _gpu_instance_id()
    if not iid:
        return [TextContent(type="text", text="No instance configured.")]
    out = _vastai("stop", "instance", iid)
    if isinstance(out, dict) and "error" in out:
        return [TextContent(type="text", text=f"Stop failed: {out['error']}")]
    return [TextContent(type="text", text=f"Instance {iid} stopping.\n{out}")]

async def _gpu_search(args):
    gpu_name = args.get("gpu_name", "RTX 4090").replace(" ", "_")
    max_price = args.get("max_price", 0.50)
    num_results = args.get("num_results", 5)
    query = f"gpu_name=={gpu_name} dph_total<{max_price}"
    out = _vastai("search", "offers", "-o", "dph_total", "--limit", str(num_results), query, timeout=25)
    if isinstance(out, dict) and "error" in out:
        return [TextContent(type="text", text=f"Search error: {out['error']}")]
    return [TextContent(type="text", text=f"-- GPU Offers --\n{out}")]

async def _gpu_create(args):
    offer_id = args.get("offer_id")
    if not offer_id:
        return [TextContent(type="text", text="Missing offer_id. Run action=search first.")]
    image = args.get("image", "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel")
    disk = args.get("disk", 40)
    label = args.get("label", "watty-gpu")
    onstart = "apt-get update -qq && apt-get install -y -qq git > /dev/null && pip install -q transformers datasets peft accelerate bitsandbytes trl sentencepiece protobuf && echo 'WATTY GPU READY'"
    out = _vastai(
        "create", "instance", str(offer_id),
        "--image", image,
        "--disk", str(disk),
        "--ssh", "--direct",
        "--onstart-cmd", onstart,
        timeout=30
    )
    if isinstance(out, dict) and "error" in out:
        return [TextContent(type="text", text=f"Create failed: {out['error']}")]
    # Try to extract instance ID from output
    if isinstance(out, str) and "new_contract" in out:
        try:
            d = json.loads(out[out.index("{"):])
            new_id = str(d.get("new_contract", ""))
            if new_id:
                _save_instance_id(new_id)
                return [TextContent(type="text", text=f"Instance created! ID: {new_id}. Wait ~60s for boot.\n{out}")]
        except Exception:
            pass
    return [TextContent(type="text", text=f"Create result:\n{out}")]

async def _gpu_destroy(args):
    iid = args.get("instance_id") or _gpu_instance_id()
    if not iid:
        return [TextContent(type="text", text="No instance to destroy.")]
    out = _vastai("destroy", "instance", iid)
    if isinstance(out, dict) and "error" in out:
        return [TextContent(type="text", text=f"Destroy failed: {out['error']}")]
    return [TextContent(type="text", text=f"Instance {iid} destroyed.\n{out}")]

async def _gpu_instances(args):
    out = _vastai("show", "instances")
    if isinstance(out, dict) and "error" in out:
        return [TextContent(type="text", text=f"Error: {out['error']}")]
    return [TextContent(type="text", text=f"-- Instances --\n{out or 'No active instances.'}")]

async def _gpu_logs(args):
    iid = args.get("instance_id") or _gpu_instance_id()
    tail = args.get("tail", 100)
    if not iid:
        return [TextContent(type="text", text="No instance configured.")]
    out = _vastai("logs", iid, "--tail", str(tail), timeout=15)
    if isinstance(out, dict) and "error" in out:
        return [TextContent(type="text", text=f"Log error: {out['error']}")]
    return [TextContent(type="text", text=f"-- GPU Logs --\n{out or '(empty)'}")]

async def _gpu_ssh_key(args):
    pub = _ssh_key_path().with_suffix(".pub")
    if not pub.exists():
        return [TextContent(type="text", text=f"No public key at {pub}. Generate with: ssh-keygen -t ed25519 -f ~/.ssh/basho_gpu")]
    pub_key = pub.read_text(encoding="utf-8").strip()
    out = _vastai("create", "ssh-key", pub_key)
    if isinstance(out, dict) and "error" in out:
        # Key might already exist
        if "already exists" in str(out.get("error", "")):
            return [TextContent(type="text", text=f"SSH key already on account: {pub_key[:50]}...")]
        return [TextContent(type="text", text=f"Upload failed: {out['error']}")]
    return [TextContent(type="text", text=f"SSH key uploaded: {pub_key[:50]}...\n{out}")]

async def _gpu_exec(args):
    """Execute a command on the GPU via SSH."""
    code = args.get("code", "")
    if not code:
        return [TextContent(type="text", text="Missing code parameter.")]

    # If it starts with "shell:" run as raw shell command, otherwise wrap in python
    if code.startswith("shell:"):
        command = code[6:].strip()
    else:
        # Write code to a temp file and execute it (avoids quoting hell)
        escaped = code.replace("'", "'\\''")
        command = f"python3 -c '{escaped}'"

    out, err = _ssh_cmd(command, timeout=60)
    if err and not out:
        return [TextContent(type="text", text=f"GPU exec error: {err}")]
    result = out
    if err:
        result += f"\nSTDERR: {err}"
    return [TextContent(type="text", text=f"GPU exec:\n{result or '(no output)'}")]

async def _gpu_rest_exec(args):
    """Execute via vastai CLI execute command."""
    command = args.get("command", "")
    if not command:
        return [TextContent(type="text", text="Missing command parameter.")]
    iid = _gpu_instance_id()
    if not iid:
        return [TextContent(type="text", text="No instance configured.")]
    out = _vastai("execute", iid, command, timeout=60)
    if isinstance(out, dict) and "error" in out:
        return [TextContent(type="text", text=f"Execute failed: {out['error']}")]
    return [TextContent(type="text", text=f"GPU exec (REST):\n{out or '(no output)'}")]

async def _gpu_jupyter_url(args):
    iid = _gpu_instance_id()
    if not iid:
        return [TextContent(type="text", text="No instance configured.")]
    data = _vastai("show", "instance", iid, raw=True)
    if isinstance(data, dict) and "error" in data:
        return [TextContent(type="text", text=f"Error: {data['error']}")]
    token = data.get("jupyter_token", "")
    ip = data.get("public_ipaddr", "")
    ports = data.get("ports", {})
    urls = []
    if ports and isinstance(ports, dict):
        for _, mappings in ports.items():
            if isinstance(mappings, list):
                for m in mappings:
                    h = m.get("HostIp", ip)
                    p = m.get("HostPort", "")
                    if h in ("0.0.0.0", "::"):
                        h = ip
                    if h and p:
                        urls.append(f"https://{h}:{p}")
    if not urls and ip:
        urls.append(f"https://{ip}:8080")
    return [TextContent(type="text", text=(
        f"-- Jupyter --\nToken: {token}\n" +
        "\n".join(f"  {u}/?token={token}" for u in urls)
    ))]

async def _gpu_jupyter_exec(args):
    """Execute code by uploading a script via SSH and running it."""
    code = args.get("code", "")
    if not code:
        return [TextContent(type="text", text="Missing code parameter.")]

    # Upload code as a temp script and execute via SSH
    import base64
    encoded = base64.b64encode(code.encode()).decode()
    command = f"echo '{encoded}' | base64 -d > /tmp/_watty_exec.py && python3 /tmp/_watty_exec.py"
    out, err = _ssh_cmd(command, timeout=120)
    if err and not out:
        return [TextContent(type="text", text=f"Jupyter exec error: {err}")]
    result = out
    if err:
        result += f"\nSTDERR: {err}"
    return [TextContent(type="text", text=f"GPU exec (Jupyter):\n{result or '(no output)'}")]

async def _gpu_credit(args):
    data = _vastai("show", "user", raw=True)
    if isinstance(data, dict) and "error" in data:
        return [TextContent(type="text", text=f"Credit error: {data['error']}")]
    try:
        bal = float(data.get("credit", data.get("balance", 0)))
        return [TextContent(type="text", text=f"Credit: ${bal:.2f} | Est: {bal/0.35:.1f}h @ $0.35/hr, {bal/0.17:.1f}h @ $0.17/hr")]
    except Exception:
        return [TextContent(type="text", text=f"Credit: {data.get('credit', '?')}")]


# ── Dispatcher ──────────────────────────────────────────────

_ACTION_MAP = {
    "status": _gpu_status,
    "start": _gpu_start,
    "stop": _gpu_stop,
    "search": _gpu_search,
    "create": _gpu_create,
    "destroy": _gpu_destroy,
    "instances": _gpu_instances,
    "logs": _gpu_logs,
    "ssh_key": _gpu_ssh_key,
    "exec": _gpu_exec,
    "rest_exec": _gpu_rest_exec,
    "jupyter_url": _gpu_jupyter_url,
    "jupyter_exec": _gpu_jupyter_exec,
    "credit": _gpu_credit,
}

async def handle_gpu(args: dict) -> list[TextContent]:
    action = args.get("action", "")
    if action not in _ACTION_MAP:
        return [TextContent(type="text", text=f"Unknown GPU action: {action}. Valid: {', '.join(GPU_ACTIONS)}")]
    return await _ACTION_MAP[action](args)


HANDLERS = {
    "watty_gpu": handle_gpu,
}
