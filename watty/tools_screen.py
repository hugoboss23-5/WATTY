"""
Watty Screen Control v2
========================
One tool: watty_screen(action=...). 8 actions.
Screenshot returns ImageContent (not base64 text blobs).
Screenshots auto-downscale to fit MCP token limits.
find_text uses OCR to locate UI elements by label.
February 2026
"""

import base64
import io
import sys
import threading
import time

import pyautogui
from PIL import Image as PILImage
from mcp.types import Tool, TextContent, ImageContent

from watty.config import WATTY_HOME


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


# ── Config ──────────────────────────────────────────────────

SCREENSHOT_DIR = WATTY_HOME / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

# Max dimension for returned screenshots (keeps base64 under ~80KB)
MAX_SCREENSHOT_DIM = 1280
JPEG_QUALITY = 60

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05


# ── Vision Daemon ───────────────────────────────────────────

class _Vision:
    """Background vision. One frame in memory. Always current."""

    def __init__(self, fps=30):
        self.fps = fps
        self._frame = None
        self._frame_size = (0, 0)
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._backend = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="watty-vision")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        self._backend = None

    def _loop(self):
        interval = 1.0 / self.fps

        # Try dxcam (fastest on Windows)
        try:
            import dxcam
            camera = dxcam.create(output_color="BGR")
            camera.start(target_fps=self.fps, video_mode=True)
            self._backend = "dxcam"
            _log(f"[Watty Screen] Vision started (dxcam, {self.fps}fps)")
            while self._running:
                frame = camera.get_latest_frame()
                if frame is not None:
                    with self._lock:
                        self._frame = frame
                        self._frame_size = (frame.shape[1], frame.shape[0])
                time.sleep(0.001)
            camera.stop()
            del camera
            return
        except Exception:
            pass

        # Try mss (cross-platform)
        try:
            import mss
            self._backend = "mss"
            _log(f"[Watty Screen] Vision started (mss, {self.fps}fps)")
            with mss.mss() as sct:
                monitor = sct.monitors[0]
                while self._running:
                    t0 = time.perf_counter()
                    shot = sct.grab(monitor)
                    with self._lock:
                        self._frame = shot
                        self._frame_size = (shot.width, shot.height)
                    elapsed = time.perf_counter() - t0
                    sleep_for = interval - elapsed
                    if sleep_for > 0:
                        time.sleep(sleep_for)
            return
        except Exception:
            pass

        # Fallback: pyautogui
        self._backend = "pyautogui"
        _log(f"[Watty Screen] Vision started (pyautogui, {self.fps}fps)")
        while self._running:
            t0 = time.perf_counter()
            try:
                img = pyautogui.screenshot()
                with self._lock:
                    self._frame = img
                    self._frame_size = img.size
            except Exception:
                pass
            elapsed = time.perf_counter() - t0
            sleep_for = interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    def see(self) -> tuple:
        """Returns (PIL.Image, width, height) or (None, 0, 0)."""
        with self._lock:
            raw = self._frame
            w, h = self._frame_size

        if raw is None:
            return None, 0, 0

        try:
            import numpy as np
            if isinstance(raw, np.ndarray):
                rgb = raw[:, :, ::-1]
                return PILImage.fromarray(rgb), w, h
        except ImportError:
            pass

        if hasattr(raw, "rgb"):
            return PILImage.frombytes("RGB", (w, h), raw.rgb), w, h

        return raw, w, h

    @property
    def alive(self):
        return self._running

    @property
    def backend(self):
        return self._backend or "none"


_eye = _Vision(fps=30)
_eye.start()


# ── Helpers ─────────────────────────────────────────────────

def _downscale(img: PILImage.Image, max_dim: int = MAX_SCREENSHOT_DIM) -> PILImage.Image:
    """Downscale image so longest side <= max_dim. Returns new image."""
    w, h = img.size
    if w <= max_dim and h <= max_dim:
        return img
    scale = max_dim / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), PILImage.LANCZOS)


def _img_to_jpeg_b64(img: PILImage.Image, quality: int = JPEG_QUALITY) -> str:
    """Convert PIL image to base64 JPEG string."""
    if img.mode == "RGBA":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _save_full_res(img: PILImage.Image) -> str:
    """Save full-res screenshot to disk. Returns path."""
    ts = int(time.time() * 1000)
    path = SCREENSHOT_DIR / f"screen_{ts}.png"
    # Keep only last 5 screenshots to avoid disk bloat
    existing = sorted(SCREENSHOT_DIR.glob("screen_*.png"))
    for old in existing[:-4]:
        try:
            old.unlink()
        except Exception:
            pass
    img.save(str(path), format="PNG")
    return str(path)


# ── Tool Definition ─────────────────────────────────────────

SCREEN_ACTIONS = ["screenshot", "click", "type", "key", "move", "scroll", "drag", "find_text"]

TOOLS = [
    Tool(
        name="watty_screen",
        description=(
            "Desktop control. One tool, many actions.\n"
            "Actions: screenshot, click, type, key, move, scroll, drag, find_text.\n\n"
            "screenshot: Returns a downscaled image + mouse position. Full-res saved to disk.\n"
            "find_text: OCR the screen to find text and return its coordinates.\n"
            "click/type/key/move/scroll/drag: Standard input actions."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": SCREEN_ACTIONS,
                           "description": "Action to perform"},
                "x": {"type": "integer", "description": "click/move: X coordinate"},
                "y": {"type": "integer", "description": "click/move: Y coordinate"},
                "right_click": {"type": "boolean", "description": "click: Right-click (default: false)"},
                "double_click": {"type": "boolean", "description": "click: Double-click (default: false)"},
                "text": {"type": "string", "description": "type/find_text: Text to type or search for"},
                "interval": {"type": "number", "description": "type: Seconds between keystrokes (default: 0.02)"},
                "combo": {"type": "string", "description": 'key: Key combo (e.g. "ctrl+c", "enter")'},
                "amount": {"type": "integer", "description": "scroll: Amount (positive=up, negative=down)"},
                "x1": {"type": "integer", "description": "drag: Start X"},
                "y1": {"type": "integer", "description": "drag: Start Y"},
                "x2": {"type": "integer", "description": "drag: End X"},
                "y2": {"type": "integer", "description": "drag: End Y"},
            },
            "required": ["action"],
        },
    ),
]


# ── Action Handlers ────────────────────────────────────────

async def _screenshot(args):
    """Capture screen. Returns ImageContent (small JPEG) + metadata text."""
    try:
        img, width, height = _eye.see()
        if img is None:
            img = pyautogui.screenshot()
            width, height = img.size

        mouse_x, mouse_y = pyautogui.position()

        # Save full-res to disk
        full_path = _save_full_res(img)

        # Downscale for MCP response
        small = _downscale(img)
        b64 = _img_to_jpeg_b64(small)

        return [
            TextContent(type="text", text=(
                f"Screen: {width}x{height} | Mouse: ({mouse_x}, {mouse_y}) | "
                f"Backend: {_eye.backend} | Full-res: {full_path}"
            )),
            ImageContent(type="image", data=b64, mimeType="image/jpeg"),
        ]
    except Exception as e:
        return [TextContent(type="text", text=f"Screenshot error: {e}")]


async def _click(args):
    x, y = args.get("x", 0), args.get("y", 0)
    try:
        if args.get("double_click"):
            pyautogui.doubleClick(x, y)
            action = "Double-clicked"
        elif args.get("right_click"):
            pyautogui.rightClick(x, y)
            action = "Right-clicked"
        else:
            pyautogui.click(x, y)
            action = "Clicked"
        return [TextContent(type="text", text=f"{action} at ({x}, {y})")]
    except Exception as e:
        return [TextContent(type="text", text=f"Click error: {e}")]


async def _type(args):
    text = args.get("text", "")
    interval = args.get("interval", 0.02)
    if not text:
        return [TextContent(type="text", text="Nothing to type.")]
    try:
        # Use pyperclip + ctrl+v for reliable text entry (handles all chars)
        import pyperclip
        pyperclip.copy(text)
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.1)
        return [TextContent(type="text", text=f"Typed {len(text)} characters (via clipboard).")]
    except ImportError:
        pass
    try:
        if text.isascii():
            pyautogui.typewrite(text, interval=interval)
        else:
            pyautogui.write(text, interval=interval)
        return [TextContent(type="text", text=f"Typed {len(text)} characters.")]
    except Exception as e:
        return [TextContent(type="text", text=f"Type error: {e}")]


async def _key(args):
    combo = args.get("combo", "")
    if not combo:
        return [TextContent(type="text", text="No key combo provided.")]
    try:
        keys = [k.strip().lower() for k in combo.split("+")]
        if len(keys) == 1:
            pyautogui.press(keys[0])
        else:
            pyautogui.hotkey(*keys)
        return [TextContent(type="text", text=f"Pressed: {combo}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Key error: {e}")]


async def _move(args):
    x, y = args.get("x", 0), args.get("y", 0)
    try:
        pyautogui.moveTo(x, y)
        return [TextContent(type="text", text=f"Moved mouse to ({x}, {y})")]
    except Exception as e:
        return [TextContent(type="text", text=f"Move error: {e}")]


async def _scroll(args):
    amount = args.get("amount", 0)
    try:
        pyautogui.scroll(amount)
        direction = "up" if amount > 0 else "down"
        return [TextContent(type="text", text=f"Scrolled {direction} by {abs(amount)}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Scroll error: {e}")]


async def _drag(args):
    x1, y1 = args.get("x1", 0), args.get("y1", 0)
    x2, y2 = args.get("x2", 0), args.get("y2", 0)
    try:
        pyautogui.moveTo(x1, y1)
        pyautogui.mouseDown()
        pyautogui.moveTo(x2, y2, duration=0.3)
        pyautogui.mouseUp()
        return [TextContent(type="text", text=f"Dragged from ({x1}, {y1}) to ({x2}, {y2})")]
    except Exception as e:
        return [TextContent(type="text", text=f"Drag error: {e}")]


async def _find_text(args):
    """OCR the screen to find text and return real-pixel coordinates."""
    search = args.get("text", "").strip().lower()
    if not search:
        return [TextContent(type="text", text="No text to search for.")]

    try:
        import pytesseract
    except ImportError:
        return [TextContent(type="text", text=(
            "pytesseract not installed. Install with: pip install pytesseract\n"
            "Also need Tesseract OCR binary: https://github.com/tesseract-ocr/tesseract"
        ))]

    try:
        img, width, height = _eye.see()
        if img is None:
            img = pyautogui.screenshot()
            width, height = img.size

        # OCR with bounding box data
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        matches = []
        n = len(data["text"])
        for i in range(n):
            txt = data["text"][i].strip()
            if not txt:
                continue
            if search in txt.lower():
                x = data["left"][i]
                y = data["top"][i]
                w = data["width"][i]
                h = data["height"][i]
                cx, cy = x + w // 2, y + h // 2
                conf = data["conf"][i]
                matches.append(f"  '{txt}' at center=({cx}, {cy}) box=({x}, {y}, {w}x{h}) conf={conf}%")

        if not matches:
            # Try multi-word matching by joining consecutive words
            words = []
            for i in range(n):
                txt = data["text"][i].strip()
                if txt:
                    words.append((i, txt))

            for wi in range(len(words)):
                combined = ""
                last_idx = wi
                for wj in range(wi, min(wi + 5, len(words))):
                    combined += (" " if combined else "") + words[wj][1]
                    last_idx = wj
                    if search in combined.lower():
                        # Bounding box spans from first to last word
                        fi = words[wi][0]
                        li = words[last_idx][0]
                        x1 = data["left"][fi]
                        y1 = min(data["top"][fi], data["top"][li])
                        x2 = data["left"][li] + data["width"][li]
                        y2 = max(data["top"][fi] + data["height"][fi],
                                 data["top"][li] + data["height"][li])
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        matches.append(f"  '{combined}' at center=({cx}, {cy}) box=({x1}, {y1}, {x2-x1}x{y2-y1})")
                        break

        if not matches:
            return [TextContent(type="text", text=f"Text '{search}' not found on screen. Screen: {width}x{height}")]

        return [TextContent(type="text", text=(
            f"Found '{search}' on screen ({width}x{height}):\n" + "\n".join(matches[:10])
            + "\n\nUse the center coordinates to click on the element."
        ))]
    except Exception as e:
        return [TextContent(type="text", text=f"find_text error: {e}")]


# ── Dispatcher ─────────────────────────────────────────────

_ACTION_MAP = {
    "screenshot": _screenshot,
    "click": _click,
    "type": _type,
    "key": _key,
    "move": _move,
    "scroll": _scroll,
    "drag": _drag,
    "find_text": _find_text,
}

async def handle_screen(args: dict) -> list:
    action = args.get("action", "")
    if action not in _ACTION_MAP:
        return [TextContent(type="text", text=f"Unknown screen action: {action}. Valid: {', '.join(SCREEN_ACTIONS)}")]
    return await _ACTION_MAP[action](args)


# ── Router ──────────────────────────────────────────────────

HANDLERS = {
    "watty_screen": handle_screen,
}
