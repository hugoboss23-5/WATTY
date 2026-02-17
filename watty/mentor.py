"""
Watty Mentor — Live Coding Tutor
=================================
Watches code changes in real-time, detects patterns and techniques,
generates learning questions so Hugo levels up by watching Claude code.

How it works:
  1. File watcher polls for changes in the watty codebase
  2. Diffs are analyzed for coding patterns (decorators, async, data structures, etc.)
  3. Questions are generated about WHY those patterns were used
  4. Questions are stored and served via web dashboard + MCP tool

Hugo & Rim · February 2026
"""

import json
import re
import time
import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

from mcp.types import Tool, TextContent

from watty.config import WATTY_HOME


# ── Config ──────────────────────────────────────────────────

MENTOR_DIR = WATTY_HOME / "mentor"
QUESTIONS_FILE = MENTOR_DIR / "questions.jsonl"
SNAPSHOTS_FILE = MENTOR_DIR / "file_snapshots.json"
STATS_FILE = MENTOR_DIR / "stats.json"

# Which directories to watch
WATCH_DIRS = [
    str(Path(__file__).parent),  # watty/ source
]

WATCH_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css"}
IGNORE_PATTERNS = {"__pycache__", ".git", "node_modules", ".pyc", "egg-info"}

# ── Pattern Definitions ─────────────────────────────────────
# Each pattern: (regex, concept_name, difficulty, question_templates)

PATTERNS = [
    # Python fundamentals
    {
        "regex": r"def\s+(\w+)\s*\(self",
        "concept": "Instance Methods",
        "difficulty": "beginner",
        "qa": [
            ("Why is `self` the first parameter in `{match}`? What does it refer to?",
             "`self` is the instance the method is called on. When you write `obj.method()`, Python passes `obj` as the first argument automatically. It's how the method accesses the object's data and other methods."),
            ("What's the difference between an instance method and a regular function?",
             "An instance method lives inside a class and receives `self` — access to the object's state. A regular function is standalone with no implicit object context. Use methods when behavior belongs to a specific object."),
        ],
    },
    {
        "regex": r"@(\w+)",
        "concept": "Decorators",
        "difficulty": "intermediate",
        "qa": [
            ("What does the `@{match}` decorator do? How does it modify the function below it?",
             "A decorator wraps a function inside another function. `@foo` above `def bar()` is shorthand for `bar = foo(bar)`. It lets you add behavior (logging, auth, caching) without changing the original function's code."),
            ("Could you write this code without using a decorator? What would it look like?",
             "Yes. Instead of `@decorator` above `def func():`, you'd write `func = decorator(func)` after defining `func`. Decorators are just syntactic sugar that makes this pattern readable."),
        ],
    },
    {
        "regex": r"async\s+def\s+(\w+)",
        "concept": "Async/Await",
        "difficulty": "intermediate",
        "qa": [
            ("Why is `{match}` async? What would happen if it were a regular function?",
             "Async lets the function pause while waiting for I/O (network, disk) without blocking other work. A regular function would block the entire thread until the I/O completes, freezing everything else."),
            ("What's the difference between `async def` and `def`? When do you need async?",
             "`async def` creates a coroutine that can be paused/resumed. Use it when your function does I/O (API calls, file reads, database queries) and you want other tasks to run during the wait. Don't need it for pure computation."),
        ],
    },
    {
        "regex": r"await\s+(\w+)",
        "concept": "Awaiting Coroutines",
        "difficulty": "intermediate",
        "qa": [
            ("Why do we `await` here? What happens if you forget the `await`?",
             "`await` pauses this coroutine until the awaited one finishes, then resumes with the result. Without `await`, you get a coroutine object instead of the actual result — the code never actually runs."),
            ("What does `await` actually do under the hood? Does it block the thread?",
             "No, it doesn't block the thread. `await` yields control back to the event loop, which can run other coroutines. When the awaited task completes, the event loop resumes this coroutine. It's cooperative multitasking."),
        ],
    },
    {
        "regex": r"class\s+(\w+)(?:\((\w+)\))?:",
        "concept": "Classes & Inheritance",
        "difficulty": "beginner",
        "qa": [
            ("What is the class `{match}` responsible for? Why was it made a class instead of loose functions?",
             "A class bundles related data and behavior together. You use a class when multiple functions operate on the same state — instead of passing data between functions, the object holds it. It's about organizing responsibility."),
            ("If it inherits from another class, what does it get for free from the parent?",
             "All of the parent's methods and attributes. The child can use them as-is, override them, or extend them. Inheritance lets you build specialized versions without rewriting shared logic."),
        ],
    },
    {
        "regex": r"\{[^}]*:\s*[^}]+\s+for\s+\w+\s+in\s+",
        "concept": "Dict Comprehension",
        "difficulty": "intermediate",
        "qa": [
            ("What does this dict comprehension build? Could you rewrite it as a regular for loop?",
             "A dict comprehension creates a dictionary in one expression: `{key: value for item in iterable}`. As a loop: `d = {}; for item in iterable: d[key] = value`. The comprehension is more concise but less debuggable."),
            ("When should you use a dict comprehension vs building a dict in a loop?",
             "Use comprehensions for simple transforms (under ~80 chars). Use a loop when you need conditionals, error handling, or the logic is complex enough that readability suffers. If you need to add a comment explaining the comprehension, use a loop."),
        ],
    },
    {
        "regex": r"\[[^\]]*\s+for\s+\w+\s+in\s+",
        "concept": "List Comprehension",
        "difficulty": "beginner",
        "qa": [
            ("What list does this comprehension create? Write it as a normal for loop.",
             "`[expr for x in items]` builds a list by evaluating `expr` for each `x`. As a loop: `result = []; for x in items: result.append(expr)`. Comprehensions are faster because Python optimizes the append internally."),
            ("When does a list comprehension become too complex and should be a regular loop?",
             "When it has nested loops, multiple conditions, or side effects. Rule of thumb: if it doesn't fit on one line or needs a comment, break it into a loop. Readability beats cleverness."),
        ],
    },
    {
        "regex": r"try:\s*\n.*?except\s+(\w+)",
        "concept": "Exception Handling",
        "difficulty": "beginner",
        "qa": [
            ("Why catch `{match}` specifically instead of a bare `except`?",
             "Catching a specific exception means you only handle errors you expect and know how to recover from. A bare `except` catches EVERYTHING — including typos (NameError), keyboard interrupts, and bugs you'd want to know about. Specific catches prevent hiding real problems."),
            ("What could go wrong in the `try` block? What does the program do when it fails?",
             "The try block wraps code that might fail (file not found, network timeout, bad data). The except block defines the fallback — maybe return a default value, log the error, or retry. The program continues instead of crashing."),
        ],
    },
    {
        "regex": r"lambda\s+\w+",
        "concept": "Lambda Functions",
        "difficulty": "intermediate",
        "qa": [
            ("What does this lambda do? Could you replace it with a named function?",
             "A lambda is an anonymous one-line function: `lambda x: x*2` equals `def double(x): return x*2`. You can always replace a lambda with a named function. Lambdas are just shorter for simple callbacks and sort keys."),
            ("When are lambdas cleaner than regular functions, and when are they worse?",
             "Cleaner: as sort keys (`sorted(items, key=lambda x: x.name)`), simple callbacks, quick transforms. Worse: when the logic needs a name to be understood, has multiple expressions, or needs to be reused. If you'd name it to explain it, don't use lambda."),
        ],
    },
    {
        "regex": r"threading\.Thread\(",
        "concept": "Threading",
        "difficulty": "advanced",
        "qa": [
            ("Why use a separate thread here? What runs concurrently?",
             "A thread lets code run in the background while the main program continues. Common uses: background timers, periodic tasks (like dream cycles), non-blocking I/O. The main thread and background thread execute concurrently."),
            ("What's the `daemon=True` flag do? What happens to daemon threads when the main program exits?",
             "Daemon threads are killed automatically when the main program exits. Without `daemon=True`, the program would hang waiting for the thread to finish. Use daemon for background services that should die with the app."),
        ],
    },
    {
        "regex": r"global\s+(\w+)",
        "concept": "Global State",
        "difficulty": "intermediate",
        "qa": [
            ("Why is `{match}` global? What are the risks of global state?",
             "Global state is shared across the entire module. Risks: any function can change it unexpectedly, it makes testing harder, and it creates hidden dependencies. But sometimes it's the simplest way to share state (like a singleton brain instance)."),
            ("How could you redesign this to avoid using `global`?",
             "Pass the value as a function parameter, use a class to hold state, or use dependency injection. For example, instead of a global `brain`, pass `brain` as an argument to every function that needs it. More explicit, more testable."),
        ],
    },
    {
        "regex": r"import\s+(\w+)|from\s+(\w+)\s+import",
        "concept": "Module Imports",
        "difficulty": "beginner",
        "qa": [
            ("What does this import give us? Why import it here instead of elsewhere?",
             "Imports bring in code from other modules. Top-level imports run once when the module loads. Imports inside functions run each time the function is called — useful for avoiding circular imports or deferring heavy loads."),
            ("What's the difference between `import x` and `from x import y`?",
             "`import x` gives you the whole module (access via `x.thing`). `from x import y` pulls `y` directly into your namespace. The first is safer (no name collisions), the second is more convenient for frequently used names."),
        ],
    },
    {
        "regex": r"with\s+open\(",
        "concept": "Context Managers",
        "difficulty": "intermediate",
        "qa": [
            ("Why use `with open(...)` instead of just `open(...)`? What happens when the `with` block ends?",
             "`with` guarantees the file gets closed when the block ends — even if an error occurs. Without it, you must manually call `.close()`, and if you forget (or an exception fires first), the file handle leaks. `with` = automatic cleanup."),
            ("What problem does a context manager solve? What could go wrong without it?",
             "Resource leaks. Open files, database connections, network sockets — all need to be released. Without a context manager, an exception mid-function means cleanup code never runs. `with` is Python's guarantee that cleanup always happens."),
        ],
    },
    {
        "regex": r"\.get\(\s*['\"](\w+)['\"]",
        "concept": "Safe Dict Access",
        "difficulty": "beginner",
        "qa": [
            ("Why use `.get('{match}')` instead of `['{match}']`? What's the difference?",
             "`.get()` returns `None` (or a default) if the key is missing. `[]` raises a `KeyError` and crashes. Use `.get()` when the key might not exist. Use `[]` when it MUST exist and missing it is a bug you want to catch."),
        ],
    },
    {
        "regex": r"json\.dumps\(|json\.loads\(",
        "concept": "JSON Serialization",
        "difficulty": "beginner",
        "qa": [
            ("What's the difference between `json.dumps()` and `json.loads()`?",
             "`dumps` = dump to string (Python object → JSON text). `loads` = load from string (JSON text → Python object). Think: 's' = string. `dump`/`load` without 's' work with files instead of strings."),
        ],
    },
    {
        "regex": r"hashlib\.\w+\(",
        "concept": "Hashing",
        "difficulty": "intermediate",
        "qa": [
            ("Why hash this data? What property of hashes makes them useful here?",
             "Hashes turn any data into a fixed-size fingerprint. Same input = same hash, always. Different input = different hash (usually). Useful for: deduplication (have I seen this before?), integrity checks, and fast lookups."),
            ("Can you reverse a hash to get the original data? Why or why not?",
             "No. Hashing is one-way by design — infinite inputs map to finite outputs, so information is permanently lost. That's what makes hashes useful for passwords (can't reverse them) and why they're NOT encryption (can't decrypt)."),
        ],
    },
    {
        "regex": r"re\.(search|match|findall|sub|compile)\(",
        "concept": "Regular Expressions",
        "difficulty": "intermediate",
        "qa": [
            ("What pattern is this regex looking for? Can you describe it in plain English?",
             "Read the regex character by character: `.` = any char, `*` = zero or more, `+` = one or more, `\\d` = digit, `\\w` = word char, `[]` = any char in set, `()` = capture group. Build the English meaning piece by piece."),
            ("What's the difference between `re.search` and `re.match`?",
             "`match` only checks the START of the string. `search` checks ANYWHERE in the string. `re.match('cat', 'the cat')` fails. `re.search('cat', 'the cat')` finds it. Almost always use `search` unless you specifically need start-anchored matching."),
        ],
    },
    {
        "regex": r"yield\s+",
        "concept": "Generators",
        "difficulty": "advanced",
        "qa": [
            ("Why use `yield` instead of `return`? How does a generator differ from a regular function?",
             "A generator produces values one at a time, pausing between each `yield`. A regular function computes everything at once and returns. Generators are lazy — they only do work when you ask for the next value. Use them for large sequences or infinite streams."),
            ("What's the memory advantage of generators over returning a full list?",
             "A list holds ALL items in memory at once. A generator holds only ONE item at a time. Processing 10 million rows? A list needs 10M items in RAM. A generator needs 1. The tradeoff: you can only iterate once (no random access)."),
        ],
    },
    {
        "regex": r"\.join\(",
        "concept": "String Joining",
        "difficulty": "beginner",
        "qa": [
            ("Why use `'x'.join(list)` instead of concatenating strings in a loop?",
             "String concatenation with `+` creates a new string every iteration (strings are immutable). For N strings, that's O(N^2) copying. `.join()` allocates once and copies once — O(N). For small lists it doesn't matter, for large ones it's dramatically faster."),
        ],
    },
    {
        "regex": r"if\s+__name__\s*==\s*['\"]__main__['\"]",
        "concept": "Main Guard",
        "difficulty": "beginner",
        "qa": [
            ("What does `if __name__ == '__main__'` do? When does this code run vs not run?",
             "When you RUN a file directly (`python file.py`), `__name__` is `'__main__'`, so the code runs. When you IMPORT it (`from file import func`), `__name__` is the module name, so it doesn't. This lets a file be both a runnable script and an importable library."),
        ],
    },
    # Design patterns
    {
        "regex": r"HANDLERS\s*=\s*\{",
        "concept": "Handler Pattern / Dispatch Table",
        "difficulty": "intermediate",
        "qa": [
            ("Why use a dispatch dictionary instead of a chain of if/elif statements?",
             "A dispatch dict maps names to functions: `{'action': handler_func}`. Adding a new action = adding one line to the dict. With if/elif, you edit a growing chain. Dispatch tables are O(1) lookup, extensible, and keep handler registration separate from dispatch logic."),
        ],
    },
    {
        "regex": r"def\s+_\w+",
        "concept": "Private Functions (Underscore Convention)",
        "difficulty": "beginner",
        "qa": [
            ("Why does this function start with `_`? What does it signal to other developers?",
             "Leading underscore means 'internal use only — don't call this from outside the module.' Python doesn't enforce it (you CAN call `_func()`), but it's a strong convention: this function is an implementation detail that may change without warning."),
        ],
    },
    {
        "regex": r"\.fetchone\(\)|\.fetchall\(\)|\.execute\(",
        "concept": "SQL Database Operations",
        "difficulty": "intermediate",
        "qa": [
            ("What SQL query is being run? What data does it return?",
             "Read the SQL string: SELECT = read data, INSERT = add data, UPDATE = modify, DELETE = remove. The WHERE clause filters which rows. `.execute()` sends the query, `.fetchone()` gets one row, `.fetchall()` gets all matching rows."),
            ("What's the difference between `fetchone()` and `fetchall()`? When use which?",
             "`fetchone()` returns a single row (or None). `fetchall()` returns all rows as a list. Use `fetchone()` when you expect one result (by ID lookup). Use `fetchall()` for queries that return multiple rows. For huge result sets, iterate instead of fetchall to save memory."),
        ],
    },
    {
        "regex": r"requests\.get\(|requests\.post\(",
        "concept": "HTTP Requests",
        "difficulty": "beginner",
        "qa": [
            ("What does this HTTP request fetch? Why GET vs POST?",
             "GET = retrieve data (reading). POST = send data (writing/creating). GET has no body, parameters go in the URL. POST has a body for sending complex data. GET requests should be safe to repeat; POST may create something new each time."),
        ],
    },
    {
        "regex": r"\.encode\(|\.decode\(",
        "concept": "String Encoding",
        "difficulty": "intermediate",
        "qa": [
            ("Why encode/decode here? What's the difference between a string and bytes in Python?",
             "Strings are text (Unicode characters). Bytes are raw data (0-255 values). Networks and files work in bytes. `.encode()` converts string→bytes for sending. `.decode()` converts bytes→string for reading. UTF-8 is the standard encoding — handles all languages."),
        ],
    },
    {
        "regex": r"defaultdict|OrderedDict|Counter|namedtuple|deque",
        "concept": "Collections Module",
        "difficulty": "intermediate",
        "qa": [
            ("Why use this specialized data structure instead of a regular dict/list?",
             "`defaultdict` auto-creates missing keys. `Counter` counts things. `deque` is fast at both ends (lists are slow at the front). `namedtuple` gives tuples named fields. Each solves a specific problem more cleanly than the generic type."),
        ],
    },
    {
        "regex": r"Path\(|pathlib",
        "concept": "Path Objects (pathlib)",
        "difficulty": "beginner",
        "qa": [
            ("Why use `Path()` instead of plain string paths?",
             "Path objects handle `/` vs `\\` automatically (cross-platform). They have methods like `.exists()`, `.read_text()`, `.mkdir()` built in. String paths require `os.path.join()`, `os.path.exists()` etc. — more verbose, more error-prone."),
        ],
    },
    {
        "regex": r"f['\"].*\{.*\}.*['\"]",
        "concept": "F-Strings",
        "difficulty": "beginner",
        "qa": [
            ("What values get inserted into this f-string? How is it different from `.format()` or `%`?",
             "F-strings evaluate expressions inside `{}` at runtime: `f'Hello {name}'`. They're faster than `.format()` and more readable than `%`. You can put any expression inside the braces — math, function calls, even conditionals."),
        ],
    },
    {
        "regex": r"\*args|\*\*kwargs",
        "concept": "Variable Arguments",
        "difficulty": "intermediate",
        "qa": [
            ("What do `*args` and `**kwargs` allow? Why use them instead of fixed parameters?",
             "`*args` collects extra positional arguments as a tuple. `**kwargs` collects extra keyword arguments as a dict. Use them when you don't know how many arguments will be passed — like wrapper functions, decorators, or flexible APIs."),
        ],
    },
]


# ── Helpers ─────────────────────────────────────────────────

def _ensure_dir():
    MENTOR_DIR.mkdir(parents=True, exist_ok=True)


def _hash_file(path: str) -> str:
    try:
        content = Path(path).read_bytes()
        return hashlib.md5(content).hexdigest()
    except (OSError, PermissionError):
        return ""


def _load_snapshots() -> dict:
    if SNAPSHOTS_FILE.exists():
        try:
            return json.loads(SNAPSHOTS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_snapshots(snaps: dict):
    _ensure_dir()
    SNAPSHOTS_FILE.write_text(json.dumps(snaps), encoding="utf-8")


def _load_stats() -> dict:
    if STATS_FILE.exists():
        try:
            return json.loads(STATS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"total_questions": 0, "concepts_seen": {}, "files_watched": 0, "scans": 0}


def _save_stats(stats: dict):
    _ensure_dir()
    STATS_FILE.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def _append_question(q: dict):
    _ensure_dir()
    with open(QUESTIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(q) + "\n")


def _load_questions(n: int = 50) -> list[dict]:
    if not QUESTIONS_FILE.exists():
        return []
    lines = QUESTIONS_FILE.read_text(encoding="utf-8").strip().split("\n")
    questions = []
    for line in lines[-n:]:
        try:
            questions.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return questions[::-1]  # most recent first


# ── Diff Detection ──────────────────────────────────────────

def _get_changed_files() -> list[tuple[str, str]]:
    """Returns list of (filepath, new_content) for files that changed since last scan."""
    snaps = _load_snapshots()
    changed = []

    for watch_dir in WATCH_DIRS:
        watch_path = Path(watch_dir)
        if not watch_path.exists():
            continue
        for f in watch_path.rglob("*"):
            if not f.is_file():
                continue
            if f.suffix not in WATCH_EXTENSIONS:
                continue
            if any(ig in str(f) for ig in IGNORE_PATTERNS):
                continue

            fpath = str(f)
            new_hash = _hash_file(fpath)
            old_hash = snaps.get(fpath, "")

            if new_hash and new_hash != old_hash:
                try:
                    content = f.read_text(encoding="utf-8", errors="ignore")
                    changed.append((fpath, content))
                except (OSError, PermissionError):
                    continue
                snaps[fpath] = new_hash

    _save_snapshots(snaps)
    return changed


def _get_diff_lines(old_content: str, new_content: str) -> list[str]:
    """Simple line-level diff — returns only new/changed lines."""
    old_lines = set(old_content.split("\n")) if old_content else set()
    new_lines = new_content.split("\n")
    return [line for line in new_lines if line.strip() and line not in old_lines]


# ── Pattern Analysis ────────────────────────────────────────

def _analyze_code(content: str, filepath: str) -> list[dict]:
    """Analyze code content for patterns and generate questions + answers."""
    generated = []
    seen_concepts = set()
    filename = Path(filepath).name

    for pattern in PATTERNS:
        matches = re.finditer(pattern["regex"], content)
        for match in matches:
            concept = pattern["concept"]
            if concept in seen_concepts:
                continue  # one question per concept per file
            seen_concepts.add(concept)

            # Pick a match group for the template
            match_text = match.group(1) if match.lastindex and match.group(1) else match.group(0)
            match_text = match_text[:60]  # truncate

            # Pick a Q&A pair
            import random
            q_text, a_text = random.choice(pattern["qa"])
            try:
                question = q_text.format(match=match_text)
            except (KeyError, IndexError):
                question = q_text.replace("{match}", match_text)
            try:
                answer = a_text.format(match=match_text)
            except (KeyError, IndexError):
                answer = a_text.replace("{match}", match_text)

            # Get context (surrounding lines)
            start = max(0, match.start() - 100)
            end = min(len(content), match.end() + 100)
            context = content[start:end].strip()

            generated.append({
                "question": question,
                "answer": answer,
                "concept": concept,
                "difficulty": pattern["difficulty"],
                "file": filename,
                "filepath": filepath,
                "line": content[:match.start()].count("\n") + 1,
                "context": context[:300],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "answered": False,
            })

    return generated


# ── Self-Learning (for future Claude sessions) ─────────────

_brain_ref = None

def set_brain(brain):
    global _brain_ref
    _brain_ref = brain


def _store_coding_insight(filepath: str, qa_pairs: list[dict], content: str):
    """Store coding Q&A insights into Watty's brain so future Claude sessions learn from them."""
    if not _brain_ref or not qa_pairs:
        return
    filename = Path(filepath).name
    concepts = list({q["concept"] for q in qa_pairs})

    # Store a compact Q&A digest per file
    lines = [f"[CODING Q&A] {filename}", f"Concepts: {', '.join(concepts)}", ""]
    for qa in qa_pairs[:5]:  # top 5 per file to keep it lean
        lines.append(f"Q: {qa['question']}")
        lines.append(f"A: {qa['answer']}")
        lines.append("")

    try:
        _brain_ref.store_memory("\n".join(lines), provider="mentor")
    except Exception:
        pass  # non-critical


def get_session_review(n: int = 10) -> list[dict]:
    """Get recent Q&A pairs for Claude to review at session start.
    Called by watty_enter to feed knowledge back into the current session."""
    questions = _load_questions(100)
    # Prioritize: recent, diverse concepts, include answers
    seen_concepts = set()
    review = []
    for q in questions:
        if q["concept"] in seen_concepts:
            continue
        if not q.get("answer"):
            continue
        seen_concepts.add(q["concept"])
        review.append(q)
        if len(review) >= n:
            break
    return review


# ── Main Scanner ────────────────────────────────────────────

def scan_for_changes() -> dict:
    """Scan watched files for changes and generate new questions.
    Also stores coding insights into brain for future Claude self-learning."""
    changed = _get_changed_files()
    stats = _load_stats()
    new_questions = []

    for filepath, content in changed:
        questions = _analyze_code(content, filepath)
        for q in questions:
            _append_question(q)
            new_questions.append(q)
            # Update stats
            stats["total_questions"] = stats.get("total_questions", 0) + 1
            concept = q["concept"]
            stats["concepts_seen"][concept] = stats["concepts_seen"].get(concept, 0) + 1

        # Store Q&A insights for future Claude sessions
        if questions:
            _store_coding_insight(filepath, questions, content)

    stats["scans"] = stats.get("scans", 0) + 1
    stats["files_watched"] = len(_load_snapshots())
    stats["last_scan"] = datetime.now(timezone.utc).isoformat()
    _save_stats(stats)

    return {
        "files_changed": len(changed),
        "questions_generated": len(new_questions),
        "questions": new_questions[:10],  # return top 10
    }


def get_quiz(n: int = 5, difficulty: str = None) -> list[dict]:
    """Get a random quiz from unanswered questions."""
    questions = _load_questions(200)
    pool = [q for q in questions if not q.get("answered")]
    if difficulty:
        pool = [q for q in pool if q.get("difficulty") == difficulty]

    import random
    if len(pool) > n:
        pool = random.sample(pool, n)
    return pool[:n]


def get_progress() -> dict:
    """Learning progress report."""
    stats = _load_stats()
    questions = _load_questions(500)
    total = len(questions)
    answered = sum(1 for q in questions if q.get("answered"))

    # Concept breakdown
    concepts = defaultdict(lambda: {"total": 0, "beginner": 0, "intermediate": 0, "advanced": 0})
    for q in questions:
        c = q.get("concept", "Unknown")
        d = q.get("difficulty", "beginner")
        concepts[c]["total"] += 1
        concepts[c][d] += 1

    # Sort by most seen
    top_concepts = sorted(concepts.items(), key=lambda x: x[1]["total"], reverse=True)[:15]

    return {
        "total_questions": total,
        "answered": answered,
        "unanswered": total - answered,
        "scans": stats.get("scans", 0),
        "files_watched": stats.get("files_watched", 0),
        "last_scan": stats.get("last_scan", "never"),
        "top_concepts": [{"concept": c, **d} for c, d in top_concepts],
        "difficulty_breakdown": {
            "beginner": sum(1 for q in questions if q.get("difficulty") == "beginner"),
            "intermediate": sum(1 for q in questions if q.get("difficulty") == "intermediate"),
            "advanced": sum(1 for q in questions if q.get("difficulty") == "advanced"),
        },
    }


# ── Background Watcher ──────────────────────────────────────

_watcher_thread = None
_watcher_running = False


def start_watcher(interval: int = 30):
    """Start background file watcher."""
    global _watcher_thread, _watcher_running
    if _watcher_running:
        return {"status": "already running"}

    _watcher_running = True

    def _watch_loop():
        while _watcher_running:
            try:
                result = scan_for_changes()
                if result["questions_generated"] > 0:
                    print(f"[Mentor] {result['questions_generated']} new questions from {result['files_changed']} files", flush=True)
            except Exception as e:
                print(f"[Mentor] Error: {e}", flush=True)
            time.sleep(interval)

    _watcher_thread = threading.Thread(target=_watch_loop, daemon=True, name="watty-mentor")
    _watcher_thread.start()
    return {"status": "started", "interval": interval}


def stop_watcher():
    """Stop background file watcher."""
    global _watcher_running
    _watcher_running = False
    return {"status": "stopped"}


# ── MCP Tool ────────────────────────────────────────────────

TOOLS = [
    Tool(
        name="watty_mentor",
        description=(
            "Watty's coding mentor. Watches code changes and generates learning Q&A.\n"
            "Actions: scan (check new code), quiz (practice questions + answers), "
            "review (Claude self-review at session start), "
            "progress (learning stats), start (begin live watching), stop (pause watching)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["scan", "quiz", "review", "progress", "start", "stop"],
                    "description": "Action to perform",
                },
                "n": {
                    "type": "integer",
                    "description": "quiz: Number of questions (default: 5)",
                },
                "difficulty": {
                    "type": "string",
                    "enum": ["beginner", "intermediate", "advanced"],
                    "description": "quiz: Filter by difficulty",
                },
            },
            "required": ["action"],
        },
    ),
]


async def handle_mentor(arguments: dict) -> list[TextContent]:
    action = arguments.get("action", "progress")

    if action == "scan":
        result = scan_for_changes()
        lines = [
            "Mentor scan complete.",
            f"  Files changed: {result['files_changed']}",
            f"  New questions: {result['questions_generated']}",
        ]
        if result["questions"]:
            lines.append("\nNew questions for you:")
            for i, q in enumerate(result["questions"][:5], 1):
                lines.append(f"\n  [{i}] ({q['difficulty']}) {q['concept']}")
                lines.append(f"      {q['question']}")
                lines.append(f"      📍 {q['file']}:{q['line']}")
        return [TextContent(type="text", text="\n".join(lines))]

    elif action == "quiz":
        n = arguments.get("n", 5)
        difficulty = arguments.get("difficulty")
        questions = get_quiz(n, difficulty)
        if not questions:
            return [TextContent(type="text", text="No questions yet. Run scan first to analyze code.")]
        lines = [f"Quiz time! {len(questions)} questions:\n"]
        for i, q in enumerate(questions, 1):
            lines.append(f"{'='*50}")
            lines.append(f"Q{i}. [{q['difficulty'].upper()}] {q['concept']}")
            lines.append(f"    {q['question']}")
            lines.append(f"    File: {q['file']}:{q['line']}")
            if q.get("context"):
                snippet = q["context"][:150].replace("\n", "\n    ")
                lines.append(f"    Code:\n    {snippet}")
            if q.get("answer"):
                lines.append(f"\n    Answer: {q['answer']}")
            lines.append("")
        return [TextContent(type="text", text="\n".join(lines))]

    elif action == "review":
        # Claude self-review: review recent Q&A to learn from own coding patterns
        review = get_session_review(arguments.get("n", 10))
        if not review:
            return [TextContent(type="text", text="No Q&A to review yet. Run scan first.")]
        lines = ["SESSION REVIEW — Coding patterns from recent work:\n"]
        for i, q in enumerate(review, 1):
            lines.append(f"[{q['difficulty']}] {q['concept']} ({q['file']}:{q['line']})")
            lines.append(f"  Q: {q['question']}")
            lines.append(f"  A: {q['answer']}")
            lines.append("")
        lines.append("Review these patterns. Apply them consistently in this session.")
        return [TextContent(type="text", text="\n".join(lines))]

    elif action == "progress":
        p = get_progress()
        lines = [
            "Learning Progress:",
            f"  Total questions generated: {p['total_questions']}",
            f"  Scans completed: {p['scans']}",
            f"  Files watched: {p['files_watched']}",
            f"  Last scan: {p['last_scan'][:19] if p['last_scan'] != 'never' else 'never'}",
            "\n  Difficulty breakdown:",
            f"    Beginner:     {p['difficulty_breakdown']['beginner']}",
            f"    Intermediate: {p['difficulty_breakdown']['intermediate']}",
            f"    Advanced:     {p['difficulty_breakdown']['advanced']}",
        ]
        if p["top_concepts"]:
            lines.append("\n  Top concepts in your code:")
            for c in p["top_concepts"][:10]:
                lines.append(f"    {c['concept']}: {c['total']} questions")
        return [TextContent(type="text", text="\n".join(lines))]

    elif action == "start":
        result = start_watcher()
        return [TextContent(type="text", text=f"Mentor watcher {result['status']}. Watching code for changes every 30s.")]

    elif action == "stop":
        result = stop_watcher()
        return [TextContent(type="text", text=f"Mentor watcher {result['status']}.")]

    return [TextContent(type="text", text=f"Unknown action: {action}")]


HANDLERS = {
    "watty_mentor": handle_mentor,
}
