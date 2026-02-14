# Contributing to Watty

Thanks for wanting to help give AI a brain. Here's how.

## Quick start

```bash
git clone https://github.com/watty-ai/watty.git
cd watty
pip install -e .
```

## How to contribute

1. **Find something to work on** — check [Issues](https://github.com/watty-ai/watty/issues) for `good first issue` labels
2. **Fork the repo** and create a branch from `main`
3. **Make your changes** — keep it focused, one thing per PR
4. **Test it** — make sure `watty` still starts and the 8 tools work
5. **Open a PR** — describe what you changed and why

## What we'd love help with

- **New file type support** — teach Watty to eat more formats (PDF, DOCX, etc.)
- **Better clustering** — the current algorithm is simple. Make it smarter.
- **Performance** — faster search, better indexing, lower memory usage
- **Platform support** — make sure Watty works everywhere (Windows, Mac, Linux)
- **Documentation** — tutorials, guides, examples
- **Integrations** — connect Watty to more AI platforms beyond MCP

## Code style

- Keep it simple. Watty is ~850 lines of logic for a reason.
- No unnecessary dependencies. Every `import` must earn its place.
- Comments explain *why*, not *what*.
- If you can't explain your change in one sentence, it might be too big.

## Philosophy

Watty follows one rule: **depth over length**. A 50-line solution that's elegant beats a 500-line solution that's thorough. If you're adding code, ask yourself: can this be shorter without losing clarity?

## Questions?

Open an issue. We're friendly.
