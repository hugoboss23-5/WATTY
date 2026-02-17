#!/usr/bin/env bash
set -e

echo ""
echo "  ============================================="
echo "          WATTY ONE-CLICK INSTALLER"
echo "    Your brain's external hard drive."
echo "  ============================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check Python
if command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo -e "  ${RED}Python not found.${NC}"
    echo ""
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  Install with Homebrew:"
        echo "    brew install python"
    else
        echo "  Install with your package manager:"
        echo "    sudo apt install python3 python3-pip   # Ubuntu/Debian"
        echo "    sudo dnf install python3 python3-pip   # Fedora"
        echo "    sudo pacman -S python python-pip       # Arch"
    fi
    echo ""
    exit 1
fi

echo -e "  Found: $($PY --version)"

# Check pip
if ! $PY -m pip --version &>/dev/null; then
    echo "  pip not found. Installing..."
    $PY -m ensurepip --upgrade 2>/dev/null || {
        echo -e "  ${RED}Could not install pip.${NC}"
        echo "  Try: sudo apt install python3-pip"
        exit 1
    }
fi

# Install Watty
echo ""
echo "  Installing Watty..."
echo ""

$PY -m pip install --upgrade pip -q 2>/dev/null

if [ -f "pyproject.toml" ]; then
    echo "  Installing from local source..."
    $PY -m pip install -e .
else
    echo "  Installing from PyPI..."
    $PY -m pip install watty
fi

# Verify
echo ""
if command -v watty &>/dev/null; then
    echo -e "  ${GREEN}$(watty version)${NC}"
else
    echo -e "  ${YELLOW}WARNING: 'watty' not on PATH. Try: $PY -m watty version${NC}"
fi

# Run setup
echo ""
echo "  Running first-time setup..."
echo "  This will scan your Documents, Desktop, and Downloads."
echo ""

watty setup 2>/dev/null || $PY -m watty setup

echo ""
echo -e "  ${GREEN}=============================================${NC}"
echo -e "  ${GREEN}        INSTALLATION COMPLETE${NC}"
echo -e "  ${GREEN}=============================================${NC}"
echo ""
echo "  Commands:"
echo "    watty recall \"search query\"   Search your memory"
echo "    watty stats                   Check brain health"
echo "    watty daemon start            Start autonomous daemon"
echo "    watty serve                   Start MCP server"
echo ""
echo "  Watty is now available in Claude Desktop."
echo ""
