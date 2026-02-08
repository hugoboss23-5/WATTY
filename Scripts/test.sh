#!/bin/bash
# Watty — Run all tests
set -e

echo "Running Watty tests..."

swift test --parallel

echo ""
echo "All tests passed."
