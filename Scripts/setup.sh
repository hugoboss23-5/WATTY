#!/bin/bash
# Watty — One-command project setup
set -e

echo "Setting up Watty..."

# Resolve dependencies
echo "Resolving Swift packages..."
swift package resolve

# Build
echo "Building..."
swift build

echo ""
echo "Watty is ready."
echo "Open in Xcode: open Package.swift"
echo ""
