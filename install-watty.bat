@echo off
setlocal enabledelayedexpansion
title Watty Installer

echo.
echo   =============================================
echo           WATTY ONE-CLICK INSTALLER
echo     Your brain's external hard drive.
echo   =============================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   Python not found. Installing...
    echo   Downloading Python from python.org...
    curl -L -o "%TEMP%\python-installer.exe" https://www.python.org/ftp/python/3.12.2/python-3.12.2-amd64.exe
    if exist "%TEMP%\python-installer.exe" (
        echo   Running Python installer...
        "%TEMP%\python-installer.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1
        del "%TEMP%\python-installer.exe"
        echo   Python installed. You may need to restart this script.
        echo   Press any key to continue anyway...
        pause >nul
    ) else (
        echo   ERROR: Could not download Python.
        echo   Please install Python 3.10+ from https://python.org
        pause
        exit /b 1
    )
)

:: Show Python version
for /f "tokens=*" %%a in ('python --version 2^>^&1') do echo   Found: %%a

:: Check pip
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   pip not found. Installing...
    python -m ensurepip --upgrade
)

:: Install Watty
echo.
echo   Installing Watty...
echo.
python -m pip install --upgrade pip >nul 2>&1

:: Check if we're in the watty repo (has pyproject.toml)
if exist "pyproject.toml" (
    echo   Installing from local source...
    python -m pip install -e .
) else (
    echo   Installing from PyPI...
    python -m pip install watty
)

:: Verify
echo.
watty version >nul 2>&1
if %errorlevel% neq 0 (
    echo   WARNING: 'watty' command not found on PATH.
    echo   Try: python -m watty version
    echo.
) else (
    for /f "tokens=*" %%a in ('watty version 2^>^&1') do echo   Installed: %%a
)

:: Run setup
echo.
echo   Running first-time setup...
echo   This will scan your Documents, Desktop, and Downloads.
echo.

watty setup 2>nul || python -m watty setup

echo.
echo   =============================================
echo           INSTALLATION COMPLETE
echo   =============================================
echo.
echo   Commands:
echo     watty recall "search query"   Search your memory
echo     watty stats                   Check brain health
echo     watty daemon start            Start autonomous daemon
echo     watty serve                   Start MCP server
echo.
echo   Watty is now available in Claude Desktop.
echo.
pause
