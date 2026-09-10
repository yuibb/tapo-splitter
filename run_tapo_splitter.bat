@echo off
setlocal
cd /d "%~dp0"

set "PYTHON="
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not defined PYTHON if exist "%~dp0..\.c230-venv\Scripts\python.exe" set "PYTHON=%~dp0..\.c230-venv\Scripts\python.exe"
if not defined PYTHON where py >nul 2>&1 && set "PYTHON=py"
if not defined PYTHON where python >nul 2>&1 && set "PYTHON=python"

if not defined PYTHON (
    echo Python 3が見つかりません。READMEの手順でPythonをインストールしてください。
    pause
    exit /b 1
)

if "%~1"=="" (
    set "RUN_ARGS=--menu"
) else (
    set "RUN_ARGS=%*"
)

%PYTHON% run_tapo_splitter.py %RUN_ARGS%
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" (
    echo 処理が失敗しました（終了コード: %EXIT_CODE%）。
) else (
    echo 処理が完了しました。
)
pause
exit /b %EXIT_CODE%
