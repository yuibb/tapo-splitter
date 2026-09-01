@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>&1
if %errorlevel%==0 (
    py run_tapo_splitter.py %*
) else (
    where python >nul 2>&1
    if %errorlevel%==0 (
        python run_tapo_splitter.py %*
    ) else (
        echo Python 3が見つかりません。READMEの手順でPythonをインストールしてください。
        pause
        exit /b 1
    )
)

if not %errorlevel%==0 (
    echo.
    echo 処理が失敗しました。
    pause
    exit /b 1
)
echo.
echo 処理が完了しました。
pause
