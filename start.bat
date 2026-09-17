@echo off
rem One-click start on Windows. The first run installs the Python packages
rem (including PyTorch with CUDA) and builds the web page; later runs skip
rem straight to starting. The server listens on this computer only.
setlocal EnableExtensions
cd /d "%~dp0"

set "PORT=8000"
set "URL=http://127.0.0.1:%PORT%"
set "PY=.venv\Scripts\python.exe"
rem A copy of requirements.txt from the last successful install. When the file
rem changes (for example after a git pull), the packages are installed again.
set "INSTALLED=.venv\installed-requirements.txt"

echo rag-for-pandas
echo.

rem ---- 1. Python packages ----------------------------------------------------
if exist "%INSTALLED%" fc /b requirements.txt "%INSTALLED%" >nul 2>&1 && goto python_ready

py -3.12 -c "pass" >nul 2>&1 || goto no_python
echo Setup installs the Python packages, including PyTorch with CUDA support.
echo It downloads up to about 4 GB and takes several minutes. Later starts skip it.
set "ANSWER="
set /p "ANSWER=Continue? [Y/N] "
if /i not "%ANSWER%"=="Y" goto cancelled

if not exist "%PY%" (
    echo Creating the virtual environment...
    py -3.12 -m venv .venv || goto failed
)
"%PY%" -m pip install --disable-pip-version-check -r requirements.txt || goto failed
"%PY%" -m pip install --disable-pip-version-check -e . --no-deps || goto failed
copy /y requirements.txt "%INSTALLED%" >nul || goto failed
echo Python packages installed.
echo.
:python_ready

rem ---- 2. Web page ----------------------------------------------------------
if exist "frontend\dist\index.html" goto page_ready
where npm >nul 2>&1 || goto no_node
echo Building the web page...
pushd frontend
call npm ci || (popd & goto failed)
call npm run build || (popd & goto failed)
popd
echo.
:page_ready

rem ---- 3. Trained retriever and corpus (made by the pipeline, not by setup) --
if not exist "data\interim\docstrings.jsonl" goto no_pipeline
if not exist "models\retriever-minilm-ft-hn\model.safetensors" goto no_pipeline

rem ---- 4. Port -------------------------------------------------------------
netstat -ano -p tcp | findstr "LISTENING" | findstr /c:":%PORT% " >nul && goto port_busy

rem ---- 5. Answers need an NVIDIA GPU; search works without one ---------------
if defined RAG_FOR_PANDAS_GENERATOR goto generator_chosen
"%PY%" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
    set "RAG_FOR_PANDAS_GENERATOR=none"
    echo No NVIDIA GPU found, so answers are off. Search still works.
) else (
    echo NVIDIA GPU found, so answers are on. If the answer model is not downloaded
    echo yet, the first start downloads it, about 2.9 GB.
)
:generator_chosen

rem ---- 6. Start --------------------------------------------------------------
rem Opens the page in the browser once the server answers.
start "" /b powershell -NoProfile -Command "for ($i = 0; $i -lt 900; $i++) { try { Invoke-WebRequest -UseBasicParsing '%URL%/health' -TimeoutSec 2 | Out-Null; Start-Process '%URL%/'; break } catch { Start-Sleep -Seconds 1 } }"

echo.
echo Starting at %URL%/  (only this computer can open it)
echo Close this window to stop the server.
echo.
"%PY%" -m uvicorn rag_for_pandas.api:app --host 127.0.0.1 --port %PORT%
echo.
echo The server stopped.
pause
exit /b 0

rem ---- Problems ---------------------------------------------------------------
:no_python
echo Python 3.12 was not found. Install it from https://www.python.org/downloads/
echo with the "py launcher" option, then run start.bat again.
goto stop

:no_node
echo Node.js was not found, so the web page cannot be built. Install Node.js 24 from
echo https://nodejs.org/, then run start.bat again.
goto stop

:no_pipeline
echo The trained retriever or the corpus is missing: models\retriever-minilm-ft-hn
echo and data\interim\docstrings.jsonl are made by the pipeline, not by setup.
echo See "Reproducing the pipeline" in README.md, then run start.bat again.
goto stop

:port_busy
echo Port %PORT% is already in use, perhaps by another copy of the server.
echo Close that program, then run start.bat again.
goto stop

:cancelled
echo Setup cancelled. Nothing was installed.
goto stop

:failed
echo.
echo Setup failed; the messages above say why. Running start.bat again retries it.
goto stop

:stop
echo.
pause
exit /b 1
