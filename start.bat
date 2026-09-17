@echo off
rem One-click start on Windows. The first run installs the Python packages
rem (including PyTorch with CUDA) and builds the web page; later runs skip
rem straight to starting. The server listens on this computer only.
rem Delayed expansion (!ANSWER!) reads typed answers safely, even ones containing quotes.
setlocal EnableExtensions EnableDelayedExpansion
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
if /i not "!ANSWER!"=="Y" goto cancelled

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
rem Built on every start, so changes from a git pull always reach the page. Its
rem packages are installed again only when package-lock.json changes.
set "PAGE_PACKAGES=frontend\node_modules\installed-package-lock.json"
where npm >nul 2>&1 || goto page_without_node
if exist "%PAGE_PACKAGES%" fc /b frontend\package-lock.json "%PAGE_PACKAGES%" >nul 2>&1 && goto page_build
echo Installing the web page's packages...
pushd frontend
call npm ci || (popd & goto failed)
popd
copy /y frontend\package-lock.json "%PAGE_PACKAGES%" >nul || goto failed
:page_build
echo Building the web page...
pushd frontend
call npm run build --silent >nul || (popd & goto page_build_failed)
popd
goto page_ready

:page_without_node
if not exist "frontend\dist\index.html" goto no_node
echo Node.js was not found, so the web page built earlier is used.
:page_ready

rem ---- 3. Trained retriever and corpus (made by the pipeline, not by setup) --
if not exist "data\interim\docstrings.jsonl" goto no_pipeline
if not exist "models\retriever-minilm-ft-hn\model.safetensors" goto no_pipeline

rem ---- 4. Port -------------------------------------------------------------
netstat -ano -p tcp | findstr "LISTENING" | findstr /c:":%PORT% " >nul && goto port_busy

rem ---- 5. Answers need an NVIDIA GPU and the answer model; search needs neither -
if defined RAG_FOR_PANDAS_GENERATOR goto generator_chosen
rem Exit code 0: GPU and model ready; 1: no GPU; 2: GPU, but the model is not downloaded.
"%PY%" -c "import sys, torch; from huggingface_hub import try_to_load_from_cache; from rag_for_pandas.generation import LOCAL_MODEL; sys.exit(1 if not torch.cuda.is_available() else 0 if isinstance(try_to_load_from_cache(LOCAL_MODEL, 'model.safetensors'), str) else 2)" >nul 2>&1
set "READY=%errorlevel%"
if "%READY%"=="0" goto answers_on
if "%READY%"=="2" goto ask_model
set "RAG_FOR_PANDAS_GENERATOR=none"
echo No NVIDIA GPU found, so answers are off. Search still works.
goto generator_chosen

:ask_model
echo An NVIDIA GPU was found. Answers need the answer model, a one-time download of
echo about 2.9 GB. Without it, search still works.
set "ANSWER="
set /p "ANSWER=Download it on this start? [Y/N] "
if /i "!ANSWER!"=="Y" goto answers_on
set "RAG_FOR_PANDAS_GENERATOR=none"
echo Answers are off for this start.
goto generator_chosen

:answers_on
echo NVIDIA GPU found, so answers are on.
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

:page_build_failed
echo The web page did not build. Run "npm run build" in the frontend folder to see why.
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
