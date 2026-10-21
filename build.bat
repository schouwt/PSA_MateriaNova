@echo off
rem ===========================================================================
rem  Build of the Particle Size Analyser (PSA) Windows executable
rem  Materia Nova - CHIPS
rem
rem  Usage: place this file in the folder of psa_app.py, then
rem         double click it, or in a console:
rem             build.bat            folder + exe (recommended)
rem             build.bat onefile    single .exe (slower start)
rem             build.bat console    keeps a console (diagnostics)
rem  Both options can be combined: build.bat onefile console
rem ===========================================================================
setlocal EnableExtensions
title Build of the PSA executable
cd /d "%~dp0"

rem A PYTHONPATH left by other software (e.g. ChemOffice / ChemScript)
rem would bundle foreign modules: it is ignored for this session.
rem (no parenthesised block: the value may contain "(x86)")
if defined PYTHONPATH echo [INFO] PYTHONPATH ignored during the build.
set "PYTHONPATH="

set "NOM=PSA_MateriaNova"
set "SCRIPT=psa_app.py"
set "MODE=--onedir"
set "FENETRE=--windowed"
set "DL=--exclude-module tensorflow --exclude-module stardist --exclude-module csbdeep --exclude-module keras --exclude-module tf_keras --exclude-module numba --exclude-module llvmlite --exclude-module torch --exclude-module torchvision --exclude-module cellpose --exclude-module napari"
for %%A in (%*) do (
    if /i "%%~A"=="onefile" set "MODE=--onefile"
    if /i "%%~A"=="console" set "FENETRE=--console"
)

echo.
echo  ==== Building %NOM% ====
echo  Folder: %CD%
echo.

rem --- 1. Source files ---------------------------------------------------------
if not exist "%SCRIPT%" (
    echo [ERROR] %SCRIPT% not found in this folder.
    goto :failed
)
if not exist "psa_nano_meb.py" echo [WARNING] psa_nano_meb.py missing: SEM and nano modes unavailable.
if not exist "psa_fibres.py"   echo [WARNING] psa_fibres.py missing: fibre mode unavailable.
if not exist "psa\__init__.py" (
    echo [ERROR] psa\ package folder missing next to %SCRIPT%.
    goto :failed
)

rem --- 2. Python: project virtual environment first ----------------------------
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY if exist "venv\Scripts\python.exe" set "PY=venv\Scripts\python.exe"
rem Then a Python version for which every package exists PRECOMPILED (a too
rem recent Python, e.g. 3.15, would need a C compiler to build scikit-image).
if not defined PY (
    where py >nul 2>&1 && for %%V in (3.13 3.12 3.14 3.11) do if not defined PY (py -%%V -c "pass" >nul 2>&1 && set "PY=py -%%V")
)
if not defined PY (where py >nul 2>&1 && set "PY=py -3")
if not defined PY (where python >nul 2>&1 && set "PY=python")
if not defined PY (
    echo [ERROR] Python not found. Install Python 3.12 or 3.13 from python.org.
    goto :failed
)
echo  Python used: %PY%
%PY% --version || goto :failed
%PY% -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 14) else 3)" >nul 2>&1
if errorlevel 3 echo [WARNING] This Python version is not supported by every package: Python 3.12 or 3.13 is recommended.

rem --- 3. Dependencies (installed only when missing) ---------------------------
echo.
echo  [1/4] Checking the dependencies...
%PY% -c "import numpy, cv2, scipy, skimage, matplotlib, ttkbootstrap, tifffile, PIL, onnxruntime" >nul 2>&1
if errorlevel 1 (
    echo        Installing the missing dependencies...
    %PY% -m pip install --only-binary=:all: numpy opencv-python scipy scikit-image matplotlib ttkbootstrap tifffile roifile pillow onnxruntime || goto :nowheel
)
%PY% -m pip install --only-binary=:all: --upgrade pyinstaller >nul || goto :nowheel
for /f "delims=" %%V in ('%PY% -m PyInstaller --version') do echo        PyInstaller %%V

rem --- 4. Optional resources ---------------------------------------------------
rem ABSOLUTE paths: with --specpath build, PyInstaller resolves relative paths
rem from the build\ folder and would not find the files.
set "ICONE="
if exist "psa.ico" set "ICONE=--icon "%CD%\psa.ico""
set "DONNEES="
if exist "guide_psa.pdf" set "DONNEES=--add-data "%CD%\guide_psa.pdf;.""
set "MODELES="
if exist "Models" set "MODELES=--add-data "%CD%\Models;Models""
set "ICONE_PNG="
if exist "psa.png" set "ICONE_PNG=--add-data "%CD%\psa.png;.""

rem --- 5. Build ----------------------------------------------------------------
echo.
echo  [2/4] Building (%MODE% %FENETRE%), please wait 2 to 5 minutes...
echo        Full details in build_exe.log
%PY% -m PyInstaller --noconfirm --clean %MODE% %FENETRE% --name "%NOM%" %ICONE% %DONNEES% %ICONE_PNG% %MODELES% %DL% ^
    --specpath build --paths "%CD%" ^
    --hidden-import psa_nano_meb --hidden-import psa_fibres ^
    --collect-submodules psa --hidden-import onnxruntime ^
    --hidden-import PIL._tkinter_finder ^
    --collect-submodules skimage --collect-data skimage ^
    --collect-data ttkbootstrap ^
    --exclude-module PyQt5 --exclude-module PyQt6 --exclude-module PySide6 ^
    --exclude-module IPython --exclude-module pytest ^
    --exclude-module pandas --exclude-module sklearn ^
    --exclude-module imageio_ffmpeg --exclude-module lxml ^
    "%SCRIPT%" > build_exe.log 2>&1
if errorlevel 1 (
    echo [ERROR] The build failed. Last lines of the log:
    powershell -NoProfile -Command "Get-Content build_exe.log -Tail 25"
    goto :failed
)

if /i "%MODE%"=="--onefile" (
    set "SORTIE=dist"
    set "EXE=dist\%NOM%.exe"
) else (
    set "SORTIE=dist\%NOM%"
    set "EXE=dist\%NOM%\%NOM%.exe"
)
if not exist "%EXE%" (
    echo [ERROR] Executable not found: %EXE%
    goto :failed
)

rem --- 6. Companion files next to the exe ---------------------------------------
echo.
echo  [3/4] Copying the companion files...
if exist "guide_psa.pdf"          copy /y "guide_psa.pdf"          "%SORTIE%\" >nul
if exist "README.txt"             copy /y "README.txt"             "%SORTIE%\" >nul
if exist "THIRD_PARTY_LICENSES.txt" copy /y "THIRD_PARTY_LICENSES.txt" "%SORTIE%\" >nul
if exist "Assets" xcopy /e /i /y /q "Assets" "%SORTIE%\Assets" >nul

rem --- 7. Self-test of the built executable ------------------------------------
echo.
echo  [4/4] Self-test of the executable (modules, analyses, GUI)...
set "RAPPORT=%TEMP%\psa_autotest.txt"
if exist "%RAPPORT%" del "%RAPPORT%"
start "" /wait "%EXE%" --autotest "%RAPPORT%"
set "CODE=%ERRORLEVEL%"
if exist "%RAPPORT%" (
    type "%RAPPORT%"
) else (
    echo [ERROR] No self-test report: the executable did not start.
    echo         Run again with the console option to see the error.
    goto :failed
)
if not "%CODE%"=="0" (
    echo [ERROR] The self-test failed: see the report above.
    goto :failed
)

echo.
echo  ==== DONE ====
echo  Executable: %CD%\%EXE%
if /i "%MODE%"=="--onedir" echo  To distribute: the whole folder %SORTIE% (copy the complete folder).
echo.
start "" explorer "%SORTIE%"
pause
exit /b 0

:nowheel
echo.
echo [ERROR] Some packages are not available precompiled for this Python version,
echo         and building them would need a C compiler.
echo         Install Python 3.12 or 3.13 from python.org, keeping the "py launcher"
echo         option, then run build.bat again: it is chosen automatically.
echo         Simpler for colleagues: copy the dist\PSA_MateriaNova folder built on
echo         another PC; the executable needs no Python at all.
goto :failed

:failed
echo.
echo  ==== FAILED ====
pause
exit /b 1
