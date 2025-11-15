@echo off
title Traffic Emission Calculator

cd /d "%~dp0"

echo ====================================
echo  Traffic Emission Calculator
echo ====================================
echo.

REM Try different Anaconda installation locations
set FOUND=0

REM Location 1: User profile anaconda3
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    echo [INFO] Found Anaconda at %USERPROFILE%\anaconda3
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
    set FOUND=1
    goto :run_app
)

REM Location 2: User AppData Local
if exist "%LOCALAPPDATA%\anaconda3\Scripts\activate.bat" (
    echo [INFO] Found Anaconda at %LOCALAPPDATA%\anaconda3
    call "%LOCALAPPDATA%\anaconda3\Scripts\activate.bat"
    set FOUND=1
    goto :run_app
)

REM Location 3: ProgramData (all users)
if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" (
    echo [INFO] Found Anaconda at C:\ProgramData\anaconda3
    call "C:\ProgramData\anaconda3\Scripts\activate.bat"
    set FOUND=1
    goto :run_app
)

REM Location 4: Miniconda in user profile
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    echo [INFO] Found Miniconda at %USERPROFILE%\miniconda3
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat"
    set FOUND=1
    goto :run_app
)

REM If not found
echo [ERROR] Anaconda/Miniconda not found in standard locations!
echo.
echo Please open Anaconda Prompt manually and run:
echo cd "%~dp0"
echo streamlit run Emmission_Cal_app.py
echo.
pause
exit /b 1

:run_app
echo.
echo Activating base environment...
call conda activate base

echo.
echo Starting Streamlit app...
echo Browser will open automatically
echo Keep this window open!
echo.

streamlit run Emmission_Cal_app.py

pause