@echo off
echo Setting up virtual environment for Deep-Unfolded-D-ADMM project...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.9+ and try again
    pause
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing required packages...
pip install -r requirements.txt

echo.
echo Virtual environment setup complete!
echo.
echo To activate the environment in the future, run:
echo     venv\Scripts\activate.bat
echo.
echo To deactivate the environment, run:
echo     deactivate
echo.
pause 