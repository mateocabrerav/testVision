@echo off
echo ========================================
echo Installing Gemini Dependencies
echo ========================================
echo.

cd /d "%~dp0"

echo Installing core dependencies...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install google-generativeai pillow opencv-python python-dotenv numpy

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Make sure .env file has: GEMINI_API_KEY=your_key_here
echo 2. Rename test\gemini\test_image.png to test\gemini\tst_image.png
echo 3. Run: .venv\Scripts\pytest.exe test\gemini\test_gemini_vision_system.py -v
echo.
pause

