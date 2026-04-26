@echo off
title GPT Inference Runner

:: 1. Set your OpenAI API Key
set OPENAI_API_KEY=your-api-key-here

:: 2. Ensure results directory exists
if not exist results mkdir results

:: 3. Run the script
echo ------------------------------------------------
echo ^> Initializing GPT-5.4 mini Inference...
echo ------------------------------------------------

python scripts/gpt_inference.py

echo ------------------------------------------------
echo ^> Task Complete. Results saved in \results
echo ------------------------------------------------
pause