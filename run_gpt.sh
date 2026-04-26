#!/bin/bash

# 1. Set your OpenAI API Key
export OPENAI_API_KEY="your-api-key-here"

# 2. Ensure results directory exists
mkdir -p results

# 3. Run the script
echo "------------------------------------------------"
echo "🚀 Initializing GPT-5.4 mini Inference..."
echo "------------------------------------------------"

python3 scripts/gpt_inference.py

echo "------------------------------------------------"
echo "Task Complete. Results saved in /results"
echo "------------------------------------------------"