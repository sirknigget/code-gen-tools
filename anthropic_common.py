from ai_common import *
import anthropic
import os
from dotenv import load_dotenv

MODEL_SONNET_3_7_LATEST = "claude-3-7-sonnet-latest"

load_dotenv(override=True)
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set")

claude = anthropic.Anthropic()

def chat(model, system, messages):
    message = claude.messages.create(
        model=model,
        system=system,
        messages=messages,
        max_tokens=2000
    )
    return message.content[0].text

def test_prompt():
    messages = add_user_message("Hello, Claude! This is a test prompt to check that you're working!", [])
    response = chat(MODEL_SONNET_3_7_LATEST, "You are an extremely kind LLM", messages)
    print(response)
