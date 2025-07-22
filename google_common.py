from ai_common import *
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

MODEL_GEMINI_2_FLASH = "gemini-2.0-flash"

load_dotenv(override=True)
google_api_key = os.getenv('GOOGLE_API_KEY')
if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:8]}")
else:
    print("Google API Key not set")

client = genai.Client(api_key=google_api_key)


def chat(model, system, messages):
    contents = []
    for message in messages:
        type = types.UserContent if (message['role'] == 'user') else types.ModelContent
        contents.append(type(parts=[
            types.Part.from_text(text=message['content'])
        ]))
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
        ),
    )

    return response.text


def test_prompt():
    messages = add_user_message("Hello, Claude! This is a test prompt to check that you're working!", [])
    response = chat(MODEL_GEMINI_2_FLASH, "You are a very generous LLM", messages)
    print(response)
