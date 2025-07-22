import ollama
from ai_common import *

# Constants
HEADERS = {"Content-Type": "application/json"}
MODEL = "llama3.2"


def chat(messages):
    response = ollama.chat(model=MODEL, messages=messages)
    return response['message']['content']


def test_chat():
    messages = create_user_message("", "Describe some of the business applications of Generative AI")
    print(chat(messages))
