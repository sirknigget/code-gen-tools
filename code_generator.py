import gradio as gr

import ai_common
import openai_common
import anthropic_common

gpt_model = openai_common.MODEL_GPT_4O_MINI
claude_model = anthropic_common.MODEL_SONNET_3_7_LATEST

docs_system_prompt = """You are a coding assistant that adds documentation to provided code, meant to clarify the 
functionality and help new engineers understand it.
The documentation must be in the syntax of the source code's 
language.
Add both inline comments, and also function documentation.
The code should not be modified in any way.

Return only the documented code and nothing else."""

unit_tests_system_prompt = """You are a coding assistant that creates unit test suites for provided source code.
The tests must cover every public functionality exposed by the source code, test successful use-cases and also error or edge cases.
The tests must be in an appropriate language and framework for the provided code. A specific testing framework might be specified by the user.

Return only the unit test suite as a code file, and nothing else.
"""

def get_user_prompt(code, requests):
    return f"""The provided code is:
    {code}
    
    Extra requests from the user:
    {requests}
"""

def generate_code(code, requests, type, model):
    print(f"generate_code called with: code length {len(code)}, requests {requests}, type {type}, model {model}")

    system_prompt = ""
    user_prompt = get_user_prompt(code, requests)
    if type == "Docs":
        system_prompt = docs_system_prompt
    if type == "Unit tests":
        system_prompt = unit_tests_system_prompt

    messages = ai_common.add_user_message(user_prompt, [])

    model_name = ""
    chat_function = None
    if model == "GPT":
        chat_function = openai_common.chat
        model_name = gpt_model
    if model == "Claude":
        chat_function = anthropic_common.chat
        model_name = claude_model

    return chat_function(model_name, system_prompt, messages)


view = gr.Interface(
    fn=generate_code,
    inputs=[
        gr.Textbox(label="Code:"),
        gr.Textbox(label="Extra requests:"),
        gr.Dropdown(["Docs", "Unit tests"], label="Select code to generate"),
        gr.Dropdown(["GPT", "Claude"], label="Select model")],
    outputs=[gr.Markdown(label="Generated:")],
    flagging_mode="never"
)
view.launch()