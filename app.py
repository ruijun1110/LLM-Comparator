import os
import streamlit as st
import math
import pathlib
import asyncio
import httpx
import threading
import queue
import json

################ Functions & Constants ###############

def load_css(file_path):
    """Load CSS file."""
    with open(file_path) as f:
        st.html(f"<style>{f.read()}</style>")

@st.dialog("🎉 Congratulations 🎉")
def final_modal_dialog(model_name, input_prompt, time_taken, tokens_used, model_response):
    """Display the final modal dialog."""
    st.html("<h3 style='text-align:center;'>You have found the best model for your task!</h3>")
    st.html(f"<h1 style='text-align:center; font-size:2em; color: yellow;'>{model_name}</h1>")
    st.write("**Input Prompt:**")
    prompt_container = st.container(border=False, key="final-prompt-container")
    prompt_container.write(input_prompt)
    st.write("**Model Response:**")
    response_container = st.container(border=False, key="final-response-container")
    response_container.write(model_response)
    st.write("**Time Taken:**", time_taken)
    st.write("**Tokens Used:**", tokens_used)
    st.balloons()

def toggle_all_models():
    """Toggle all model checkboxes based on the current state."""
    # Flip the state
    st.session_state["all_models_selected"] = not st.session_state["all_models_selected"]
    
    # Set all checkboxes according to the new state
    for model_key in models_options.keys():
        st.session_state[f"{model_key}-checkbox"] = st.session_state["all_models_selected"]

    if st.session_state["all_models_selected"]:
        st.session_state["num_selected_models"] = len(models_options)
    else:
        st.session_state["num_selected_models"] = 0

def update_selected_models_count():
    """Update the count of selected models in session state."""
    count = sum(st.session_state.get(f"{key}-checkbox", False) for key in models_options.keys())
    st.session_state["num_selected_models"] = count

def update_all_selected_state():
    """Update the state of 'all_models_selected' based on individual checkboxes. """
    all_selected = all(st.session_state.get(f"{key}-checkbox", False) for key in models_options.keys())
    st.session_state["all_models_selected"] = all_selected
    
#  Function to get selected model names
def get_selected_model_names():
    """Return a list of selected model names."""
    selected_models = []
    for model_key, model_name in models_options.items():
        if st.session_state.get(f"{model_key}-checkbox", False):
            selected_models.append(model_key)
    return selected_models

def check_api_keys_for_selected_models():
    """
    Check if the user has provided API keys for all selected models.
    
    Returns:
        dict: A dictionary with model names as keys and boolean values indicating
              whether the required API key is available.
    """
    # Map models to their required API keys
    model_to_api_key = {
        # "gpt-3-5": "open_ai_api_key",
        # "gpt-4o": "open_ai_api_key",
        # "o1-preview": "open_ai_api_key",
        # "o4-mini-preview": "open_ai_api_key",
        # "gemini-2-0-flash": "google_gemini_api_key",
        "gemini-2-5-pro": "openrouter_api_key",
        "DeepSeek-V3": "openrouter_api_key",
        "Qwen3": "openrouter_api_key"
    }
    
    # Initialize result dictionary
    api_key_status = {}
    
    # Check each selected model
    for model_key, _ in models_options.items():
        if st.session_state.get(f"{model_key}-checkbox", False):
            # Get the required API key for this model
            required_key = model_to_api_key.get(model_key)
            
            # Check if the required API key is in session state and not empty
            has_key = (
                required_key is not None and 
                required_key in st.session_state and 
                st.session_state[required_key]
            )
            
            # Store the result
            api_key_status[model_key] = has_key
    
    return api_key_status

def has_num_selected_models_error():
    """Check error for missing API keys for selected models."""
    if st.session_state.num_selected_models < 2:
        return True
    return False

def has_api_key_error():
    """Check error for missing API keys for selected models."""
    api_key_status = check_api_keys_for_selected_models()
    
    # Check if any selected model is missing its API key
    missing_keys = [model for model, has_key in api_key_status.items() if not has_key]
    
    if missing_keys:
        missing_models = ", ".join(missing_keys)
        return True, missing_models
    
    return False, ""

def update_api_key(key_name, key_value):
    """Update the API key for a given model."""
    st.session_state[key_name] = key_value
    has_api_key_error()

def stream_openai_response(url, prompt, api_key, model, temperature=0.5, max_tokens=1000):
    """Stream OpenAI response asynchronously, compatible with Streamlit write_stream. Only yield the 'content' field."""
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    async def fetch():
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                print(response)
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            delta = json.loads(data)
                            # OpenAI stream response: {"choices":[{"delta":{"content":"..."}}]}
                            content = delta.get("choices", [{}])[0].get("delta", {}).get("content")
                            if content:
                                yield content
                        except Exception:
                            continue
    q = queue.Queue()
    sentinel = object()
    def runner():
        async def consume():
            async for chunk in fetch():
                q.put(chunk)
            q.put(sentinel)
        asyncio.run(consume())
    threading.Thread(target=runner, daemon=True).start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        yield item

# List of models available for comparison
# Key: model class name
# Value: model id for the API
models_options = {
    "gpt-3-5": "gpt-3.5-turbo-0125",
    "gpt-4o": "GPT-4o",
    "o1-preview": "o1 (Preview)",
    "o4-mini-preview": "o4 mini (Preview)",
    "gemini-2-0-flash": "Gemini 2.0 Flash",
    "gemini-2-5-pro": "gemini-2.5-pro-exp-03-25",
    "Qwen3": "qwen/qwen3-235b-a22b:free",
    "DeepSeek-V3": "deepseek/deepseek-chat-v3-0324:free",
}

# Key: model class name
# Value: API endpoint
models_endpoints = {
    "gpt-3-5": "https://api.openai.com/v1/chat/completions",
    "gpt-4o": "https://api.openai.com/v1/chat/completions",
    "o1-preview": "https://api.openai.com/v1/chat/completions",
    "o4-mini-preview": "https://api.openai.com/v1/chat/completions",
    "gemini-2-0-flash": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    "gemini-2-5-pro": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    "Qwen3": "https://openrouter.ai/api/v1/chat/completions",
    "DeepSeek-V3": "https://openrouter.ai/api/v1/chat/completions",
}

# Key: model class name
# Value: API key name
models_api_keys = {
    "gpt-3-5": "open_ai_api_key",
    "gpt-4o": "open_ai_api_key",
    "o1-preview": "open_ai_api_key",
    "o4-mini-preview": "open_ai_api_key",
    "gemini-2-0-flash": "google_gemini_api_key",
    "gemini-2-5-pro": "google_gemini_api_key",
    "Qwen3": "openrouter_api_key",
    "DeepSeek-V3": "openrouter_api_key",
}

# Default selected models
default_selected = ["DeepSeek-V3", "Qwen3"]

# Initialize session states
if "all_models_selected" not in st.session_state:
    st.session_state["all_models_selected"] = False

if "num_selected_models" not in st.session_state:
    st.session_state["num_selected_models"] = len(default_selected)

if "current_displayed_models" not in st.session_state:
    st.session_state["current_displayed_models"] = []



################ Streamlit Configuration ################

css_path = pathlib.Path(__file__).parent.parent / "assets" / "style.css"

# Load CSS
load_css(css_path)

# Page title
st.html("<h1 style='text-align:center; font-size:2em;'>LLM Model Comparison</h1>")

###### Sidebar Configuration ######

#### Config sidebar #####
config_sidebar = st.sidebar

config_sidebar.subheader("Basic Configuration", divider=True)
# Temperature slider
temperature = config_sidebar.slider("Temperature", 0.0, 1.0, 0.5, key="temperature_slider")
if temperature:
    st.session_state.temperature = temperature
# Max tokens slider
max_token = config_sidebar.slider("Max Tokens", 1, 4000, 1000, key="max_tokens_slider")
if max_token:
    st.session_state.max_token = max_token

## Model Selection Section ###
config_sidebar.subheader("Models to compare", divider=True)

# Environment Variables Expander
env_expander = config_sidebar.expander("Environment Variables", expanded=False)
open_ai_api_key = env_expander.text_input("OpenAI API Key", type="password", on_change=has_api_key_error)
update_api_key("open_ai_api_key", open_ai_api_key)

anthropic_api_key = env_expander.text_input("Anthropic API Key", type="password", on_change=has_api_key_error)
update_api_key("anthropic_api_key", anthropic_api_key)

google_gemini_api_key = env_expander.text_input("Google Gemini API Key", type="password", on_change=has_api_key_error)
update_api_key("google_gemini_api_key", google_gemini_api_key)

mistral_api_key = env_expander.text_input("Mistral API Key", type="password", on_change=has_api_key_error)
update_api_key("mistral_api_key", mistral_api_key)

grok_api_key = env_expander.text_input("Grok API Key", type="password", on_change=has_api_key_error)
update_api_key("grok_api_key", grok_api_key)

openrouter_api_key = env_expander.text_input("OpenRouter API Key", type="password", on_change=has_api_key_error)
update_api_key("openrouter_api_key", openrouter_api_key)

# Model Selection Checkboxes
for model_key, model_name in models_options.items():
    is_default = model_key in default_selected
    config_sidebar.checkbox(model_key, value=is_default, key=f"{model_key}-checkbox", on_change=update_selected_models_count)

# Display either "Select All" or "Unselect All" button based on current state
button_text = "Unselect All" if st.session_state["all_models_selected"] else "Select All"
config_sidebar.button(button_text, key="select_all_button", use_container_width=True, on_click=toggle_all_models)

# Update the all_selected state based on current checkboxes
update_all_selected_state()



############# Main Page Configuration ###########
chat_input_container = st.container(border=False)

enough_models_error = has_num_selected_models_error()
required_api_keys_error, missing_models = has_api_key_error()

# Chat input section
prompt = chat_input_container.chat_input(
    placeholder="Enter your test prompt here...", 
    accept_file=True, key="prompt_input", 
    file_type=["jpg", "jpeg", "png", "txt", "pdf", "docx", "csv"], 
    disabled=enough_models_error or required_api_keys_error
    )

if prompt:
    st.session_state.prompt = prompt
    st.session_state["current_displayed_models"] = get_selected_model_names()

if enough_models_error:
    st.error("Please select at least two models to compare.")
if required_api_keys_error:
    st.error(f"Missing API keys for: {missing_models}. Please provide them in the Environment Variables section.")

############ Model Response Section #############
# Only display if a prompt is provided
if "prompt" in st.session_state and "current_displayed_models" in st.session_state:

    response_container = st.container(border=False)
    response_container_header = response_container.container(border=False, key="response-container-header")
    response_container_header.html("<p style='font-size: 1.5rem; font-weight: bold;'>Model Responses</p>")
    response_container_header.html("<p style='color: gray;'>Click on the model you want to select as the best.</p>")
    
    # Get selected model names
    selected_models = st.session_state["current_displayed_models"]

    # Calculate number of rows needed for the model response cards
    num_rows = math.ceil(len(selected_models) / 3)

    # Create rows and model response cards
    for row_idx in range(num_rows):
        # Create a row with 3 columns
        cols = response_container.columns(3)
        # Add cards to this row
        for col_idx in range(3):
            # Check if the column index is within the range of selected models
            card_index = row_idx * 3 + col_idx
            if card_index < len(selected_models):
                with cols[col_idx]:
                    current_model = selected_models[card_index]
                    model_card_key = f"model-card-{selected_models[card_index]}"

                    # Create a card for the model
                    model_card = st.container(key=model_card_key, border=False)
                    with model_card:
                        model_response_header = st.container(border=False, key=f"model-response-header-{model_card_key}")
                        # Display the model name
                        model_response_header.write(f"{selected_models[card_index]}")
                        # Placeholder for model response time
                        # model_response_header.write("2.5s")
                        print(current_model)
                        url = models_endpoints[current_model]
                        api_key = st.session_state.get(models_api_keys[current_model], "")
                        if api_key == "":
                            st.error(f"Missing API key for {current_model}. Please provide it in the Environment Variables section.")
                        model_id = models_options[current_model]
                        prompt = st.session_state.prompt.text
                        st.write_stream(
                            stream_openai_response(
                                url=url,
                                prompt=prompt,
                                api_key=api_key,
                                model=model_id,
                                temperature=st.session_state.get("temperature", 0.5),
                                max_tokens=st.session_state.get("max_token", 1000)
                            )
                        )

                        # Model response footer
                        model_response_footer = st.container(border=False, key=f"model-response-footer-{model_card_key}")
                        # Placeholder for tokens used
                        # model_response_footer.write("1293 tokens")
                        # Button to select the model as the best
                        if model_response_footer.button("Select", key=f"select-button-{model_card_key}"):
                            # Display the final modal dialog with the selected model
                            final_modal_dialog(current_model, st.session_state.prompt.text, "? s", "? tokens", "This is a test response")
                            