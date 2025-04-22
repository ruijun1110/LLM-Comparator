import streamlit as st
import math
import pathlib

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
        st.session_state[f"{model_key}_checkbox"] = st.session_state["all_models_selected"]

def update_all_selected_state():
    """Update the state of 'all_models_selected' based on individual checkboxes. """
    all_selected = all(st.session_state.get(f"{key}_checkbox", False) for key in models_options.keys())
    st.session_state["all_models_selected"] = all_selected

#  Function to get selected model names
def get_selected_model_names():
    """Return a list of selected model names."""
    selected_models = []
    for model_key, model_name in models_options.items():
        if st.session_state.get(f"{model_key}_checkbox", False):
            selected_models.append(model_name)
    return selected_models

# List of models available for comparison
# Key: model class name
# Value: model display name
models_options = {
    "gpt-3-5": "GPT-3.5",
    "gpt-4o": "GPT-4o",
    "o1-preview": "o1 (Preview)",
    "o4-mini-preview": "o4 mini (Preview)",
    "gemini-2-0-flash": "Gemini 2.0 Flash",
    "gemini-2-5-pro": "Gemini 2.5 Pro",
    "claude-sonnet-3-5": "Claude Sonnet 3.5",
    "claude-sonnet-3-7": "Claude Sonnet 3.7",
    "grok-2": "Grok 2",
    "mistral-7b": "Mistral 7B"
}
# Default selected models
default_selected = ["gpt-3-5", "gpt-4o", "gemini-2-5-pro", "claude-sonnet-3-7"]

# Initialize session states
if "all_models_selected" not in st.session_state:
    st.session_state["all_models_selected"] = False




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
open_ai_api_key = env_expander.text_input("OpenAI API Key", type="password")
if open_ai_api_key:
    st.session_state.open_ai_api_key = open_ai_api_key

anthropic_api_key = env_expander.text_input("Anthropic API Key", type="password")
if anthropic_api_key:
    st.session_state.anthropic_api_key = anthropic_api_key

google_gemini_api_key = env_expander.text_input("Google Gemini API Key", type="password")
if google_gemini_api_key:
    st.session_state.google_gemini_api_key = google_gemini_api_key

mistral_api_key = env_expander.text_input("Mistral API Key", type="password")
if mistral_api_key:
    st.session_state.mistral_api_key = mistral_api_key

grok_api_key = env_expander.text_input("Grok API Key", type="password")
if grok_api_key:
    st.session_state.grok_api_key = grok_api_key

# Model Selection Checkboxes
for model_key, model_name in models_options.items():
    is_default = model_key in default_selected
    config_sidebar.checkbox(model_name, value=is_default, key=f"{model_key}_checkbox")

# Display either "Select All" or "Unselect All" button based on current state
button_text = "Unselect All" if st.session_state["all_models_selected"] else "Select All"
config_sidebar.button(button_text, key="select_all_button", use_container_width=True, on_click=toggle_all_models)

# Update the all_selected state based on current checkboxes
update_all_selected_state()



############# Main Page Configuration ###########
chat_input_container = st.container(border=False)
# Chat input section
prompt = chat_input_container.chat_input(placeholder="Enter your test prompt here...", accept_file=True, key="prompt_input", file_type=["jpg", "jpeg", "png", "txt", "pdf", "docx", "csv"])
if prompt:
    st.session_state.prompt = prompt

############ Model Response Section #############
# Only display if a prompt is provided
if "prompt" in st.session_state:

    response_container = st.container(border=False)
    response_container_header = response_container.container(border=False, key="response-container-header")
    response_container_header.html("<p style='font-size: 1.5rem; font-weight: bold;'>Model Responses</p>")
    response_container_header.html("<p style='color: gray;'>Click on the model you want to select as the best.</p>")
    
    # Get selected model names
    selected_models = get_selected_model_names()

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
                        model_response_header.write("2.5s")

                        # Placeholder for model response
                        st.write("Loading response...")

                        # Model response footer
                        model_response_footer = st.container(border=False, key=f"model-response-footer-{model_card_key}")
                        # Placeholder for tokens used
                        model_response_footer.write("1293 tokens")
                        # Button to select the model as the best
                        if model_response_footer.button("Select", key=f"select-button-{model_card_key}"):
                            # Display the final modal dialog with the selected model
                            final_modal_dialog(current_model, st.session_state.prompt.text, "2.5s", "1293 tokens", "This is a test response")
                            