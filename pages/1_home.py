import streamlit as st
import streamlit_shadcn_ui as ui
import math

models_options = {
    "gpt_3_5": "GPT-3.5",
    "gpt_4o": "GPT-4o",
    "o1_preview": "o1 (Preview)",
    "o4_mini_preview": "o4 mini (Preview)",
    "gemini_2_0_flash": "Gemini 2.0 Flash",
    "gemini_2_5_pro": "Gemini 2.5 Pro",
    "claude_sonnet_3_5": "Claude Sonnet 3.5",
    "claude_sonnet_3_7": "Claude Sonnet 3.7",
    "grok_2": "Grok 2",
    "mistral_7b": "Mistral 7B"
}

default_selected = ["gpt_3_5", "gpt_4o", "gemini_2_5_pro", "claude_sonnet_3_7"]


st.markdown("<h1 align='center'>LLM Model Comparison</h1>", unsafe_allow_html=True)

config_sidebar = st.sidebar
config_sidebar.subheader("Basic Configuration", divider=True)
config_sidebar.slider("Temperature", 0.0, 1.0, 0.5, key="temperature_slider")
config_sidebar.slider("Max Tokens", 1, 4000, 1000, key="max_tokens_slider")
config_sidebar.subheader("Models to compare", divider=True)

if "all_models_selected" not in st.session_state:
    st.session_state["all_models_selected"] = False

for model_key, model_name in models_options.items():
    is_default = model_key in default_selected
    config_sidebar.checkbox(model_name, value=is_default, key=f"{model_key}_checkbox")

def toggle_all_models():
    """Toggle all model checkboxes based on the current state."""
    # Flip the state
    st.session_state["all_models_selected"] = not st.session_state["all_models_selected"]
    
    # Set all checkboxes according to the new state
    for model_key in models_options.keys():
        st.session_state[f"{model_key}_checkbox"] = st.session_state["all_models_selected"]

# Display either "Select All" or "Unselect All" button based on current state
button_text = "Unselect All" if st.session_state["all_models_selected"] else "Select All"
config_sidebar.button(button_text, key="select_all_button", use_container_width=True, on_click=toggle_all_models)

# Function to check if all models are selected and update the state accordingly
def update_all_selected_state():
    all_selected = all(st.session_state.get(f"{key}_checkbox", False) for key in models_options.keys())
    st.session_state["all_models_selected"] = all_selected

# Update the all_selected state based on current checkboxes
update_all_selected_state()

chat_input_container = st.container(border=False)
prompt = chat_input_container.chat_input(placeholder="Enter your test prompt here...", accept_file=True, key="prompt_input", file_type=["jpg", "jpeg", "png", "txt", "pdf", "docx", "csv"])


# Function to get selected model names
def get_selected_model_names():
    """Return a list of selected model names."""
    selected_models = []
    for model_key, model_name in models_options.items():
        if st.session_state.get(f"{model_key}_checkbox", False):
            selected_models.append(model_name)
    return selected_models
    

if prompt:

    response_container = st.container(border=False)

    # Create the cards section
    response_container.subheader("Model Responses", divider=True)

    selected_models = get_selected_model_names()

    # Calculate number of rows needed
    num_rows = math.ceil(len(selected_models) / 3)

    # Create rows and cards
    for row_idx in range(num_rows):
        # Create a row with 3 columns
        cols = response_container.columns(3)
        # Add cards to this row
        for col_idx in range(3):
            # Check if the column index is within the range of selected models
            card_index = row_idx * 3 + col_idx
            if card_index < len(selected_models):
                with cols[col_idx]:
                    model_card = st.container(border=True, height=400)
                    with model_card:
                        # Display the model name
                        st.write(f"### {selected_models[card_index]}", divider=True)
                        # Placeholder for model response
                        st.markdown("Loading response...", unsafe_allow_html=True)
                        