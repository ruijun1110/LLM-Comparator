import streamlit as st
import pathlib
from model_manager import ModelManager
from apikey_manager import APIKeyManager
from utils import Utils

class LLMComparatorApp:
    def __init__(self):
        self.model_manager = ModelManager()
        self.api_key_manager = APIKeyManager()
        self.utils = Utils()
        self.css_path = pathlib.Path(__file__).parent / "assets" / "style.css"

    def render_sidebar(self):
        config_sidebar = st.sidebar
        config_sidebar.subheader("Basic Configuration", divider=True)
        temperature = config_sidebar.slider("Temperature", 0.0, 1.0, 0.5, key="temperature_slider")
        if temperature:
            st.session_state.temperature = temperature
        max_token = config_sidebar.slider("Max Tokens", 1, 4000, 1000, key="max_tokens_slider")
        if max_token:
            st.session_state.max_token = max_token
        config_sidebar.subheader("Models to compare", divider=True)
        env_expander = config_sidebar.expander("Environment Variables", expanded=False)
        for key in self.api_key_manager.api_keys.keys():
            # Use the proper display name
            display_name = self.api_key_manager.get_display_name(key)
            value = env_expander.text_input(display_name, type="password")
            self.api_key_manager.update_key(key, value, st.session_state)
        for model_key, model_name in self.model_manager.models_options.items():
            is_default = model_key in self.model_manager.default_selected
            config_sidebar.checkbox(model_key, value=is_default, key=f"{model_key}-checkbox", on_change=lambda: self.model_manager.update_selected_models_count(st.session_state))
        button_text = "Unselect All" if st.session_state["all_models_selected"] else "Select All"
        config_sidebar.button(button_text, key="select_all_button", use_container_width=True, on_click=lambda: self.model_manager.toggle_all(st.session_state))
        self.model_manager.update_all_selected_state(st.session_state)

    def render_model_card(self, current_model, prompt):
        model_card_key = f"model-card-{current_model}"
        model_card = st.container(key=model_card_key, border=False)
        with model_card:
            # Create a consistently styled header
            model_response_header = st.container(border=False, key=f"model-response-header-{model_card_key}")
            model_response_header.write(f"{current_model}")
            
            url = self.model_manager.models_endpoints[current_model]
            api_key_name = self.model_manager.models_api_keys[current_model]
            api_key = st.session_state.get(api_key_name, "")
            if api_key == "":
                # Use the proper display name from APIKeyManager
                api_key_display_name = self.api_key_manager.get_display_name(api_key_name)
                
                # Create a more concise error message
                st.error(f"Missing {api_key_display_name} for {current_model}")
                
                # Mark this model as completed with error
                st.session_state.completed_models.add(current_model)
                return
            model_id = self.model_manager.models_options[current_model]
            
            # Check if we already have a cached response for this model and prompt
            cache_key = f"{current_model}_{prompt}"
            if not st.session_state.get("should_query_api", False) and cache_key in st.session_state.get("response_cache", {}):
                cached_data = st.session_state.response_cache[cache_key]
                response_placeholder = st.empty()
                if cached_data.get("error"):
                    response_placeholder.error(f"Error: {cached_data['error']}")
                else:
                    response_placeholder.markdown(cached_data["full_response"])
                    # Update the time in the header 
                    model_response_header.write(f"{cached_data['time']:.2f}s")
                    model_response_footer = st.container(border=False, key=f"model-response-footer-{model_card_key}")
                    # Display token count or "None" if not available
                    token_display = f"{cached_data['total_tokens']} tokens" if cached_data['total_tokens'] is not None else "Unknown tokens"
                    model_response_footer.write(token_display)
                    temperature = st.session_state.get("temperature", 0.5)
                    if model_response_footer.button("Select", key=f"select-button-{model_card_key}"):
                        self.final_modal_dialog(current_model, prompt, f"{cached_data['time']:.2f}s", cached_data.get('total_tokens', 'Unknown'), cached_data["full_response"], temperature)
                # Mark this model as completed (from cache)
                st.session_state.completed_models.add(current_model)
                return
            
            # Streaming logic with metrics
            response_placeholder = st.empty()
            full_response = ""
            response_metrics = {
                "time": 0.0,
                "total_tokens": None,
                "model": model_id,
                "finish_reason": None
            }
            error_message = None
            has_received_content = False
            temperature = st.session_state.get("temperature", 0.5)
            
            try:
                for chunk in self.utils.stream_openai_response(
                    url=url,
                    prompt=prompt,
                    api_key=api_key,
                    model=model_id,
                    temperature=temperature,
                    max_tokens=st.session_state.get("max_token", 1000)
                ):
                    if chunk["type"] == "content":
                        if chunk["content"]:
                            has_received_content = True
                            full_response += chunk["content"]
                            response_placeholder.markdown(full_response + "▌")
                        usage = chunk.get("usage", {})
                        if usage:
                            total_tokens = usage.get("total_tokens", None) # OpenAI compatibility
                            if not total_tokens:
                                total_tokens = usage.get("output_tokens", None) # Anthropic compatibility
                            response_metrics["total_tokens"] = total_tokens
                                
                        if chunk.get("finish_reason"):
                            response_metrics["finish_reason"] = chunk["finish_reason"]
                    elif chunk["type"] == "done":
                        response_metrics["time"] = chunk["time"]
                        # Check if the done event contains token usage data
                        usage = chunk.get("usage", {})
                        if usage:
                            total_tokens = usage.get("total_tokens", None)
                            if total_tokens:
                                response_metrics["total_tokens"] = total_tokens
                        
                        # If we've reached the end without receiving any content and no error,
                        # it's likely an empty response due to an error not properly caught
                        if not has_received_content and not error_message:
                            error_message = "No response received from the model. This may be due to a rate limit or other API error."
                            response_placeholder.error(f"Error: {error_message}")
                    elif chunk["type"] == "error":
                        error_message = chunk["error"]
                        response_placeholder.error(f"Error: {error_message}")
                        break
            except Exception as e:
                error_message = str(e)
                response_placeholder.error(f"Error: {error_message}")
            
            # Cache the response
            if "response_cache" not in st.session_state:
                st.session_state.response_cache = {}
                
            if error_message:
                st.session_state.response_cache[cache_key] = {
                    "error": error_message
                }
                # Mark this model as completed with error
                st.session_state.completed_models.add(current_model)
                # Force a page rerun to update the counter
                st.rerun()
                return
                
            # Handle the case where we finished without errors but also without content
            if not has_received_content:
                error_message = "No response received from the model. This may be due to a rate limit or other API error."
                response_placeholder.error(f"Error: {error_message}")
                st.session_state.response_cache[cache_key] = {
                    "error": error_message
                }
                # Mark this model as completed with error
                st.session_state.completed_models.add(current_model)
                # Force a page rerun to update the counter
                st.rerun()
                return
                
            # Remove cursor and display result if no error occurred
            response_placeholder.markdown(full_response)
            # Display metrics
            model_response_header.write(f"{response_metrics['time']:.2f}s")
            model_response_footer = st.container(border=False, key=f"model-response-footer-{model_card_key}")
            # Display token count or "Unknown" if not available
            token_display = f"{response_metrics['total_tokens']} tokens" if response_metrics['total_tokens'] is not None else "Unknown tokens"
            model_response_footer.write(token_display)
            
            # Cache the successful response
            st.session_state.response_cache[cache_key] = {
                "full_response": full_response,
                "time": response_metrics["time"],
                "total_tokens": response_metrics["total_tokens"]
            }
            
            # Mark this model as completed successfully
            st.session_state.completed_models.add(current_model)
            
            # Force a page rerun to update the counter
            st.rerun()
            
            if model_response_footer.button("Select", key=f"select-button-{model_card_key}"):
                token_count = response_metrics['total_tokens'] if response_metrics['total_tokens'] is not None else "Unknown"
                self.final_modal_dialog(current_model, prompt, f"{response_metrics['time']:.2f}s", token_count, full_response, temperature)

    def render_main(self):
        chat_input_container = st.container(border=False)
        selected_models = self.model_manager.get_selected_models(st.session_state)
        st.session_state["num_selected_models"] = len(selected_models)
        enough_models_error = len(selected_models) < 2
        missing_keys = self.api_key_manager.missing_keys(selected_models, self.model_manager, st.session_state)
        required_api_keys_error = len(missing_keys) > 0
        prompt = chat_input_container.chat_input(
            placeholder="Enter your test prompt here...",
            accept_file=True, key="prompt_input",
            file_type=["jpg", "jpeg", "png", "txt", "pdf", "docx", "csv"],
            disabled=enough_models_error or required_api_keys_error
        )
        if prompt:
            # New prompt submitted - clear previous responses
            if "response_cache" in st.session_state:
                st.session_state.response_cache = {}
            
            # Set the new prompt and query flag
            st.session_state.prompt = prompt
            st.session_state.should_query_api = True
            st.session_state["current_displayed_models"] = selected_models
            
            # Reset completed models tracking
            st.session_state.completed_models = set()
        else:
            # Set flag to not query API if no new prompt was submitted
            st.session_state.should_query_api = False
            
        if enough_models_error:
            st.error("Please select at least two models to compare.")
        if required_api_keys_error:
            # Create a more concise but informative error message
            required_apis = {}
            for model in missing_keys:
                api_key_name = self.model_manager.models_api_keys[model]
                # Use the proper display name from APIKeyManager
                api_display_name = self.api_key_manager.get_display_name(api_key_name)
                if api_display_name not in required_apis:
                    required_apis[api_display_name] = []
                required_apis[api_display_name].append(model)
            
            # Display a simple, clean error message
            if len(required_apis) == 1:
                # Single API key missing - very simple message
                api = list(required_apis.keys())[0]
                models = required_apis[api]
                if len(models) == 1:
                    error_msg = f"Missing {api} for {models[0]}"
                else:
                    # For multiple models but single API
                    model_count = len(models)
                    first_model = models[0]
                    error_msg = f"Missing {api} for {first_model} and {model_count-1} other model(s)"
            else:
                # Multiple API keys - still keep it simple
                keys_list = ", ".join(required_apis.keys())
                error_msg = f"Missing API keys: {keys_list}"
            
            st.error(error_msg)
        if "prompt" in st.session_state and "current_displayed_models" in st.session_state:
            response_container = st.container(border=False)
            response_container_header = response_container.container(border=False, key="response-container-header")
            response_container_header.html("<p style='font-size: 1.5rem; font-weight: bold;'>Model Responses</p>")
            
            # Make sure completed_models exists
            if "completed_models" not in st.session_state:
                st.session_state.completed_models = set()
                
            # Add progress indicator for model processing
            total_models = len(st.session_state["current_displayed_models"])
            completed_models = len(st.session_state.completed_models)
            
            # Check if all models are complete
            if completed_models >= total_models:
                response_container_header.html(
                    """
                    <p style='color: gray; font-weight: 500; font-size: 1.1em;'>
                        Please select the response you prefer the most
                    </p>
                    """
                )
            else:
                progress_text = f"Processing model responses: {completed_models}/{total_models}"
                response_container_header.html(
                    f"""
                    <div class="model-loading-indicator">
                        <div class="progress-spinner">⏳</div>
                        <div class="progress-text">{progress_text}</div>
                    </div>
                    """
                )
            
            selected_models = st.session_state["current_displayed_models"]
            import math
            num_rows = math.ceil(len(selected_models) / 3)
            for row_idx in range(num_rows):
                cols = response_container.columns(3)
                for col_idx in range(3):
                    card_index = row_idx * 3 + col_idx
                    if card_index < len(selected_models):
                        with cols[col_idx]:
                            current_model = selected_models[card_index]
                            self.render_model_card(current_model, st.session_state.prompt.text)
            
            # Reset the API query flag after rendering all models
            st.session_state.should_query_api = False

    @staticmethod
    @st.dialog("🎉 Congratulations 🎉")
    def final_modal_dialog(model_name, input_prompt, time_taken, tokens_used, model_response, temperature=0.5):
        # Header with model name
        st.html("<h3 style='text-align:center; margin-bottom:5px;'>You have found the best model for your task!</h3>")
        st.html(f"<h1 style='text-align:center; font-size:2.2em; color: #f1c40f; text-shadow: 1px 1px 2px rgba(0,0,0,0.3); margin-bottom:20px;'>{model_name}</h1>")
        
        # Style for section titles with gradient underline effect - added consistent margins
        section_title_style = "color: #3498db; font-size: 1.2em; font-weight: bold; margin-top: 20px; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 2px solid; border-image: linear-gradient(to right, #3498db, transparent) 1;"
        
        # Input Prompt Section
        st.markdown(f"<div style='{section_title_style}'>Input Prompt</div>", unsafe_allow_html=True)
        prompt_container = st.container(border=False, key="final-prompt-container")
        prompt_container.markdown(f"<div style='font-family:monospace; padding: 5px;'>{input_prompt}</div>", unsafe_allow_html=True)
        
        # Model Response Section - ensure consistent spacing
        st.markdown(f"<div style='{section_title_style}'>Model Response</div>", unsafe_allow_html=True)
        response_container = st.container(border=False, key="final-response-container")
        # Convert markdown to HTML if possible, otherwise use plain text
        try:
            import markdown
            response_html = markdown.markdown(model_response)
            response_container.markdown(f"<div style='font-family:sans-serif; line-height:1.5;'>{response_html}</div>", unsafe_allow_html=True)
        except:
            response_container.markdown(model_response)
        
        # Performance Metrics Section (renamed from Model Parameters)
        st.markdown(f"<div style='{section_title_style.replace('#3498db', '#f39c12')}'>Performance Metrics</div>", unsafe_allow_html=True)
        param_container = st.container(border=False, key="final-params-container")
        
        # Create three columns for parameters
        param_cols = param_container.columns(3)
        
        # Column 1: Time Taken
        with param_cols[0]:
            st.markdown("<span style='color: #95a5a6; font-weight: 500; font-size: 0.9em;'>TIME TAKEN</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size: 1.3em; font-family: monospace; color: #2ecc71;'>{time_taken}</span>", unsafe_allow_html=True)
        
        # Column 2: Tokens Used
        with param_cols[1]:
            st.markdown("<span style='color: #95a5a6; font-weight: 500; font-size: 0.9em;'>TOKENS USED</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size: 1.3em; font-family: monospace; color: #e67e22;'>{tokens_used}</span>", unsafe_allow_html=True)
            
        # Column 3: Temperature
        with param_cols[2]:
            st.markdown("<span style='color: #95a5a6; font-weight: 500; font-size: 0.9em;'>TEMPERATURE</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size: 1.3em; font-family: monospace; color: #3498db;'>{temperature}</span>", unsafe_allow_html=True)
        
        st.balloons()

    def run(self):
        # Initialize session state
        if "all_models_selected" not in st.session_state:
            st.session_state["all_models_selected"] = False
        if "num_selected_models" not in st.session_state:
            st.session_state["num_selected_models"] = len(self.model_manager.default_selected)
        if "current_displayed_models" not in st.session_state:
            st.session_state["current_displayed_models"] = []
        if "response_cache" not in st.session_state:
            st.session_state.response_cache = {}
        if "should_query_api" not in st.session_state:
            st.session_state.should_query_api = False
        if "completed_models" not in st.session_state:
            st.session_state.completed_models = set()
        # Load CSS
        self.utils.load_css(self.css_path)
        st.html("<h1 style='text-align:center; font-size:2em;'>LLM Model Comparison</h1>")
        self.render_sidebar()
        self.render_main() 