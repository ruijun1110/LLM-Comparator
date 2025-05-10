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
            value = env_expander.text_input(key.replace('_', ' ').title(), type="password")
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
            model_response_header = st.container(border=False, key=f"model-response-header-{model_card_key}")
            model_response_header.write(f"{current_model}")
            url = self.model_manager.models_endpoints[current_model]
            api_key = st.session_state.get(self.model_manager.models_api_keys[current_model], "")
            if api_key == "":
                st.error(f"Missing API key for {current_model}. Please provide it in the Environment Variables section.")
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
                    model_response_header.write(f"{cached_data['time']:.2f}s")
                    model_response_footer = st.container(border=False, key=f"model-response-footer-{model_card_key}")
                    # Display token count or "None" if not available
                    token_display = f"{cached_data['total_tokens']} tokens" if cached_data['total_tokens'] is not None else "Unknown tokens"
                    model_response_footer.write(token_display)
                    if model_response_footer.button("Select", key=f"select-button-{model_card_key}"):
                        self.final_modal_dialog(current_model, prompt, f"{cached_data['time']:.2f}s", cached_data.get('total_tokens', 'Unknown'), cached_data["full_response"])
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
            
            try:
                for chunk in self.utils.stream_openai_response(
                    url=url,
                    prompt=prompt,
                    api_key=api_key,
                    model=model_id,
                    temperature=st.session_state.get("temperature", 0.5),
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
                self.final_modal_dialog(current_model, prompt, f"{response_metrics['time']:.2f}s", token_count, full_response)

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
            st.error(f"Missing API keys for: {', '.join(missing_keys)}. Please provide them in the Environment Variables section.")
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
    def final_modal_dialog(model_name, input_prompt, time_taken, tokens_used, model_response):
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