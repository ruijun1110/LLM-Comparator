class APIKeyManager:
    def __init__(self):
        self.api_keys = {
            "open_ai_api_key": None,
            "anthropic_api_key": None,
            "google_gemini_api_key": None,
            "mistral_api_key": None,
            "grok_api_key": None,
            "openrouter_api_key": None,
        }

    def update_key(self, key_name, value, session_state):
        session_state[key_name] = value

    def missing_keys(self, selected_models, model_manager, session_state):
        missing = []
        for model_key in selected_models:
            api_key_name = model_manager.models_api_keys.get(model_key)
            if not api_key_name or not session_state.get(api_key_name):
                missing.append(model_key)
        return missing 