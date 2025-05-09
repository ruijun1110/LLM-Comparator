class ModelManager:
    def __init__(self):
        self.models_options = {
            "gpt-3-5": "gpt-3.5-turbo-0125",
            "gpt-4o": "GPT-4o",
            "o1-preview": "o1 (Preview)",
            "o4-mini-preview": "o4 mini (Preview)",
            "gemini-2-0-flash": "Gemini 2.0 Flash",
            "gemini-2-5-pro": "gemini-2.5-pro-exp-03-25",
            "claude-3-7-sonnet": "claude-3-7-sonnet-20250219",
            "claude-3-5-haiku": "claude-3-5-haiku-20241022",
            "Qwen3": "qwen/qwen3-235b-a22b:free",
            "DeepSeek-V3": "deepseek/deepseek-chat-v3-0324:free",
        }
        self.models_endpoints = {
            "gpt-3-5": "https://api.openai.com/v1/chat/completions",
            "gpt-4o": "https://api.openai.com/v1/chat/completions",
            "o1-preview": "https://api.openai.com/v1/chat/completions",
            "o4-mini-preview": "https://api.openai.com/v1/chat/completions",
            "gemini-2-0-flash": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            "gemini-2-5-pro": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            "claude-3-7-sonnet": "https://api.anthropic.com/v1/messages",
            "claude-3-5-haiku": "https://api.anthropic.com/v1/messages",
            "Qwen3": "https://openrouter.ai/api/v1/chat/completions",
            "DeepSeek-V3": "https://openrouter.ai/api/v1/chat/completions",
        }
        self.models_api_keys = {
            "gpt-3-5": "open_ai_api_key",
            "gpt-4o": "open_ai_api_key",
            "o1-preview": "open_ai_api_key",
            "o4-mini-preview": "open_ai_api_key",
            "gemini-2-0-flash": "google_gemini_api_key",
            "gemini-2-5-pro": "google_gemini_api_key",
            "claude-3-7-sonnet": "anthropic_api_key",
            "claude-3-5-haiku": "anthropic_api_key",
            "Qwen3": "openrouter_api_key",
            "DeepSeek-V3": "openrouter_api_key",
        }
        self.default_selected = ["DeepSeek-V3", "Qwen3"]

    def get_selected_models(self, session_state):
        selected_models = []
        for model_key in self.models_options.keys():
            if session_state.get(f"{model_key}-checkbox", False):
                selected_models.append(model_key)
        return selected_models

    def toggle_all(self, session_state):
        session_state["all_models_selected"] = not session_state["all_models_selected"]
        for model_key in self.models_options.keys():
            session_state[f"{model_key}-checkbox"] = session_state["all_models_selected"]
        if session_state["all_models_selected"]:
            session_state["num_selected_models"] = len(self.models_options)
        else:
            session_state["num_selected_models"] = 0

    def update_selected_models_count(self, session_state):
        count = sum(session_state.get(f"{key}-checkbox", False) for key in self.models_options.keys())
        session_state["num_selected_models"] = count

    def update_all_selected_state(self, session_state):
        all_selected = all(session_state.get(f"{key}-checkbox", False) for key in self.models_options.keys())
        session_state["all_models_selected"] = all_selected 