class ModelManager:
    def __init__(self):
        self.models_options = {
            "GPT-3.5 Turbo": "gpt-3.5-turbo",
            "GPT-4o": "gpt-4o",
            "Gemini 2.0 Flash Experimental": "google/gemini-2.0-flash-exp:free",
            "Gemini 2.5 Pro Experimental": "google/gemini-2.5-pro-exp-03-25",
            "Claude 3.7 Sonnet": "claude-3-7-sonnet-20250219",
            "Claude 3.5 Haiku": "claude-3-5-haiku-20241022",
            "Llama 4 Maverick": "meta-llama/llama-4-maverick:free",
            "Mistral Small 3.1 24B": "mistralai/mistral-small-3.1-24b-instruct:free",
            "Qwen3": "qwen/qwen3-235b-a22b:free",
            "DeepSeek-V3": "deepseek/deepseek-chat-v3-0324:free",
        }
        self.models_endpoints = {
            "GPT-3.5 Turbo": "https://api.openai.com/v1/chat/completions",
            "GPT-4o": "https://api.openai.com/v1/chat/completions",
            "Gemini 2.0 Flash Experimental": "https://openrouter.ai/api/v1/chat/completions",
            "Gemini 2.5 Pro Experimental": "https://openrouter.ai/api/v1/chat/completions",
            "Claude 3.7 Sonnet": "https://api.anthropic.com/v1/messages",
            "Claude 3.5 Haiku": "https://api.anthropic.com/v1/messages",
            "Llama 4 Maverick": "https://openrouter.ai/api/v1/chat/completions",
            "Mistral Small 3.1 24B": "https://openrouter.ai/api/v1/chat/completions",
            "Qwen3": "https://openrouter.ai/api/v1/chat/completions",
            "DeepSeek-V3": "https://openrouter.ai/api/v1/chat/completions",
        }
        self.models_api_keys = {
            "GPT-3.5 Turbo": "open_ai_api_key",
            "GPT-4o": "open_ai_api_key",
            "Gemini 2.0 Flash Experimental": "openrouter_api_key",
            "Gemini 2.5 Pro Experimental": "openrouter_api_key",
            "Claude 3.7 Sonnet": "anthropic_api_key",
            "Claude 3.5 Haiku": "anthropic_api_key",
            "Llama 4 Maverick": "openrouter_api_key",
            "Mistral Small 3.1 24B": "openrouter_api_key",
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