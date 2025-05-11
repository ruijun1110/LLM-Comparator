import os
from pathlib import Path
from dotenv import load_dotenv

class APIKeyManager:
    def __init__(self):
        self.api_keys = {
            "open_ai_api_key": None,
            "anthropic_api_key": None,
            "openrouter_api_key": None,
        }
        
        # Mapping for proper display names
        self.display_names = {
            "open_ai_api_key": "OpenAI API Key",
            "anthropic_api_key": "Anthropic API Key",
            "openrouter_api_key": "OpenRouter API Key",
        }

        # Mapping between environment variable names and session state keys
        self.env_to_session_mapping = {
            "OPENAI_API_KEY": "open_ai_api_key",
            "ANTHROPIC_API_KEY": "anthropic_api_key",
            "OPENROUTER_API_KEY": "openrouter_api_key",
        }

    def load_from_env_file(self, session_state):
        """Load API keys from .env file if it exists and update session state."""
        # Check if .env file exists
        env_path = Path('.') / '.env'
        if not env_path.exists():
            return False

        # Load environment variables from .env file
        load_dotenv()

        # Update session state with values from environment variables
        keys_loaded = False
        for env_var, session_key in self.env_to_session_mapping.items():
            value = os.getenv(env_var)
            if value and value.strip() and value != f"your_{env_var.lower()}_here":
                session_state[session_key] = value
                keys_loaded = True
        
        return keys_loaded

    def get_display_name(self, key_name):
        """Return a properly formatted display name for an API key."""
        if key_name in self.display_names:
            return self.display_names[key_name]
        
        # Fallback formatting for any keys not explicitly mapped
        words = key_name.replace('_', ' ').split()
        formatted_words = []
        
        for word in words:
            # Handle special acronyms properly
            if word.lower() in ['api', 'ai', 'ui', 'url', 'id']:
                formatted_words.append(word.upper())
            else:
                # Capitalize other words
                formatted_words.append(word.capitalize())
                
        return ' '.join(formatted_words)

    def update_key(self, key_name, value, session_state):
        session_state[key_name] = value

    def missing_keys(self, selected_models, model_manager, session_state):
        missing = []
        for model_key in selected_models:
            api_key_name = model_manager.models_api_keys.get(model_key)
            if not api_key_name or not session_state.get(api_key_name):
                missing.append(model_key)
        return missing 