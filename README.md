# LLM-Comparator

A Streamlit application for comparing large language model responses with collaborative voting.

## 🚀 Getting Started for Team Members

### Development Environment Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/LLM-Comparator.git
   cd LLM-Comparator
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Create a `.env` file in the project root
   - Add your API keys (ask the team lead for development keys if needed)
   ```
   OPENAI_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here
   GOOGLE_API_KEY=your_key_here
   MISTRAL_API_KEY=your_key_here
   GROK_API_KEY=your_key_here
   OPENROUTER_API_KEY=your_key_here
   ```

5. **Run the application locally**
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure

```
LLM-Comparator/
├── assets/                      # Static assets
│   ├── images/                  # Image resources
│   └── style.css                # Custom CSS styling
├── app.py                       # Application entry point (minimal)
├── model_manager.py             # Model selection and management logic
├── apikey_manager.py            # API key management logic
├── utils.py                     # Utility functions (CSS, streaming, etc.)
├── ui.py                        # Main app UI and orchestration
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (not tracked by git)
├── .gitignore                   # Git ignore file
└── README.md                    # Project documentation
```

## 🔄 Git Workflow

We follow a feature branch workflow:

1. **Always pull the latest main branch before creating a new feature branch**
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Create a feature branch with a descriptive name**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Commit changes with clear messages**
   ```bash
   git add .
   git commit -m "Add feature: brief description of changes"
   ```

4. **Push your branch and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Request code review from at least one team member**

6. **Merge into main after approval**

## 🧩 Adding New Features

### Adding a New LLM Model

1. Add the model to the `models_options` dictionary in `model_manager.py`:
   ```python
   self.models_options = {
       # Existing models...
       "your-model-key": "Your Model Display Name",
   }
   self.models_endpoints = {
       # ...
       "your-model-key": "https://your-model-endpoint.com/v1/chat/completions",
   }
   self.models_api_keys = {
       # ...
       "your-model-key": "your_api_key_name",
   }
   ```

2. If the model requires a new API key, add it to `apikey_manager.py`:
   ```python
   self.api_keys = {
       # ...
       "your_api_key_name": None,
   }
   ```

3. The sidebar will automatically show a field for the new API key if needed.

4. (Optional) Update any custom logic in `utils.py` if your model requires a different streaming or request format.

### Creating a New Page

- This app is currently single-page, but you can add new pages using Streamlit's multipage feature if needed.
- Add a new Python file and import your components as needed.

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
