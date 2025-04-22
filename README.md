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
   ```

5. **Run the application locally**
   ```bash
   streamlit run ./pages/1_home.py
   ```

## 📂 Project Structure

```
LLM-Comparator/
├── assets/                      # Static assets
│   ├── images/                  # Image resources
│   └── style.css                # Custom CSS styling
├── pages/                       # Streamlit pages
│   └── 1_home.py                # Home page with comparison UI
├── main.py                      # Application entry point
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

1. Add the model to the `models_options` dictionary in `pages/1_home.py`:
   ```python
   models_options = {
       # Existing models...
       "your-model-key": "Your Model Display Name",
   }
   ```

2. Implement the API integration in the backend

3. Add the corresponding API key input in the sidebar

### Creating a New Page

1. Add a new Python file in the `pages/` directory
   - File naming format: `<page_number>_<page_name>.py`
   - Example: `2_analytics.py`

2. Import required Streamlit components and add your page content


## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
