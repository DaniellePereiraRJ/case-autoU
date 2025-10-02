# Email Classifier & Auto-Responder - Demo

This is a demo Flask application that classifies emails as **Produtivo** or **Improdutivo** and suggests automatic replies.
It is built as a simplified proof-of-concept for a financial company challenge.

## Files in this package
- `app.py` — Flask app with UI and classifier
- `requirements.txt` — Python dependencies
- `Procfile` — for Heroku deployment (optional)
- `sample_email.txt` — example file to test upload
- `README.md` — this file

## Run locally (recommended)
1. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate    # Windows (PowerShell)
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python app.py
   ```
4. Open http://127.0.0.1:5000 in your browser.

## OpenAI (optional)
If you want nicer generated replies using OpenAI, set the environment variable:
```bash
export OPENAI_API_KEY="your_key_here"    # Linux / macOS
setx OPENAI_API_KEY "your_key_here"      # Windows (restart required)
```
The app will attempt to use `gpt-4o-mini` for reply generation; if the key is not present a built-in template is used.

## Deploying
- **Heroku**: create a Heroku app, push the repo, and ensure `Procfile` is present.
- **Google Cloud Run**: create a Dockerfile and deploy the container.
- **Vercel**: use serverless functions or a Python server.

## Notes
- This is a demo. For production, train on real labeled email data, add authentication, logging, monitoring and robust error handling.
- PDF extraction quality depends on how the PDF is generated (text vs images).