uv add -r requirements.txt

.venv\Scripts\activate
uv run run.py 

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
