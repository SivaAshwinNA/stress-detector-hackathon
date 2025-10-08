# Stress Detector

A small web-based stress-detection demo that classifies short pieces of text for signs of stress. This repository contains a minimal web frontend (HTML/CSS/JS), a Python entrypoint, and a tiny script for running the prediction model against example sentences.

## Highlights

- Lightweight, easy to run (single-folder project)
- Minimal frontend using `templates/chat.html` and `static/` assets
- `model_run.py` provides an example harness that calls the prediction function

## Quick overview

Files and folders you'll find in this repo:

- `main.py` — Project entrypoint / web server (start here to run the app)
- `model_run.py` — Small test harness that calls `predict_stress.predict_stress()` against example sentences
- `predict_stress.py` _(expected)_ — Module that exposes `predict_stress(text: str) -> dict/str` used by `model_run.py` (not included here; must exist or be implemented)
- `requirements.txt` — Python dependencies
- `templates/` — HTML templates (contains `chat.html`)
- `static/` — Static assets (contains `chat.css`, `chat.js`)
- `app.db` — (optional) local SQLite DB used by the app if present

> Note: The repository expects a `predict_stress` module (used by `model_run.py`). If you don't have it, create one that exposes a `predict_stress(text: str)` function which returns a classification or dictionary.

## Prerequisites

- Python 3.8 or newer (3.10/3.11 recommended)
- pip
- (Optional) virtual environment tooling (venv, virtualenv, conda)

## Setup (Windows PowerShell)

1. Open PowerShell in the repository root (where `requirements.txt` is located).
2. (Optional) Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

If `requirements.txt` is missing or empty, install the packages your project needs (for a typical small web app you may need `flask` or `fastapi`, `gunicorn`, and ML libraries like `scikit-learn` / `transformers` depending on the implementation).

## Run the web app

The repository provides `main.py` as the expected server entrypoint. Start the application using:

```powershell
python main.py
```

Open your browser and go to http://127.0.0.1:5000 (or the host/port printed by `main.py`). If the project uses a different host/port configuration, follow the printed log message.

If the project actually uses Flask with `flask run`, you can also run:

```powershell
$env:FLASK_APP = 'main.py'
flask run
```

## Run the prediction example script

`model_run.py` demonstrates how to call the prediction function from code. Run it with:

```powershell
python model_run.py
```

It will loop through several sample sentences and print the predicted result for each.

## Expected `predict_stress` contract

If you need to implement `predict_stress.py`, aim for this small contract:

- Function: `predict_stress(text: str) -> Union[str, dict]`
- Input: plain text (string)
- Output (suggested): either a label string (e.g. `"stressed"` / `"not_stressed"`) or a dict with probabilities, for example:

```py
{
  "label": "stressed",
  "score": 0.92
}
```

Keep the implementation small and well-documented so `model_run.py` can import and use it without extra plumbing.

## Files of interest

- `templates/chat.html` — simple chat-style frontend. Use this to prototype text input for the model.
- `static/chat.js` — frontend logic; likely sends text to the server via fetch/AJAX.
- `static/chat.css` — styling for the chat UI.

If you plan to extend the app, consider adding an API endpoint such as `/api/predict` that accepts POST requests with JSON `{ "text": "..." }` and returns the model's prediction.

## Troubleshooting

- Import errors: Ensure `predict_stress.py` is on the Python path (same folder or installed as a package) and exports `predict_stress`.
- Port already in use: change the port in `main.py` or use an available port: `python main.py --port 8080` (if supported).
- Missing packages: install missing packages shown in the error message with `pip install <package>` or add them to `requirements.txt`.

## Extending this project

- Add unit tests for the prediction function and any API endpoints.
- Wrap the model code into a class or lightweight service to support batching and caching.
- Add Dockerfile + docker-compose for easy deployment.
- Add CI (GitHub Actions) to run tests and linting.

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Open a pull request with a clear description of the changes.

## License

This repo does not contain a license file. Add one (MIT/Apache-2.0) if you want to make the project open-source.

---

If you'd like, I can:

- Update `README.md` with more specific run instructions after I inspect `main.py` (I can read the file and add exact host/port and framework info).
- Create a minimal `predict_stress.py` stub so `model_run.py` runs out-of-the-box.

Tell me which of the two you'd like me to do next.

# stress-detector-hackathon
