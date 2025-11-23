# Housing Price Prediction

Streamlit app to predict housing prices using a pre-trained scikit-learn LinearRegression model.

## Run locally

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the app:

```powershell
& "C:/Program Files/Python312/python.exe" -m streamlit run app.py
```

## Files

- `app.py` - Streamlit application
- `model.pkl` - pickled scikit-learn model (already present)
- `requirements.txt` - Python dependencies

## Push to GitHub

Create a new repository on GitHub named `housing-price-prediction` (public). Then add the remote and push:

```powershell
# SSH (recommended if you have SSH keys set up)
git remote add origin git@github.com:YOUR_GITHUB_USERNAME/housing-price-prediction.git
git push -u origin main

# or HTTPS
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/housing-price-prediction.git
git push -u origin main
```

If you prefer, use the GitHub CLI (must be authenticated):

```powershell
gh repo create YOUR_GITHUB_USERNAME/housing-price-prediction --public --source=. --remote=origin --push
```

Security note: If you do not want the `model.pkl` in a public repo, either remove it before pushing or add `model.pkl` to `.gitignore`.
