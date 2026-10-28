# Student Performance Analyzer

A full-stack machine-learning project with a FastAPI backend and Next.js dashboard.

## Stack
- Python, pandas, scikit-learn, FastAPI
- Linear Regression, Random Forest Regressor, Gradient Boosting Regressor
- Next.js App Router, TypeScript, Recharts

## 1. Run the backend

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend runs at `http://127.0.0.1:8000`. API docs: `http://127.0.0.1:8000/docs`.

## 2. Run the frontend

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

The frontend expects the backend at `http://127.0.0.1:8000`. Set `NEXT_PUBLIC_API_URL` if needed.

## Dataset
The included `student_performance_500.csv` should be placed in `backend/` or uploaded using the dashboard. The app defaults to `Average_Score` as the target if that column exists.

## Modeling notes
- The target defaults to `Average_Score`.
- When the target is `Average_Score`, the five subject-score columns are excluded from predictor features to reduce target leakage.
- The app compares three regression models with a held-out test set and cross-validation on the training portion.
- Metrics are dataset-dependent; do not assume a model is good just because it is more complex.
- This is a demonstration project, not a validated educational assessment system. Small datasets and synthetic data can produce optimistic or misleading results.
