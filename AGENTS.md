# Interactive Curve Fitting Lab

## Project Structure

```
math-functions/
├── frontend/          # Next.js TypeScript app
├── backend/           # Python FastAPI for curve fitting
├── prd.json           # Product requirements (Ralph format)
├── progress.txt       # Ralph progress log
└── scripts/ralph/     # Ralph automation scripts
```

## Tech Stack

- **Frontend**: Next.js 14+, TypeScript, Tailwind CSS
- **Backend**: Python 3.11+, FastAPI
- **Curve Fitting**: scikit-learn, scipy, numpy, sympy, gplearn, pygam, statsmodels
- **Charting**: TBD (D3.js, Plotly, or similar)

## Constraints

- Fully offline - no cloud APIs
- Max 50,000 data points
- Max 120s model runtime
- Runs on localhost with ~16GB RAM

## Running the App

```bash
# Frontend
cd frontend && npm install && npm run dev

# Backend
cd backend && pip install -r requirements.txt && uvicorn main:app --reload
```

## Quality Checks

```bash
# Frontend
cd frontend && npm run typecheck && npm run lint && npm test

# Backend
cd backend && python -m mypy . && python -m pytest
```

## Design Guidelines

- Modern product aesthetics with exploratory playfulness
- Color palette: deep neutrals, vibrant greens for success, soft gradients
- Clean, math-friendly typography with clear hierarchy
- Bold, smooth, purposeful animations
- Quality comparable to Linear, Vercel, or Desmos
