# Claude Code Context

## Project Overview
Interactive Curve Fitting Lab - A local-first web app that enables advanced math/data users to draw data points, upload CSV data, fit mathematical functions using multiple model families, and analyze fitted functions (derivatives, extrema, integrals) with a world-class UI.

## Tech Stack
- **Frontend**: Next.js 16 (App Router), TypeScript, Tailwind CSS, D3.js
- **Backend**: Python 3.11+, FastAPI
- **Curve Fitting**: scikit-learn, scipy, numpy, sympy, statsmodels
- **Fonts**: Geist Sans & Geist Mono

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│                           http://localhost:3000                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────┐    ┌─────────────────────────────────┐ │
│  │         CHART AREA              │    │           SIDEBAR               │ │
│  │   ┌─────────────────────────┐   │    │                                 │ │
│  │   │                         │   │    │  ┌───────────────────────────┐  │ │
│  │   │    CartesianChart.tsx   │   │    │  │  Input Mode               │  │ │
│  │   │    ─────────────────    │   │    │  │  • Paint Points toggle    │  │ │
│  │   │    • D3.js rendering    │   │    │  │  • CSV Upload             │  │ │
│  │   │    • Zoom/Pan           │   │    │  └───────────────────────────┘  │ │
│  │   │    • Point plotting     │   │    │                                 │ │
│  │   │    • Curve display      │   │    │  ┌───────────────────────────┐  │ │
│  │   │    • Integral shading   │   │    │  │  Curve Fitting            │  │ │
│  │   │    • Extrema markers    │   │    │  │  • Objective selector     │  │ │
│  │   │                         │   │    │  │  • Estimate Function btn  │  │ │
│  │   └─────────────────────────┘   │    │  └───────────────────────────┘  │ │
│  │                                 │    │                                 │ │
│  │   x: [-10, 10] | y: [-10, 10]   │    │  ┌───────────────────────────┐  │ │
│  └─────────────────────────────────┘    │  │  Results                  │  │ │
│                                         │  │  • Function expression    │  │ │
│                                         │  │  • R², RMSE, MAE stats    │  │ │
│                                         │  │  • Quality badge          │  │ │
│                                         │  └───────────────────────────┘  │ │
│                                         │                                 │ │
│                                         │  ┌───────────────────────────┐  │ │
│                                         │  │  Analysis                 │  │ │
│                                         │  │  • Derivatives & Extrema  │  │ │
│                                         │  │  • Area Under Curve       │  │ │
│                                         │  └───────────────────────────┘  │ │
│                                         │                                 │ │
│                                         │  [Erase & Restart]              │ │
│                                         └─────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   │ HTTP/JSON
                                   │
┌──────────────────────────────────▼──────────────────────────────────────────┐
│                           FRONTEND (Next.js)                                 │
│                                                                              │
│  page.tsx ─────────────────────────────────────────────────────────────────  │
│  │                                                                           │
│  ├── State Management (React hooks)                                          │
│  │   • points: Point[]           • fittingObjective: 'accuracy'|...         │
│  │   • fittedCurve: FittedCurve  • isFitting: boolean                       │
│  │   • fitResult: FitResult      • analyticalProps: AnalyticalProperties    │
│  │                                                                           │
│  ├── Event Handlers                                                          │
│  │   • handlePointAdd()          • handleEstimateFunction()                  │
│  │   • handleCSVUpload()         • handleComputeAnalytical()                 │
│  │   • handleReset()             • handleStartIntegral()                     │
│  │                                                                           │
│  └── Fallback: simulateFit() ──► Linear regression when backend unavailable │
│                                                                              │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   │ fetch() to localhost:8000
                                   │
┌──────────────────────────────────▼──────────────────────────────────────────┐
│                           BACKEND (FastAPI)                                  │
│                          http://localhost:8000                               │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                           API ENDPOINTS                                 │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │  POST /fit                                                              │ │
│  │  ├── Input: {points: [{x,y}], objective}                                │ │
│  │  ├── Process: Fit multiple models, score by objective, select best     │ │
│  │  └── Output: {expression, statistics, quality, curvePoints, heuristics}│ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │  POST /analyze                                                          │ │
│  │  ├── Input: {expression}                                                │ │
│  │  ├── Process: Sympy differentiation, solve for extrema, find asymptotes│ │
│  │  └── Output: {firstDerivative, secondDerivative, extrema, asymptotes}  │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │  POST /integrate                                                        │ │
│  │  ├── Input: {expression, a, b}                                          │ │
│  │  ├── Process: Symbolic or numerical integration                         │ │
│  │  └── Output: {value, expression}                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         MODEL FITTING ENGINE                            │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                         │ │
│  │   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐      │ │
│  │   │   Linear    │ │ Polynomial  │ │ Exponential │ │ Logarithmic │      │ │
│  │   │  (sklearn)  │ │  (deg 2-4)  │ │   y=ae^bx   │ │ y=a·ln(x-c) │      │ │
│  │   └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘      │ │
│  │                                                                         │ │
│  │   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐      │ │
│  │   │    Power    │ │  Rational   │ │ Sinusoidal  │ │ Square Root │      │ │
│  │   │   y=ax^b    │ │ (ax+b)/(cx+d)│ │ A·sin(Bx+C) │ │ y=a√(x-c)+d │      │ │
│  │   └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘      │ │
│  │                                                                         │ │
│  │   Scoring: accuracy → R²  |  interpretability → R² - 0.1·complexity    │ │
│  │                           |  balanced → R² - 0.05·complexity            │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Libraries: numpy, scipy, scikit-learn, sympy, statsmodels                   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Action                    Frontend                         Backend
───────────                    ────────                         ───────
Paint point ──────────────────► Add to points[]
                                Re-render chart

Upload CSV ───────────────────► Parse & validate
                                Downsample if > 50k
                                Add to points[]

Click "Estimate Function" ────► POST /fit ─────────────────────► Fit all models
                                Show progress UI                  Score & rank
                                                                  Return best
                              ◄─────────────────────────────────
                                Display curve
                                Show statistics

Click "Derivatives" ──────────► POST /analyze ─────────────────► Sympy diff()
                                                                  Solve extrema
                              ◄─────────────────────────────────
                                Display results
                                Mark on chart

Select integral range ────────► Calculate area (frontend)
                                Shade region
                                Display result

Click "Erase & Restart" ──────► Clear all state
                                Reset chart
```

## Project Structure
```
math-functions/
├── frontend/                 # Next.js application
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx      # Main app (all UI state & logic)
│   │   │   ├── layout.tsx    # Root layout with fonts
│   │   │   └── globals.css   # Design system & animations
│   │   ├── components/
│   │   │   └── CartesianChart.tsx  # D3.js chart component
│   │   ├── hooks/
│   │   │   └── useChartDimensions.ts
│   │   └── types/
│   │       └── chart.ts      # TypeScript interfaces
│   └── package.json
├── backend/                  # Python FastAPI server
│   ├── main.py               # API endpoints & fitting logic
│   └── requirements.txt
├── prd.json                  # Product requirements (Ralph format)
└── progress.txt              # Development progress log
```

## Key Files

### Frontend
- `frontend/src/app/page.tsx` - Main application with all state management, handlers for painting, CSV upload, fitting, analysis
- `frontend/src/components/CartesianChart.tsx` - D3.js chart with zoom/pan, point rendering, curve display, integral shading
- `frontend/src/types/chart.ts` - TypeScript types for Point, FitResult, AnalyticalProperties, etc.

### Backend
- `backend/main.py` - FastAPI server with `/fit`, `/analyze`, `/integrate` endpoints
- Model families: linear, polynomial (degree 2-4), exponential, logarithmic (basic + shifted), square root (shifted), power, rational, reciprocal, sinusoidal

## Running the App

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:3000

### Backend
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Quality Checks

### Frontend
```bash
cd frontend
npm run build          # TypeScript check + build
npm run lint           # ESLint
```

### Backend
```bash
cd backend
python -m mypy main.py  # Type checking (optional)
```

## API Endpoints

### POST /fit
Fit a curve to data points.
```json
{
  "points": [{"x": 1, "y": 2}, {"x": 2, "y": 4}],
  "objective": "accuracy" | "interpretability" | "balanced"
}
```

### POST /analyze
Compute analytical properties of a function.
```json
{
  "expression": "y = 2x + 1"
}
```

### POST /integrate
Compute definite integral.
```json
{
  "expression": "y = x^2",
  "a": 0,
  "b": 1
}
```

## Design System

### Colors
- **Background**: zinc-50 (light), zinc-900 (dark)
- **Accent/Success**: green-500 (#22c55e)
- **Data Points**: indigo-500 (#6366f1)
- **Borders**: zinc-200 (light), zinc-800 (dark)

### Typography
- Geist Sans for UI text
- Geist Mono for numbers and code

### Animations
- `animate-fade-in`: Fade in with slight upward motion
- `animate-slide-in`: Slide in from left
- `animate-pulse-subtle`: Subtle opacity pulse
- `animate-spin`: Rotation for loading states

## Constraints
- **Fully offline**: No cloud APIs or external services
- **Max 50,000 points**: Enforced for performance
- **Max 120s fitting time**: Acceptable model runtime limit
- **~16GB RAM**: Assumed available memory

## Code Style
- TypeScript strict mode
- Functional React components with hooks
- Tailwind CSS for styling (no CSS modules)
- FastAPI with Pydantic models for validation

## Known Quirks
- Frontend includes fallback simulation when backend is unavailable
- D3.js zoom requires manual bounds state tracking for axis updates
- Backend scoring heuristics vary by objective (accuracy favors complex models, interpretability penalizes complexity)

## Recent Additions
- **Multi-Start Model Selection**: CandidateSearchEngine tries multiple parameter seeds per model family with unified CV-based scoring (CV_RMSE + complexity regularization). Prevents overfitting and ensures fair selection across families.
- **LaTeX Rendering**: Functions, derivatives, and integrals rendered with KaTeX (proper math symbols: e^{x}, √, ln, fractions, superscripts)
- **Asymptote Display**: Derivatives & Extrema card shows vertical/horizontal asymptotes for reciprocal/rational models
- **Extrapolation**: Fitted curves extend beyond data range with dotted lines for extrapolation regions
- **Collapsible Analysis Cards**: Derivatives & Extrema and Area Under Curve sections are collapsible
- **Square Root Model**: Fits y = a√(x-c) + d with domain handling (NaN for x < c)
- **Logarithmic Shifted Model**: Fits y = a·ln(x-c) + d with domain handling (NaN for x ≤ c)
- **Chart Zoom/Pan Polish**: Cursor-centered zoom, trackpad gestures, space+drag for pan in paint mode

## Ralph Integration
This project uses Ralph-style PRD tracking:
- `prd.json` contains user stories with `passes: true/false`
- `progress.txt` logs implementation progress
- `scripts/ralph/` contains automation scripts
