# Curve Fitting Lab

An interactive web application for fitting mathematical functions to data points. Draw points on a chart, upload CSV data, and let the app find the best-fitting function from multiple model families.

![Curve Fitting Lab](https://img.shields.io/badge/status-stable-green) ![License](https://img.shields.io/badge/license-MIT-blue)

## Features

- **Draw Points**: Click to paint data points directly on the chart
- **CSV Upload**: Import data from CSV files (up to 50,000 points)
- **Smart Curve Fitting**: Automatically finds the best model from 12+ function families:
  - Linear, Polynomial (degree 2-4)
  - Exponential, Logarithmic, Square Root
  - Power, Sinusoidal, Reciprocal, Rational
- **Model Pre-Selection**: Choose "Auto" for best fit, or force a specific model family
- **Edit Parameters**: Fine-tune fitted function parameters with live preview
- **Analysis Tools**: Compute derivatives, find extrema, calculate integrals
- **Beautiful Math**: LaTeX-rendered equations and professional visualizations
- **Fully Offline**: No cloud services required - everything runs locally

## Quick Start

### Prerequisites

You'll need to install these first (if you don't have them):

1. **Node.js** (v18 or newer)
   - Download from: https://nodejs.org/
   - Choose the "LTS" version
   - Run the installer

2. **Python** (v3.9 or newer)
   - Download from: https://www.python.org/downloads/
   - During installation, check "Add Python to PATH"

### Setup Instructions

#### Step 1: Download the Project

**Option A - Using Git:**
```bash
git clone https://github.com/francochiaro/math-functions.git
cd math-functions
```

**Option B - Download ZIP:**
1. Click the green "Code" button on GitHub
2. Click "Download ZIP"
3. Extract the ZIP file
4. Open a terminal in the extracted folder

#### Step 2: Set Up the Backend (Python)

Open a terminal and run:

```bash
# Navigate to the backend folder
cd backend

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Step 3: Set Up the Frontend (Node.js)

Open a **new terminal** and run:

```bash
# Navigate to the frontend folder
cd frontend

# Install dependencies
npm install
```

### Running the App

You need **two terminals** running simultaneously:

**Terminal 1 - Start the Backend:**
```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
uvicorn main:app --port 8000
```

**Terminal 2 - Start the Frontend:**
```bash
cd frontend
npm run dev
```

Then open your browser to: **http://localhost:3000**

### Stopping the App

Press `Ctrl+C` in both terminal windows.

## How to Use

### 1. Add Data Points

- Click **"Paint Points"** to enable drawing mode
- Click anywhere on the chart to add points
- Or click **"Upload CSV"** to import data (format: `x,y` per line)

### 2. Fit a Curve

- Select a **Model**:
  - **Auto (best fit)**: Let the app find the best model automatically
  - Or choose a specific model (Linear, Exponential, Sinusoidal, etc.)
- Select a fitting **Objective**:
  - **Accuracy**: Best mathematical fit
  - **Interpretability**: Simpler functions preferred
  - **Balanced**: Mix of both
- Click **"Estimate Function"**
- The app will fit using your selected model (or find the best one if Auto)

### 3. Edit Parameters (Optional)

- Click the **pencil icon** next to the function
- Adjust parameter values (a, b, c, d...)
- See live preview of changes
- Click **Apply** to save or **Cancel** to revert

### 4. Analyze

- **Derivatives & Extrema**: Find slopes and turning points
- **Area Under Curve**: Calculate definite integrals

### 5. Reset

Click **"Erase & Restart"** to clear everything and start over.

## Supported Function Types

| Type | Form | Example |
|------|------|---------|
| Linear | y = mx + b | y = 2x + 1 |
| Quadratic | y = ax² + bx + c | y = x² - 2x + 1 |
| Cubic | y = ax³ + bx² + cx + d | y = x³ - 3x |
| Exponential | y = a·eᵇˣ + c | y = 2·e⁰·⁵ˣ |
| Logarithmic | y = a·ln(x-c) + d | y = 3·ln(x) + 1 |
| Square Root | y = a·√(x-c) + d | y = 2·√x |
| Power | y = a·xᵇ | y = 3·x⁰·⁵ |
| Sinusoidal | y = A·sin(Bx+C) + D | y = sin(2x) |
| Reciprocal | y = a/(x-c) + d | y = 1/x |
| Rational | y = (ax+b)/(cx+d) | y = (2x+1)/(x-1) |

## Troubleshooting

### "Failed to fetch" error
The backend isn't running. Make sure Terminal 1 is running `uvicorn main:app --port 8000`.

### "Module not found" error
Dependencies aren't installed. Run `pip install -r requirements.txt` in the backend folder or `npm install` in the frontend folder.

### Port already in use
Another app is using the port. Either close that app or use a different port:
```bash
# Backend on different port
uvicorn main:app --port 8001

# Frontend on different port
npm run dev -- -p 3001
```

### Python command not found
Try `python` instead of `python3`, or ensure Python is in your PATH.

## Project Structure

```
math-functions/
├── frontend/           # Web interface (Next.js + React)
│   ├── src/
│   │   ├── app/        # Main application pages
│   │   ├── components/ # Reusable UI components
│   │   └── types/      # TypeScript definitions
│   └── package.json
├── backend/            # API server (Python + FastAPI)
│   ├── main.py         # All fitting logic and endpoints
│   └── requirements.txt
└── README.md
```

## Tech Stack

- **Frontend**: Next.js 16, TypeScript, Tailwind CSS, D3.js, KaTeX
- **Backend**: Python 3.11+, FastAPI, NumPy, SciPy, scikit-learn, SymPy

## License

MIT License - feel free to use, modify, and distribute.

## Contributing

Contributions welcome! Please open an issue or pull request.
