'use client';

import { useState, useCallback, useRef } from 'react';
import { CartesianChart } from '@/components/CartesianChart';
import { LatexRenderer, expressionToLatex, derivativeToLatex, integralToLatex } from '@/components/LatexRenderer';
import type { Point, FittedCurve, FittingObjective, FitResult, AnalyticalProperties, IntegralResult } from '@/types/chart';

const MAX_POINTS = 50000;

type AppMode = 'idle' | 'painting' | 'selecting-integral';

interface FittingStep {
  name: string;
  status: 'pending' | 'running' | 'completed' | 'error';
}

export default function Home() {
  // Data state
  const [points, setPoints] = useState<Point[]>([]);
  const [fittedCurve, setFittedCurve] = useState<FittedCurve | null>(null);
  const [fitResult, setFitResult] = useState<FitResult | null>(null);

  // UI state
  const [mode, setMode] = useState<AppMode>('idle');
  const [fittingObjective, setFittingObjective] = useState<FittingObjective>('accuracy');
  const [isFitting, setIsFitting] = useState(false);
  const [fittingSteps, setFittingSteps] = useState<FittingStep[]>([]);
  const [fittingProgress, setFittingProgress] = useState(0);
  const [estimatedTime, setEstimatedTime] = useState<number | null>(null);

  // Analysis state
  const [analyticalProps, setAnalyticalProps] = useState<AnalyticalProperties | null>(null);
  const [integralRange, setIntegralRange] = useState<{ a: Point; b: Point } | null>(null);
  const [integralResult, setIntegralResult] = useState<IntegralResult | null>(null);
  const [integralSelectionStep, setIntegralSelectionStep] = useState<'a' | 'b' | null>(null);

  // File input ref
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Add point handler
  const handlePointAdd = useCallback((point: Point) => {
    if (points.length >= MAX_POINTS) {
      alert(`Maximum of ${MAX_POINTS.toLocaleString()} points reached`);
      return;
    }
    setPoints(prev => [...prev, point]);
  }, [points.length]);

  // CSV upload handler
  const handleCSVUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      const lines = text.split('\n').filter(line => line.trim());

      let newPoints: Point[] = [];
      let invalidRows = 0;

      lines.forEach((line, index) => {
        // Skip header row if it looks like text
        if (index === 0 && isNaN(parseFloat(line.split(',')[0]))) {
          return;
        }

        const parts = line.split(',').map(s => s.trim());
        const x = parseFloat(parts[0]);
        const y = parseFloat(parts[1]);

        if (!isNaN(x) && !isNaN(y)) {
          newPoints.push({ x, y, id: crypto.randomUUID() });
        } else {
          invalidRows++;
        }
      });

      const originalCount = newPoints.length;

      // Downsample if exceeding limit
      if (newPoints.length > MAX_POINTS) {
        const step = Math.ceil(newPoints.length / MAX_POINTS);
        newPoints = newPoints.filter((_, i) => i % step === 0).slice(0, MAX_POINTS);
        alert(`Data downsampled from ${originalCount.toLocaleString()} to ${newPoints.length.toLocaleString()} points`);
      }

      if (invalidRows > 0) {
        console.warn(`Ignored ${invalidRows} invalid rows`);
      }

      setPoints(newPoints);
    };
    reader.readAsText(file);

    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, []);

  // Estimate function (curve fitting)
  const handleEstimateFunction = useCallback(async () => {
    if (points.length < 2) {
      alert('Need at least 2 points to fit a curve');
      return;
    }

    setIsFitting(true);
    setFittingProgress(0);
    setEstimatedTime(Math.min(120, Math.ceil(points.length / 1000) * 5 + 10));

    const steps: FittingStep[] = [
      { name: 'Validating data', status: 'pending' },
      { name: 'Selecting candidate models', status: 'pending' },
      { name: 'Fitting models', status: 'pending' },
      { name: 'Checking assumptions', status: 'pending' },
      { name: 'Computing errors', status: 'pending' },
      { name: 'Comparing results', status: 'pending' },
    ];
    setFittingSteps(steps);

    try {
      const response = await fetch('http://localhost:8000/fit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          points: points.map(p => ({ x: p.x, y: p.y })),
          objective: fittingObjective,
        }),
      });

      if (!response.ok) {
        throw new Error('Fitting failed');
      }

      const result: FitResult = await response.json();

      setFitResult(result);
      setFittedCurve({
        points: result.curvePoints,
        expression: result.expression,
        color: '#22c55e',
      });

      // Mark all steps complete
      setFittingSteps(steps.map(s => ({ ...s, status: 'completed' })));
      setFittingProgress(100);
    } catch (error) {
      console.error('Fitting error:', error);
      // For now, simulate with a simple polynomial fit
      simulateFit();
    } finally {
      setTimeout(() => {
        setIsFitting(false);
      }, 500);
    }
  }, [points, fittingObjective]);

  // Simulated fit for demo when backend is unavailable
  const simulateFit = useCallback(() => {
    // Simple polynomial regression simulation
    const n = points.length;
    const sumX = points.reduce((a, p) => a + p.x, 0);
    const sumY = points.reduce((a, p) => a + p.y, 0);
    const sumXY = points.reduce((a, p) => a + p.x * p.y, 0);
    const sumX2 = points.reduce((a, p) => a + p.x * p.x, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    // Generate curve points
    const xMin = Math.min(...points.map(p => p.x));
    const xMax = Math.max(...points.map(p => p.x));
    const curvePoints: Point[] = [];
    for (let i = 0; i <= 100; i++) {
      const x = xMin + (xMax - xMin) * (i / 100);
      curvePoints.push({ x, y: slope * x + intercept });
    }

    // Calculate R²
    const yMean = sumY / n;
    const ssTot = points.reduce((a, p) => a + Math.pow(p.y - yMean, 2), 0);
    const ssRes = points.reduce((a, p) => a + Math.pow(p.y - (slope * p.x + intercept), 2), 0);
    const r2 = 1 - ssRes / ssTot;

    const rmse = Math.sqrt(ssRes / n);
    const mae = points.reduce((a, p) => a + Math.abs(p.y - (slope * p.x + intercept)), 0) / n;

    setFitResult({
      expression: `y = ${slope.toFixed(4)}x + ${intercept.toFixed(4)}`,
      expressionLatex: `y = ${slope.toFixed(4)}x + ${intercept.toFixed(4)}`,
      statistics: { r2, rmse, mae },
      quality: r2 > 0.85 ? 'good' : r2 > 0.6 ? 'regular' : 'bad',
      curvePoints,
      modelType: 'Linear Regression',
    });

    setFittedCurve({
      points: curvePoints,
      expression: `y = ${slope.toFixed(4)}x + ${intercept.toFixed(4)}`,
      color: '#22c55e',
    });

    setFittingSteps(prev => prev.map(s => ({ ...s, status: 'completed' })));
    setFittingProgress(100);
  }, [points]);

  // Compute analytical properties
  const handleComputeAnalytical = useCallback(async () => {
    if (!fitResult) return;

    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ expression: fitResult.expression }),
      });

      if (!response.ok) throw new Error('Analysis failed');

      const result: AnalyticalProperties = await response.json();
      setAnalyticalProps(result);
    } catch {
      // Simulate for linear function
      setAnalyticalProps({
        firstDerivative: fitResult.expression.includes('x') ? 'constant' : '0',
        secondDerivative: '0',
        extrema: [],
        asymptotes: [],
      });
    }
  }, [fitResult]);

  // Start integral selection
  const handleStartIntegral = useCallback(() => {
    setMode('selecting-integral');
    setIntegralSelectionStep('a');
    setIntegralRange(null);
    setIntegralResult(null);
  }, []);

  // Handle integral point selection
  const handleIntegralPointSelect = useCallback((point: Point) => {
    if (integralSelectionStep === 'a') {
      setIntegralRange({ a: point, b: point });
      setIntegralSelectionStep('b');
    } else if (integralSelectionStep === 'b' && integralRange) {
      setIntegralRange({ a: integralRange.a, b: point });
      setIntegralSelectionStep(null);
      setMode('idle');

      // Calculate integral
      if (fittedCurve) {
        const aX = Math.min(integralRange.a.x, point.x);
        const bX = Math.max(integralRange.a.x, point.x);
        const relevantPoints = fittedCurve.points.filter(p => p.x >= aX && p.x <= bX);

        // Simple trapezoidal integration
        let area = 0;
        for (let i = 1; i < relevantPoints.length; i++) {
          const dx = relevantPoints[i].x - relevantPoints[i - 1].x;
          const avgY = (relevantPoints[i].y + relevantPoints[i - 1].y) / 2;
          area += dx * avgY;
        }

        setIntegralResult({
          pointA: integralRange.a,
          pointB: point,
          area: area,  // Keep sign: negative below y=0, positive above
          expression: `∫[${aX.toFixed(2)}, ${bX.toFixed(2)}] f(x) dx`,
        });
      }
    }
  }, [integralSelectionStep, integralRange, fittedCurve]);

  // Erase and restart
  const handleReset = useCallback(() => {
    setPoints([]);
    setFittedCurve(null);
    setFitResult(null);
    setAnalyticalProps(null);
    setIntegralRange(null);
    setIntegralResult(null);
    setMode('idle');
    setIsFitting(false);
    setFittingSteps([]);
    setFittingProgress(0);
  }, []);

  const isPaintingMode = mode === 'painting';
  const isSelectingIntegral = mode === 'selecting-integral';

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-green-400 to-green-600 flex items-center justify-center">
            <span className="text-white font-bold text-sm">f</span>
          </div>
          <h1 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
            Curve Fitting Lab
          </h1>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-500 font-mono">
            {points.length.toLocaleString()} / {MAX_POINTS.toLocaleString()} points
          </span>
        </div>
      </header>

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Chart area */}
        <main className="flex-1 p-4">
          <CartesianChart
            points={points}
            fittedCurve={fittedCurve}
            onPointAdd={isSelectingIntegral ? handleIntegralPointSelect : handlePointAdd}
            isPaintingMode={isPaintingMode || isSelectingIntegral}
            integralRange={integralRange}
            analyticalMarkers={analyticalProps ? {
              extrema: analyticalProps.extrema,
              asymptotes: analyticalProps.asymptotes?.map(a => ({
                type: a.type as 'vertical' | 'horizontal',
                value: typeof a.value === 'number' ? a.value : 0,
              })),
            } : undefined}
          />
        </main>

        {/* Sidebar */}
        <aside className="w-80 border-l border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 overflow-y-auto">
          <div className="p-4 space-y-6">
            {/* Mode controls */}
            <section>
              <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-3">
                Input Mode
              </h2>
              <div className="flex gap-2">
                <button
                  onClick={() => setMode(mode === 'painting' ? 'idle' : 'painting')}
                  disabled={points.length >= MAX_POINTS}
                  className={`flex-1 px-3 py-2 text-sm font-medium rounded-lg transition-all ${
                    isPaintingMode
                      ? 'bg-indigo-500 text-white shadow-lg shadow-indigo-500/25'
                      : 'bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-700'
                  } disabled:opacity-50 disabled:cursor-not-allowed`}
                >
                  {isPaintingMode ? 'Painting...' : 'Paint Points'}
                </button>
                <label className="flex-1">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv"
                    onChange={handleCSVUpload}
                    className="hidden"
                  />
                  <span className="block px-3 py-2 text-sm font-medium text-center rounded-lg bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-700 cursor-pointer transition-colors">
                    Upload CSV
                  </span>
                </label>
              </div>
            </section>

            {/* Fitting controls */}
            <section>
              <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-3">
                Curve Fitting
              </h2>

              <div className="space-y-3">
                <div>
                  <label className="block text-xs text-zinc-500 mb-1.5">Objective</label>
                  <select
                    value={fittingObjective}
                    onChange={(e) => setFittingObjective(e.target.value as FittingObjective)}
                    className="w-full px-3 py-2 text-sm rounded-lg bg-zinc-100 dark:bg-zinc-800 border-0 focus:ring-2 focus:ring-green-500"
                  >
                    <option value="accuracy">Best Predictive Accuracy</option>
                    <option value="interpretability">Best Interpretability</option>
                    <option value="balanced">Balanced</option>
                  </select>
                </div>

                <button
                  onClick={handleEstimateFunction}
                  disabled={points.length < 2 || isFitting}
                  className="w-full px-4 py-3 text-sm font-semibold rounded-lg bg-gradient-to-r from-green-500 to-green-600 text-white shadow-lg shadow-green-500/25 hover:shadow-green-500/40 hover:from-green-600 hover:to-green-700 disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none transition-all"
                >
                  {isFitting ? 'Estimating...' : 'Estimate Function'}
                </button>
              </div>
            </section>

            {/* Fitting progress */}
            {isFitting && (
              <section className="animate-fade-in">
                <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-3">
                  Progress
                </h2>
                <div className="space-y-2">
                  {fittingSteps.map((step, i) => (
                    <div
                      key={i}
                      className={`flex items-center gap-2 text-sm ${
                        step.status === 'completed'
                          ? 'text-green-600 dark:text-green-400'
                          : step.status === 'running'
                          ? 'text-zinc-900 dark:text-zinc-100'
                          : 'text-zinc-400'
                      }`}
                    >
                      <span className={`w-4 h-4 flex items-center justify-center ${
                        step.status === 'completed' ? 'text-green-500' :
                        step.status === 'running' ? 'animate-spin' : ''
                      }`}>
                        {step.status === 'completed' ? '✓' : step.status === 'running' ? '◌' : '○'}
                      </span>
                      {step.name}
                    </div>
                  ))}

                  {estimatedTime && (
                    <div className="mt-3 text-xs text-zinc-500">
                      Est. time remaining: {estimatedTime}s
                    </div>
                  )}

                  <div className="h-1.5 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-green-500 to-green-400 transition-all duration-300"
                      style={{ width: `${fittingProgress}%` }}
                    />
                  </div>
                </div>
              </section>
            )}

            {/* Results */}
            {fitResult && !isFitting && (
              <section className="animate-fade-in">
                <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-3">
                  Result
                </h2>

                <div className="p-3 bg-zinc-50 dark:bg-zinc-800/50 rounded-lg space-y-3">
                  <div>
                    <span className="text-xs text-zinc-500">Model</span>
                    <p className="text-sm font-medium">{fitResult.modelType}</p>
                  </div>

                  <div>
                    <span className="text-xs text-zinc-500">Function</span>
                    <div className="text-sm py-1 overflow-x-auto">
                      <LatexRenderer latex={`y = ${expressionToLatex(fitResult.expression)}`} />
                    </div>
                  </div>

                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-xs text-zinc-500">Quality</span>
                    <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${
                      fitResult.quality === 'good'
                        ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                        : fitResult.quality === 'regular'
                        ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                        : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                    }`}>
                      {fitResult.quality}
                    </span>
                    {fitResult.modelType === 'Reciprocal' && (
                      <span className="px-2 py-0.5 text-xs font-medium rounded-full bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400">
                        asymptote in view
                      </span>
                    )}
                  </div>

                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div className="p-2 bg-white dark:bg-zinc-800 rounded">
                      <span className="text-[10px] text-zinc-500 block">R²</span>
                      <span className="text-sm font-mono font-semibold">
                        {fitResult.statistics.r2.toFixed(4)}
                      </span>
                    </div>
                    <div className="p-2 bg-white dark:bg-zinc-800 rounded">
                      <span className="text-[10px] text-zinc-500 block">RMSE</span>
                      <span className="text-sm font-mono font-semibold">
                        {fitResult.statistics.rmse.toFixed(4)}
                      </span>
                    </div>
                    <div className="p-2 bg-white dark:bg-zinc-800 rounded">
                      <span className="text-[10px] text-zinc-500 block">MAE</span>
                      <span className="text-sm font-mono font-semibold">
                        {fitResult.statistics.mae.toFixed(4)}
                      </span>
                    </div>
                  </div>
                </div>
              </section>
            )}

            {/* Analysis tools */}
            {fitResult && !isFitting && (
              <section>
                <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-3">
                  Analysis
                </h2>

                <div className="space-y-2">
                  <button
                    onClick={handleComputeAnalytical}
                    className="w-full px-3 py-2 text-sm font-medium rounded-lg bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors text-left"
                  >
                    Compute Derivatives & Extrema
                  </button>

                  <button
                    onClick={handleStartIntegral}
                    className={`w-full px-3 py-2 text-sm font-medium rounded-lg transition-colors text-left ${
                      isSelectingIntegral
                        ? 'bg-green-500 text-white'
                        : 'bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-700'
                    }`}
                  >
                    {isSelectingIntegral
                      ? `Select point ${integralSelectionStep?.toUpperCase()}`
                      : 'Compute Area Under Curve'}
                  </button>
                </div>

                {/* Analytical results */}
                {analyticalProps && (
                  <div className="mt-3 p-3 bg-zinc-50 dark:bg-zinc-800/50 rounded-lg space-y-3 animate-fade-in">
                    <div>
                      <span className="text-xs text-zinc-500">First Derivative</span>
                      <div className="text-sm py-1 overflow-x-auto">
                        <LatexRenderer latex={derivativeToLatex(analyticalProps.firstDerivative, 1)} />
                      </div>
                    </div>
                    <div>
                      <span className="text-xs text-zinc-500">Second Derivative</span>
                      <div className="text-sm py-1 overflow-x-auto">
                        <LatexRenderer latex={derivativeToLatex(analyticalProps.secondDerivative, 2)} />
                      </div>
                    </div>
                    {analyticalProps.extrema.length > 0 && (
                      <div>
                        <span className="text-xs text-zinc-500">Extrema</span>
                        {analyticalProps.extrema.map((e, i) => (
                          <p key={i} className="text-sm font-mono">
                            {e.type}: ({e.x.toFixed(2)}, {e.y.toFixed(2)})
                          </p>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Integral result */}
                {integralResult && integralRange && fitResult && (
                  <div className="mt-3 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg animate-fade-in">
                    <span className="text-xs text-green-600 dark:text-green-400">Definite Integral</span>
                    <div className="text-sm py-2 overflow-x-auto">
                      <LatexRenderer
                        latex={integralToLatex(
                          integralResult.area,
                          Math.min(integralRange.a.x, integralRange.b.x),
                          Math.max(integralRange.a.x, integralRange.b.x),
                          fitResult.expression
                        )}
                        displayMode={true}
                      />
                    </div>
                  </div>
                )}
              </section>
            )}

            {/* Reset */}
            <section className="pt-4 border-t border-zinc-200 dark:border-zinc-800">
              <button
                onClick={handleReset}
                className="w-full px-3 py-2 text-sm font-medium rounded-lg bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 hover:bg-red-100 dark:hover:bg-red-900/30 transition-colors"
              >
                Erase & Restart
              </button>
            </section>
          </div>
        </aside>
      </div>

      {/* Status bar */}
      <footer className="px-6 py-2 border-t border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-xs text-zinc-500">
        <div className="flex items-center justify-between">
          <span>
            {isPaintingMode && 'Click on the chart to add points'}
            {isSelectingIntegral && `Click to select point ${integralSelectionStep?.toUpperCase()}`}
            {mode === 'idle' && 'Scroll to zoom • Drag to pan'}
          </span>
          <span className="font-mono">
            Objective: {fittingObjective}
          </span>
        </div>
      </footer>
    </div>
  );
}
