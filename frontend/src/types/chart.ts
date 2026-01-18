export interface Point {
  x: number;
  y: number;
  id?: string;
}

export interface ChartBounds {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

export interface ChartDimensions {
  width: number;
  height: number;
  margin: {
    top: number;
    right: number;
    bottom: number;
    left: number;
  };
}

export interface FittedCurve {
  points: Point[];
  expression: string;
  color: string;
}

export type FittingObjective = 'accuracy' | 'interpretability' | 'balanced';

export interface FitStatistics {
  r2: number;
  rmse: number;
  mae: number;
  aic?: number;
  bic?: number;
}

export interface ModelParameter {
  name: string;
  value: number;
  label: string;
  hint?: string;
  min?: number;
  max?: number;
  step?: number;
}

export interface ModelParameterSchema {
  modelFamily: string;
  expressionTemplate: string;
  parameters: ModelParameter[];
}

export type FitQuality = 'bad' | 'regular' | 'good';

export interface FitResult {
  expression: string;
  expressionLatex: string;
  statistics: FitStatistics;
  quality: FitQuality;
  curvePoints: Point[];
  modelType: string;
  parameterSchema?: ModelParameterSchema;
  mode?: 'auto' | 'forced';
}

export interface ModelInfo {
  modelId: string;
  displayName: string;
  parameterCount: number;
  supportsEditing: boolean;
  domain?: string;
}

export interface AnalyticalProperties {
  firstDerivative: string;
  secondDerivative: string;
  extrema: {
    type: 'maximum' | 'minimum';
    x: number;
    y: number;
  }[];
  asymptotes: {
    type: 'vertical' | 'horizontal' | 'oblique';
    value: number | string;
  }[];
}

export interface IntegralResult {
  pointA: Point;
  pointB: Point;
  area: number;
  expression: string;
}
