'use client';

import { useMemo } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

interface LatexRendererProps {
  latex: string;
  displayMode?: boolean;
  className?: string;
}

/**
 * Round a number to n significant figures for display
 */
function roundToSignificant(num: number, sigFigs: number = 2): string {
  if (num === 0) return '0';
  if (!Number.isFinite(num)) return String(num);

  const absNum = Math.abs(num);
  const sign = num < 0 ? '-' : '';

  // For numbers >= 1, round to specified decimal places
  if (absNum >= 100) {
    return sign + Math.round(absNum).toString();
  } else if (absNum >= 10) {
    return sign + absNum.toFixed(1).replace(/\.0$/, '');
  } else if (absNum >= 1) {
    return sign + absNum.toFixed(sigFigs).replace(/\.?0+$/, '');
  } else {
    // For numbers < 1, show enough decimals to get significant figures
    const magnitude = Math.floor(Math.log10(absNum));
    const decimals = Math.max(0, -magnitude + sigFigs - 1);
    const rounded = absNum.toFixed(decimals);
    return sign + rounded.replace(/\.?0+$/, '');
  }
}

/**
 * Format a coefficient for display, handling special cases
 */
function formatCoefficient(coef: number, isFirst: boolean = false): string {
  const rounded = roundToSignificant(coef, 2);
  const num = parseFloat(rounded);

  if (num === 0) return '';
  if (num === 1 && !isFirst) return '+';
  if (num === -1) return '-';
  if (num > 0 && !isFirst) return `+ ${rounded}`;
  if (num < 0) return `- ${Math.abs(num)}`;
  return rounded;
}

/**
 * Convert a plain math expression to proper LaTeX with math symbols
 */
export function expressionToLatex(expr: string): string {
  if (!expr) return '';

  let latex = expr;

  // Remove "y = " prefix - we'll add it back properly
  latex = latex.replace(/^y\s*=\s*/, '');

  // Round all numbers in the expression
  latex = latex.replace(/-?\d+\.?\d*(?:e[+-]?\d+)?/gi, (match) => {
    const num = parseFloat(match);
    if (isNaN(num)) return match;
    return roundToSignificant(num, 2);
  });

  // Convert exp(ax) to e^{ax} with proper formatting
  latex = latex.replace(/exp\(([^)]+)\)/gi, (_, inner) => {
    // Clean up the inner expression
    let cleanInner = inner.trim();
    // Remove leading coefficient multiplication if it's just a number times x
    cleanInner = cleanInner.replace(/^\s*\*\s*/, '');
    return `e^{${cleanInner}}`;
  });

  // Handle trigonometric functions
  latex = latex.replace(/\bsin\(/g, '\\sin(');
  latex = latex.replace(/\bcos\(/g, '\\cos(');
  latex = latex.replace(/\btan\(/g, '\\tan(');

  // Handle logarithmic functions
  latex = latex.replace(/\bln\(/g, '\\ln(');
  latex = latex.replace(/\blog\(/g, '\\log(');

  // Handle sqrt
  latex = latex.replace(/\bsqrt\(([^)]+)\)/g, '\\sqrt{$1}');

  // Handle exponentiation ** BEFORE converting * to \cdot
  // Handle (expression)**n -> (expression)^{n}
  latex = latex.replace(/\)\s*\*\*\s*(\d+)/g, ')^{$1}');
  // Handle variable**n -> variable^{n}
  latex = latex.replace(/([a-zA-Z])\s*\*\*\s*(\d+)/g, '$1^{$2}');
  // Handle number**n -> number^{n}
  latex = latex.replace(/(\d+\.?\d*)\s*\*\*\s*(\d+)/g, '$1^{$2}');

  // Convert fractions: number/(expression)^n -> \frac{number}{(expression)^n}
  latex = latex.replace(/(-?\d+\.?\d*)\s*\/\s*(\([^)]+\)\^?\{?\d*\}?)/g, (_, num, denom) => {
    return `\\frac{${num}}{${denom}}`;
  });

  // Convert fractions: number/(expression) -> \frac{number}{expression}
  latex = latex.replace(/(-?\d+\.?\d*)\s*\/\s*\(([^)]+)\)/g, (_, num, denom) => {
    return `\\frac{${num}}{${denom}}`;
  });

  // Handle polynomial exponents: x^2 -> x^{2}
  latex = latex.replace(/\^(\d+)(?!\})/g, '^{$1}');

  // Convert * to \cdot (middle dot) - only single asterisks now
  latex = latex.replace(/\s*\*\s*/g, ' \\cdot ');

  // Handle minus signs in expressions (convert to proper minus)
  // But be careful not to mess with exponents
  latex = latex.replace(/\s+-\s+/g, ' - ');
  latex = latex.replace(/\s+\+\s+/g, ' + ');

  // Clean up: remove coefficient of 1 before variables (but not 1.x or 10.x)
  latex = latex.replace(/(?<![.\d])1\s*\\cdot\s*([ex])/g, '$1');
  latex = latex.replace(/(?<![.\d])1\s*\\cdot\s*\\sin/g, '\\sin');
  latex = latex.replace(/(?<![.\d])1\s*\\cdot\s*\\cos/g, '\\cos');

  // Clean up: handle -1 coefficient
  latex = latex.replace(/-\s*1\s*\\cdot\s*([ex])/g, '-$1');

  // Clean up double spaces
  latex = latex.replace(/\s+/g, ' ').trim();

  // Handle edge case: + - should be just -
  latex = latex.replace(/\+\s*-/g, '-');
  latex = latex.replace(/-\s*-/g, '+');

  // Clean up leading + sign
  latex = latex.replace(/^\+\s*/, '');

  return latex;
}

/**
 * Convert derivative expression to proper LaTeX
 */
export function derivativeToLatex(expr: string, order: number = 1): string {
  if (!expr) return '';

  const prefix = order === 1 ? "\\frac{dy}{dx} = " : "\\frac{d^2y}{dx^2} = ";
  return prefix + expressionToLatex(expr);
}

/**
 * Convert integral result to LaTeX
 */
export function integralToLatex(value: number, a: number, b: number, expr: string): string {
  const aRounded = roundToSignificant(a, 2);
  const bRounded = roundToSignificant(b, 2);
  const valueRounded = roundToSignificant(value, 3);
  const exprLatex = expressionToLatex(expr);
  return `\\int_{${aRounded}}^{${bRounded}} (${exprLatex}) \\, dx = ${valueRounded}`;
}

export function LatexRenderer({ latex, displayMode = false, className = '' }: LatexRendererProps) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(latex, {
        displayMode,
        throwOnError: false,
        strict: false,
        trust: true,
      });
    } catch (error) {
      console.error('KaTeX error:', error);
      return `<span class="text-red-500">${latex}</span>`;
    }
  }, [latex, displayMode]);

  return (
    <span
      className={className}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

export default LatexRenderer;
