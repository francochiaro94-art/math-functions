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
 * Convert a plain math expression to LaTeX format
 */
export function expressionToLatex(expr: string): string {
  if (!expr) return '';

  let latex = expr;

  // Remove "y = " prefix for cleaner display
  latex = latex.replace(/^y\s*=\s*/, '');

  // Handle special functions
  latex = latex.replace(/\bsin\(/g, '\\sin(');
  latex = latex.replace(/\bcos\(/g, '\\cos(');
  latex = latex.replace(/\btan\(/g, '\\tan(');
  latex = latex.replace(/\bln\(/g, '\\ln(');
  latex = latex.replace(/\blog\(/g, '\\log(');
  latex = latex.replace(/\bexp\(/g, '\\exp(');
  latex = latex.replace(/\bsqrt\(/g, '\\sqrt{');

  // Handle exponents: x^2 -> x^{2}, x^{-1} stays same
  latex = latex.replace(/\^(\d+)/g, '^{$1}');
  latex = latex.replace(/\^(-?\d+\.?\d*)/g, '^{$1}');

  // Handle fractions: a/(b) -> \frac{a}{b}
  // Simple pattern for expressions like "1/(x - c)"
  latex = latex.replace(/(\d+\.?\d*)\s*\/\s*\(([^)]+)\)/g, '\\frac{$1}{$2}');

  // Handle multiplication: 2.5 * x -> 2.5x, but keep spaces around +/-
  latex = latex.replace(/\s*\*\s*/g, ' \\cdot ');

  // Clean up double spaces
  latex = latex.replace(/\s+/g, ' ');

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
  const exprLatex = expressionToLatex(expr);
  return `\\int_{${a.toFixed(2)}}^{${b.toFixed(2)}} (${exprLatex}) \\, dx = ${value.toFixed(4)}`;
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
