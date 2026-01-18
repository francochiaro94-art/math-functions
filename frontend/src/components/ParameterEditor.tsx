'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import type { ModelParameterSchema, ModelParameter } from '@/types/chart';

// Helper to parse number with locale tolerance (comma as decimal separator)
function parseNumber(raw: string | number | null | undefined): number {
  if (raw === '' || raw === null || raw === undefined) return NaN;
  if (typeof raw === 'number') return raw;
  if (typeof raw === 'string') {
    const normalized = raw.trim().replace(',', '.');
    return Number(normalized);
  }
  return NaN;
}

// Helper to check if a bound is a valid finite number
function isValidBound(value: number | null | undefined): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

interface ParameterEditorProps {
  schema: ModelParameterSchema;
  onApply: (params: Record<string, number>) => void;
  onCancel: () => void;
  onPreview: (params: Record<string, number>) => void;
  onDraftChange?: (hasChanges: boolean) => void;
  isLoading?: boolean;
  error?: string | null;
}

export function ParameterEditor({
  schema,
  onApply,
  onCancel,
  onPreview,
  onDraftChange,
  isLoading,
  error,
}: ParameterEditorProps) {
  // Draft parameters for editing
  const [draftParams, setDraftParams] = useState<Record<string, number>>({});
  const [originalParams, setOriginalParams] = useState<Record<string, number>>({});
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});
  const [isDirty, setIsDirty] = useState(false);

  // Debounce timer ref
  const previewTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Initialize parameters from schema
  useEffect(() => {
    const params: Record<string, number> = {};
    schema.parameters.forEach((p) => {
      params[p.name] = p.value;
    });
    setDraftParams(params);
    setOriginalParams(params);
    setIsDirty(false);
    setValidationErrors({});
  }, [schema]);

  // Notify parent when draft changes
  useEffect(() => {
    onDraftChange?.(isDirty);
  }, [isDirty, onDraftChange]);

  // Validate a single parameter
  const validateParam = useCallback((param: ModelParameter, value: number): string | null => {
    if (!Number.isFinite(value)) {
      return 'Must be a valid number';
    }
    // Only apply min/max validation if bounds are valid finite numbers
    if (isValidBound(param.min) && value < param.min) {
      return `Must be at least ${param.min}`;
    }
    if (isValidBound(param.max) && value > param.max) {
      return `Must be at most ${param.max}`;
    }
    return null;
  }, []);

  // Handle parameter change
  const handleParamChange = useCallback((name: string, value: string) => {
    // Use locale-tolerant parsing (comma -> dot)
    const numValue = parseNumber(value);
    const param = schema.parameters.find((p) => p.name === name);

    // Update draft (keep 0 for empty/invalid to allow continued editing)
    setDraftParams((prev) => ({ ...prev, [name]: Number.isFinite(numValue) ? numValue : 0 }));
    setIsDirty(true);

    // Validate
    if (param) {
      const error = validateParam(param, numValue);
      setValidationErrors((prev) => {
        if (error) {
          return { ...prev, [name]: error };
        }
        const { [name]: _, ...rest } = prev;
        return rest;
      });
    }

    // Debounced preview
    if (previewTimerRef.current) {
      clearTimeout(previewTimerRef.current);
    }

    if (Number.isFinite(numValue) && Object.keys(validationErrors).length === 0) {
      previewTimerRef.current = setTimeout(() => {
        onPreview({ ...draftParams, [name]: numValue });
      }, 150);
    }
  }, [schema.parameters, validateParam, validationErrors, draftParams, onPreview]);

  // Handle apply
  const handleApply = useCallback(() => {
    // Final validation
    const errors: Record<string, string> = {};
    schema.parameters.forEach((param) => {
      const error = validateParam(param, draftParams[param.name]);
      if (error) errors[param.name] = error;
    });

    if (Object.keys(errors).length > 0) {
      setValidationErrors(errors);
      return;
    }

    onApply(draftParams);
    setOriginalParams(draftParams);
    setIsDirty(false);
  }, [schema.parameters, draftParams, validateParam, onApply]);

  // Handle reset to original fitted values
  const handleReset = useCallback(() => {
    const params: Record<string, number> = {};
    schema.parameters.forEach((p) => {
      params[p.name] = p.value;
    });
    setDraftParams(params);
    setValidationErrors({});
    setIsDirty(false);
    onPreview(params);
  }, [schema.parameters, onPreview]);

  const hasErrors = Object.keys(validationErrors).length > 0;

  return (
    <div className="space-y-4">
      {/* Expression template (locked) */}
      <div className="p-2 bg-zinc-100 dark:bg-zinc-800 rounded text-sm font-mono text-zinc-600 dark:text-zinc-400">
        {schema.expressionTemplate}
      </div>

      {/* Parameter inputs */}
      <div className="space-y-3">
        {schema.parameters.map((param) => (
          <div key={param.name} className="space-y-1">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
                {param.label}
                {param.hint && (
                  <span className="ml-1 text-zinc-400 dark:text-zinc-500 font-normal">
                    ({param.hint})
                  </span>
                )}
              </label>
              <span className="text-[10px] text-zinc-400 dark:text-zinc-500 font-mono">
                {param.name}
              </span>
            </div>
            <div className="relative">
              <input
                type="text"
                inputMode="decimal"
                value={draftParams[param.name] ?? 0}
                onChange={(e) => handleParamChange(param.name, e.target.value)}
                className={`w-full px-3 py-2 text-sm font-mono rounded-lg bg-white dark:bg-zinc-900 border ${
                  validationErrors[param.name]
                    ? 'border-red-500 focus:ring-red-500'
                    : 'border-zinc-300 dark:border-zinc-700 focus:ring-green-500'
                } focus:outline-none focus:ring-2`}
              />
              {validationErrors[param.name] && (
                <p className="mt-1 text-xs text-red-500">
                  {validationErrors[param.name]}
                </p>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Error message from backend */}
      {error && (
        <div className="p-2 bg-red-50 dark:bg-red-900/20 rounded text-sm text-red-600 dark:text-red-400">
          {error}
        </div>
      )}

      {/* Action buttons */}
      <div className="flex gap-2">
        <button
          onClick={handleApply}
          disabled={hasErrors || isLoading}
          className="flex-1 px-3 py-2 text-sm font-medium rounded-lg bg-green-500 text-white hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isLoading ? 'Applying...' : 'Apply'}
        </button>
        <button
          onClick={handleReset}
          disabled={!isDirty && !error}
          className="px-3 py-2 text-sm font-medium rounded-lg bg-zinc-200 dark:bg-zinc-700 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-300 dark:hover:bg-zinc-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Reset
        </button>
        <button
          onClick={onCancel}
          className="px-3 py-2 text-sm font-medium rounded-lg border border-zinc-300 dark:border-zinc-700 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors"
        >
          Cancel
        </button>
      </div>

      {/* Dirty indicator */}
      {isDirty && !hasErrors && (
        <p className="text-xs text-zinc-500 dark:text-zinc-400 text-center">
          Preview showing. Click Apply to save changes.
        </p>
      )}
    </div>
  );
}
