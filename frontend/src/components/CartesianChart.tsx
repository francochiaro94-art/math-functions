'use client';

import { useRef, useEffect, useCallback, useState } from 'react';
import * as d3 from 'd3';
import { useChartDimensions } from '@/hooks/useChartDimensions';
import type { Point, ChartBounds, FittedCurve } from '@/types/chart';

interface CartesianChartProps {
  points: Point[];
  fittedCurve?: FittedCurve | null;
  onPointAdd?: (point: Point) => void;
  isPaintingMode?: boolean;
  integralRange?: { a: Point; b: Point } | null;
  analyticalMarkers?: {
    extrema?: { type: 'maximum' | 'minimum'; x: number; y: number }[];
    asymptotes?: { type: 'vertical' | 'horizontal'; value: number }[];
  };
}

const DEFAULT_BOUNDS: ChartBounds = {
  xMin: -10,
  xMax: 10,
  yMin: -10,
  yMax: 10,
};

// Zoom constraints
const MIN_RANGE = 0.1;
const MAX_RANGE = 1e6;
const ZOOM_SENSITIVITY = 0.002;
const PAN_THRESHOLD = 4; // pixels before committing to pan

type InteractionState = 'idle' | 'panning' | 'painting' | 'potentialPan';

export function CartesianChart({
  points,
  fittedCurve,
  onPointAdd,
  isPaintingMode = false,
  integralRange,
  analyticalMarkers,
}: CartesianChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const dimensions = useChartDimensions(containerRef);
  const [bounds, setBounds] = useState<ChartBounds>(DEFAULT_BOUNDS);
  const [hoveredPoint, setHoveredPoint] = useState<Point | null>(null);

  // Interaction state
  const [interactionState, setInteractionState] = useState<InteractionState>('idle');
  const [isSpacePressed, setIsSpacePressed] = useState(false);
  const panStartRef = useRef<{ screenX: number; screenY: number; bounds: ChartBounds } | null>(null);
  const rafRef = useRef<number | null>(null);

  const innerWidth = dimensions.width - dimensions.margin.left - dimensions.margin.right;
  const innerHeight = dimensions.height - dimensions.margin.top - dimensions.margin.bottom;

  // Coordinate conversion helpers
  const screenToWorld = useCallback((screenX: number, screenY: number) => {
    const x = bounds.xMin + (screenX / innerWidth) * (bounds.xMax - bounds.xMin);
    const y = bounds.yMax - (screenY / innerHeight) * (bounds.yMax - bounds.yMin);
    return { x, y };
  }, [bounds, innerWidth, innerHeight]);

  const worldToScreen = useCallback((worldX: number, worldY: number) => {
    const screenX = ((worldX - bounds.xMin) / (bounds.xMax - bounds.xMin)) * innerWidth;
    const screenY = ((bounds.yMax - worldY) / (bounds.yMax - bounds.yMin)) * innerHeight;
    return { screenX, screenY };
  }, [bounds, innerWidth, innerHeight]);

  // Create scales for D3
  const xScale = useCallback(() => {
    return d3.scaleLinear()
      .domain([bounds.xMin, bounds.xMax])
      .range([0, innerWidth]);
  }, [bounds.xMin, bounds.xMax, innerWidth]);

  const yScale = useCallback(() => {
    return d3.scaleLinear()
      .domain([bounds.yMin, bounds.yMax])
      .range([innerHeight, 0]);
  }, [bounds.yMin, bounds.yMax, innerHeight]);

  // Clamp viewport to limits
  const clampBounds = useCallback((newBounds: ChartBounds): ChartBounds => {
    let { xMin, xMax, yMin, yMax } = newBounds;

    // Enforce minimum range
    const xRange = xMax - xMin;
    const yRange = yMax - yMin;

    if (xRange < MIN_RANGE) {
      const mid = (xMin + xMax) / 2;
      xMin = mid - MIN_RANGE / 2;
      xMax = mid + MIN_RANGE / 2;
    }
    if (yRange < MIN_RANGE) {
      const mid = (yMin + yMax) / 2;
      yMin = mid - MIN_RANGE / 2;
      yMax = mid + MIN_RANGE / 2;
    }

    // Enforce maximum range
    if (xRange > MAX_RANGE) {
      const mid = (xMin + xMax) / 2;
      xMin = mid - MAX_RANGE / 2;
      xMax = mid + MAX_RANGE / 2;
    }
    if (yRange > MAX_RANGE) {
      const mid = (yMin + yMax) / 2;
      yMin = mid - MAX_RANGE / 2;
      yMax = mid + MAX_RANGE / 2;
    }

    return { xMin, xMax, yMin, yMax };
  }, []);

  // Cursor-anchored zoom
  const applyZoom = useCallback((zoomFactor: number, anchorScreenX: number, anchorScreenY: number) => {
    setBounds(prevBounds => {
      // Get world coordinate under cursor before zoom
      const xRange = prevBounds.xMax - prevBounds.xMin;
      const yRange = prevBounds.yMax - prevBounds.yMin;
      const worldX = prevBounds.xMin + (anchorScreenX / innerWidth) * xRange;
      const worldY = prevBounds.yMax - (anchorScreenY / innerHeight) * yRange;

      // Apply zoom factor
      const newXRange = xRange / zoomFactor;
      const newYRange = yRange / zoomFactor;

      // Recompute bounds so world point stays under cursor
      const ratioX = anchorScreenX / innerWidth;
      const ratioY = anchorScreenY / innerHeight;

      const newXMin = worldX - ratioX * newXRange;
      const newXMax = worldX + (1 - ratioX) * newXRange;
      const newYMax = worldY + ratioY * newYRange;
      const newYMin = worldY - (1 - ratioY) * newYRange;

      return clampBounds({ xMin: newXMin, xMax: newXMax, yMin: newYMin, yMax: newYMax });
    });
  }, [innerWidth, innerHeight, clampBounds]);

  // Pan by screen delta
  const applyPan = useCallback((deltaScreenX: number, deltaScreenY: number) => {
    setBounds(prevBounds => {
      const xRange = prevBounds.xMax - prevBounds.xMin;
      const yRange = prevBounds.yMax - prevBounds.yMin;

      const deltaWorldX = (deltaScreenX / innerWidth) * xRange;
      const deltaWorldY = (deltaScreenY / innerHeight) * yRange;

      return clampBounds({
        xMin: prevBounds.xMin - deltaWorldX,
        xMax: prevBounds.xMax - deltaWorldX,
        yMin: prevBounds.yMin + deltaWorldY,
        yMax: prevBounds.yMax + deltaWorldY,
      });
    });
  }, [innerWidth, innerHeight, clampBounds]);

  // Reset view
  const resetView = useCallback(() => {
    setBounds(DEFAULT_BOUNDS);
  }, []);

  // Handle wheel/pinch zoom
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;

    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();

      const rect = svg.getBoundingClientRect();
      const screenX = event.clientX - rect.left - dimensions.margin.left;
      const screenY = event.clientY - rect.top - dimensions.margin.top;

      // Check if pointer is within chart area
      if (screenX < 0 || screenX > innerWidth || screenY < 0 || screenY > innerHeight) {
        return;
      }

      // Pinch zoom (trackpad) vs scroll
      if (event.ctrlKey || event.metaKey) {
        // Pinch zoom - deltaY is zoom amount
        const zoomFactor = Math.pow(2, -event.deltaY * ZOOM_SENSITIVITY * 10);

        if (rafRef.current) cancelAnimationFrame(rafRef.current);
        rafRef.current = requestAnimationFrame(() => {
          applyZoom(zoomFactor, screenX, screenY);
        });
      } else {
        // Mouse wheel zoom or trackpad two-finger scroll
        // Detect if it's a trackpad scroll (usually has both deltaX and deltaY, and smaller values)
        const isTrackpadScroll = Math.abs(event.deltaX) > 0 || (Math.abs(event.deltaY) < 50 && !Number.isInteger(event.deltaY));

        if (isTrackpadScroll) {
          // Two-finger scroll = pan
          if (rafRef.current) cancelAnimationFrame(rafRef.current);
          rafRef.current = requestAnimationFrame(() => {
            applyPan(event.deltaX, event.deltaY);
          });
        } else {
          // Mouse wheel = zoom
          const zoomFactor = Math.pow(2, -event.deltaY * ZOOM_SENSITIVITY);

          if (rafRef.current) cancelAnimationFrame(rafRef.current);
          rafRef.current = requestAnimationFrame(() => {
            applyZoom(zoomFactor, screenX, screenY);
          });
        }
      }
    };

    svg.addEventListener('wheel', handleWheel, { passive: false });

    return () => {
      svg.removeEventListener('wheel', handleWheel);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [applyZoom, applyPan, dimensions.margin, innerWidth, innerHeight]);

  // Handle keyboard for space key (pan modifier)
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.code === 'Space' && !event.repeat) {
        event.preventDefault();
        setIsSpacePressed(true);
      }
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      if (event.code === 'Space') {
        setIsSpacePressed(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, []);

  // Handle mouse down for pan/paint
  const handleMouseDown = useCallback((event: React.MouseEvent<SVGSVGElement>) => {
    if (event.button !== 0) return; // Only left click

    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;

    const screenX = event.clientX - rect.left - dimensions.margin.left;
    const screenY = event.clientY - rect.top - dimensions.margin.top;

    // Check if within chart area
    if (screenX < 0 || screenX > innerWidth || screenY < 0 || screenY > innerHeight) {
      return;
    }

    // If space is pressed, always pan
    if (isSpacePressed) {
      setInteractionState('panning');
      panStartRef.current = { screenX: event.clientX, screenY: event.clientY, bounds };
      return;
    }

    // If painting mode is off, pan directly
    if (!isPaintingMode) {
      setInteractionState('potentialPan');
      panStartRef.current = { screenX: event.clientX, screenY: event.clientY, bounds };
      return;
    }

    // In painting mode, wait to see if it's a click or drag
    setInteractionState('potentialPan');
    panStartRef.current = { screenX: event.clientX, screenY: event.clientY, bounds };
  }, [isPaintingMode, isSpacePressed, bounds, dimensions.margin, innerWidth, innerHeight]);

  // Handle mouse move for pan
  const handleMouseMove = useCallback((event: React.MouseEvent<SVGSVGElement>) => {
    if (!panStartRef.current) return;

    const deltaX = event.clientX - panStartRef.current.screenX;
    const deltaY = event.clientY - panStartRef.current.screenY;
    const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);

    if (interactionState === 'potentialPan' && distance > PAN_THRESHOLD) {
      setInteractionState('panning');
    }

    if (interactionState === 'panning') {
      const startBounds = panStartRef.current.bounds;
      const xRange = startBounds.xMax - startBounds.xMin;
      const yRange = startBounds.yMax - startBounds.yMin;

      const deltaWorldX = (deltaX / innerWidth) * xRange;
      const deltaWorldY = (deltaY / innerHeight) * yRange;

      setBounds(clampBounds({
        xMin: startBounds.xMin - deltaWorldX,
        xMax: startBounds.xMax - deltaWorldX,
        yMin: startBounds.yMin + deltaWorldY,
        yMax: startBounds.yMax + deltaWorldY,
      }));
    }
  }, [interactionState, innerWidth, innerHeight, clampBounds]);

  // Handle mouse up
  const handleMouseUp = useCallback((event: React.MouseEvent<SVGSVGElement>) => {
    const wasInteractionState = interactionState;

    // If we were in potentialPan and didn't exceed threshold, it's a click
    if (wasInteractionState === 'potentialPan' && isPaintingMode && onPointAdd) {
      const rect = svgRef.current?.getBoundingClientRect();
      if (rect) {
        const screenX = event.clientX - rect.left - dimensions.margin.left;
        const screenY = event.clientY - rect.top - dimensions.margin.top;

        if (screenX >= 0 && screenX <= innerWidth && screenY >= 0 && screenY <= innerHeight) {
          const world = screenToWorld(screenX, screenY);
          onPointAdd({
            x: world.x,
            y: world.y,
            id: crypto.randomUUID(),
          });
        }
      }
    }

    setInteractionState('idle');
    panStartRef.current = null;
  }, [interactionState, isPaintingMode, onPointAdd, screenToWorld, dimensions.margin, innerWidth, innerHeight]);

  // Handle mouse leave
  const handleMouseLeave = useCallback(() => {
    setInteractionState('idle');
    panStartRef.current = null;
  }, []);

  // Calculate appropriate tick count based on zoom level
  const getTickCount = useCallback((range: number) => {
    if (range <= 2) return 20;
    if (range <= 5) return 10;
    if (range <= 20) return 10;
    if (range <= 50) return 10;
    return 10;
  }, []);

  // Draw chart
  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const x = xScale();
    const y = yScale();

    // Create main group with margin transform
    const g = svg
      .append('g')
      .attr('transform', `translate(${dimensions.margin.left},${dimensions.margin.top})`);

    // Calculate tick counts based on current range
    const xRange = bounds.xMax - bounds.xMin;
    const yRange = bounds.yMax - bounds.yMin;
    const xTickCount = getTickCount(xRange);
    const yTickCount = getTickCount(yRange);

    // Draw subtle gridlines
    const xTicks = x.ticks(xTickCount);
    const yTicks = y.ticks(yTickCount);

    // Vertical gridlines
    g.selectAll('.grid-line-x')
      .data(xTicks)
      .enter()
      .append('line')
      .attr('class', 'grid-line-x')
      .attr('x1', d => x(d))
      .attr('x2', d => x(d))
      .attr('y1', 0)
      .attr('y2', innerHeight)
      .attr('stroke', 'currentColor')
      .attr('stroke-opacity', 0.08)
      .attr('stroke-width', 1);

    // Horizontal gridlines
    g.selectAll('.grid-line-y')
      .data(yTicks)
      .enter()
      .append('line')
      .attr('class', 'grid-line-y')
      .attr('x1', 0)
      .attr('x2', innerWidth)
      .attr('y1', d => y(d))
      .attr('y2', d => y(d))
      .attr('stroke', 'currentColor')
      .attr('stroke-opacity', 0.08)
      .attr('stroke-width', 1);

    // Draw X axis
    const xAxisY = Math.max(0, Math.min(innerHeight, y(0)));
    g.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${xAxisY})`)
      .call(d3.axisBottom(x).ticks(xTickCount).tickSize(6))
      .call(g => g.select('.domain').attr('stroke', 'currentColor').attr('stroke-opacity', 0.4))
      .call(g => g.selectAll('.tick line').attr('stroke', 'currentColor').attr('stroke-opacity', 0.4))
      .call(g => g.selectAll('.tick text')
        .attr('fill', 'currentColor')
        .attr('opacity', 0.7)
        .style('font-size', '11px')
        .style('font-family', 'var(--font-geist-mono)')
      );

    // Draw Y axis
    const yAxisX = Math.max(0, Math.min(innerWidth, x(0)));
    g.append('g')
      .attr('class', 'y-axis')
      .attr('transform', `translate(${yAxisX},0)`)
      .call(d3.axisLeft(y).ticks(yTickCount).tickSize(6))
      .call(g => g.select('.domain').attr('stroke', 'currentColor').attr('stroke-opacity', 0.4))
      .call(g => g.selectAll('.tick line').attr('stroke', 'currentColor').attr('stroke-opacity', 0.4))
      .call(g => g.selectAll('.tick text')
        .attr('fill', 'currentColor')
        .attr('opacity', 0.7)
        .style('font-size', '11px')
        .style('font-family', 'var(--font-geist-mono)')
      );

    // Axis labels
    g.append('text')
      .attr('class', 'axis-label')
      .attr('x', innerWidth)
      .attr('y', xAxisY - 10)
      .attr('text-anchor', 'end')
      .attr('fill', 'currentColor')
      .attr('opacity', 0.5)
      .style('font-size', '12px')
      .style('font-weight', '500')
      .text('x');

    g.append('text')
      .attr('class', 'axis-label')
      .attr('x', yAxisX + 10)
      .attr('y', 5)
      .attr('text-anchor', 'start')
      .attr('fill', 'currentColor')
      .attr('opacity', 0.5)
      .style('font-size', '12px')
      .style('font-weight', '500')
      .text('y');

    // Draw integral shading if present
    if (integralRange && fittedCurve) {
      const aX = Math.min(integralRange.a.x, integralRange.b.x);
      const bX = Math.max(integralRange.a.x, integralRange.b.x);

      const curvePointsInRange = fittedCurve.points.filter(p => p.x >= aX && p.x <= bX);

      if (curvePointsInRange.length > 0) {
        const areaGenerator = d3.area<Point>()
          .x(d => x(d.x))
          .y0(y(0))
          .y1(d => y(d.y))
          .curve(d3.curveMonotoneX);

        g.append('path')
          .datum(curvePointsInRange)
          .attr('fill', '#22c55e')
          .attr('fill-opacity', 0.2)
          .attr('d', areaGenerator);

        // Vertical lines at A and B
        g.append('line')
          .attr('x1', x(aX))
          .attr('x2', x(aX))
          .attr('y1', y(0))
          .attr('y2', y(curvePointsInRange[0]?.y ?? 0))
          .attr('stroke', '#22c55e')
          .attr('stroke-width', 2)
          .attr('stroke-dasharray', '4,4');

        g.append('line')
          .attr('x1', x(bX))
          .attr('x2', x(bX))
          .attr('y1', y(0))
          .attr('y2', y(curvePointsInRange[curvePointsInRange.length - 1]?.y ?? 0))
          .attr('stroke', '#22c55e')
          .attr('stroke-width', 2)
          .attr('stroke-dasharray', '4,4');
      }
    }

    // Draw fitted curve with discontinuity handling and extrapolation styling
    if (fittedCurve && fittedCurve.points.length > 0) {
      const lineGenerator = d3.line<Point>()
        .x(d => x(d.x))
        .y(d => y(d.y))
        .curve(d3.curveMonotoneX);

      // Determine data range from user points (for interpolation vs extrapolation)
      const dataXMin = points.length > 0 ? Math.min(...points.map(p => p.x)) : bounds.xMin;
      const dataXMax = points.length > 0 ? Math.max(...points.map(p => p.x)) : bounds.xMax;

      // Split curve into segments at discontinuities (NaN, Infinity, large jumps)
      const DISCONTINUITY_THRESHOLD = (bounds.yMax - bounds.yMin) * 0.5;

      // Helper to split points into continuous segments
      const splitIntoSegments = (curvePoints: Point[]): Point[][] => {
        const segments: Point[][] = [];
        let currentSegment: Point[] = [];

        for (let i = 0; i < curvePoints.length; i++) {
          const point = curvePoints[i];
          const isValid = Number.isFinite(point.y) && Number.isFinite(point.x);

          if (!isValid) {
            if (currentSegment.length > 1) {
              segments.push(currentSegment);
            }
            currentSegment = [];
            continue;
          }

          if (currentSegment.length > 0) {
            const prevPoint = currentSegment[currentSegment.length - 1];
            const yJump = Math.abs(point.y - prevPoint.y);
            const xStep = Math.abs(point.x - prevPoint.x);

            if (xStep > 0 && yJump > DISCONTINUITY_THRESHOLD) {
              if (currentSegment.length > 1) {
                segments.push(currentSegment);
              }
              currentSegment = [];
            }
          }

          currentSegment.push(point);
        }

        if (currentSegment.length > 1) {
          segments.push(currentSegment);
        }

        return segments;
      };

      // Separate curve points into interpolation and extrapolation regions
      const interpolationPoints = fittedCurve.points.filter(p => p.x >= dataXMin && p.x <= dataXMax);
      const leftExtrapolationPoints = fittedCurve.points.filter(p => p.x < dataXMin);
      const rightExtrapolationPoints = fittedCurve.points.filter(p => p.x > dataXMax);

      // Draw interpolation (solid line)
      const interpolationSegments = splitIntoSegments(interpolationPoints);
      interpolationSegments.forEach(segment => {
        g.append('path')
          .datum(segment)
          .attr('fill', 'none')
          .attr('stroke', fittedCurve.color || '#22c55e')
          .attr('stroke-width', 2.5)
          .attr('d', lineGenerator);
      });

      // Draw left extrapolation (dotted line)
      const leftSegments = splitIntoSegments(leftExtrapolationPoints);
      leftSegments.forEach(segment => {
        g.append('path')
          .datum(segment)
          .attr('fill', 'none')
          .attr('stroke', fittedCurve.color || '#22c55e')
          .attr('stroke-width', 2)
          .attr('stroke-dasharray', '6,4')
          .attr('stroke-opacity', 0.7)
          .attr('d', lineGenerator);
      });

      // Draw right extrapolation (dotted line)
      const rightSegments = splitIntoSegments(rightExtrapolationPoints);
      rightSegments.forEach(segment => {
        g.append('path')
          .datum(segment)
          .attr('fill', 'none')
          .attr('stroke', fittedCurve.color || '#22c55e')
          .attr('stroke-width', 2)
          .attr('stroke-dasharray', '6,4')
          .attr('stroke-opacity', 0.7)
          .attr('d', lineGenerator);
      });
    }

    // Draw analytical markers
    if (analyticalMarkers) {
      // Draw extrema points
      if (analyticalMarkers.extrema) {
        analyticalMarkers.extrema.forEach(ext => {
          const markerGroup = g.append('g')
            .attr('transform', `translate(${x(ext.x)},${y(ext.y)})`);

          markerGroup.append('circle')
            .attr('r', 6)
            .attr('fill', ext.type === 'maximum' ? '#ef4444' : '#3b82f6')
            .attr('stroke', 'white')
            .attr('stroke-width', 2);

          markerGroup.append('text')
            .attr('y', -12)
            .attr('text-anchor', 'middle')
            .attr('fill', 'currentColor')
            .attr('opacity', 0.8)
            .style('font-size', '10px')
            .style('font-family', 'var(--font-geist-mono)')
            .text(`${ext.type === 'maximum' ? 'max' : 'min'} (${ext.x.toFixed(2)}, ${ext.y.toFixed(2)})`);
        });
      }

      // Draw asymptotes
      if (analyticalMarkers.asymptotes) {
        analyticalMarkers.asymptotes.forEach(asymp => {
          if (asymp.type === 'vertical') {
            g.append('line')
              .attr('x1', x(asymp.value))
              .attr('x2', x(asymp.value))
              .attr('y1', 0)
              .attr('y2', innerHeight)
              .attr('stroke', '#f59e0b')
              .attr('stroke-width', 1.5)
              .attr('stroke-dasharray', '6,4');
          } else if (asymp.type === 'horizontal') {
            g.append('line')
              .attr('x1', 0)
              .attr('x2', innerWidth)
              .attr('y1', y(asymp.value))
              .attr('y2', y(asymp.value))
              .attr('stroke', '#f59e0b')
              .attr('stroke-width', 1.5)
              .attr('stroke-dasharray', '6,4');
          }
        });
      }
    }

    // Draw data points
    if (points.length > 0) {
      g.selectAll('.data-point')
        .data(points)
        .enter()
        .append('circle')
        .attr('class', 'data-point')
        .attr('cx', d => x(d.x))
        .attr('cy', d => y(d.y))
        .attr('r', 4)
        .attr('fill', '#6366f1')
        .attr('stroke', 'white')
        .attr('stroke-width', 1.5)
        .style('cursor', 'pointer')
        .on('mouseenter', function(event, d) {
          d3.select(this).attr('r', 6);
          setHoveredPoint(d);
        })
        .on('mouseleave', function() {
          d3.select(this).attr('r', 4);
          setHoveredPoint(null);
        });
    }

    // Clip path for chart area
    svg.append('defs')
      .append('clipPath')
      .attr('id', 'chart-clip')
      .append('rect')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', innerWidth)
      .attr('height', innerHeight);

  }, [dimensions, bounds, points, fittedCurve, integralRange, analyticalMarkers, xScale, yScale, innerWidth, innerHeight, getTickCount]);

  // Determine cursor style
  const getCursorClass = () => {
    if (interactionState === 'panning') return 'cursor-grabbing';
    if (isSpacePressed) return 'cursor-grab';
    if (isPaintingMode) return 'cursor-crosshair';
    return 'cursor-grab';
  };

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full min-h-[400px] bg-zinc-50 dark:bg-zinc-900 rounded-xl overflow-hidden"
    >
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        className={getCursorClass()}
      />

      {/* Zoom controls */}
      <div className="absolute top-4 left-4 flex flex-col gap-1">
        <button
          onClick={() => applyZoom(1.5, innerWidth / 2, innerHeight / 2)}
          className="w-8 h-8 flex items-center justify-center bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-lg shadow-sm hover:bg-zinc-50 dark:hover:bg-zinc-700 transition-colors"
          title="Zoom in"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
        </button>
        <button
          onClick={() => applyZoom(0.67, innerWidth / 2, innerHeight / 2)}
          className="w-8 h-8 flex items-center justify-center bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-lg shadow-sm hover:bg-zinc-50 dark:hover:bg-zinc-700 transition-colors"
          title="Zoom out"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
          </svg>
        </button>
        <button
          onClick={resetView}
          className="w-8 h-8 flex items-center justify-center bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-lg shadow-sm hover:bg-zinc-50 dark:hover:bg-zinc-700 transition-colors"
          title="Reset view"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
          </svg>
        </button>
      </div>

      {/* Interaction hint */}
      <div className="absolute top-4 right-4 text-[10px] text-zinc-400 dark:text-zinc-500 bg-white/80 dark:bg-zinc-800/80 px-2 py-1 rounded backdrop-blur-sm">
        Scroll to zoom {isPaintingMode ? '• Space+drag to pan' : '• Drag to pan'}
      </div>

      {/* Hover tooltip */}
      {hoveredPoint && (
        <div className="absolute top-12 right-4 bg-zinc-800/90 dark:bg-zinc-100/90 text-white dark:text-zinc-900 px-3 py-2 rounded-lg text-sm font-mono shadow-lg backdrop-blur-sm">
          ({hoveredPoint.x.toFixed(4)}, {hoveredPoint.y.toFixed(4)})
        </div>
      )}

      {/* Origin indicator */}
      <div className="absolute bottom-4 left-4 text-xs text-zinc-400 dark:text-zinc-500 font-mono">
        x: [{bounds.xMin.toFixed(1)}, {bounds.xMax.toFixed(1)}] | y: [{bounds.yMin.toFixed(1)}, {bounds.yMax.toFixed(1)}]
      </div>
    </div>
  );
}
