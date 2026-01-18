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

  const innerWidth = dimensions.width - dimensions.margin.left - dimensions.margin.right;
  const innerHeight = dimensions.height - dimensions.margin.top - dimensions.margin.bottom;

  // Create scales
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

  // Handle zoom and pan
  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 100])
      .on('zoom', (event) => {
        const transform = event.transform;
        const x = xScale();
        const y = yScale();

        const newXMin = x.invert(-transform.x / transform.k);
        const newXMax = x.invert((innerWidth - transform.x) / transform.k);
        const newYMin = y.invert((innerHeight - transform.y) / transform.k);
        const newYMax = y.invert(-transform.y / transform.k);

        setBounds({
          xMin: newXMin,
          xMax: newXMax,
          yMin: newYMin,
          yMax: newYMax,
        });
      });

    svg.call(zoom);

    return () => {
      svg.on('.zoom', null);
    };
  }, [innerWidth, innerHeight, xScale, yScale]);

  // Handle click for painting mode
  const handleClick = useCallback((event: React.MouseEvent<SVGSVGElement>) => {
    if (!isPaintingMode || !onPointAdd) return;

    const svg = svgRef.current;
    if (!svg) return;

    const rect = svg.getBoundingClientRect();
    const x = xScale();
    const y = yScale();

    const clickX = event.clientX - rect.left - dimensions.margin.left;
    const clickY = event.clientY - rect.top - dimensions.margin.top;

    if (clickX >= 0 && clickX <= innerWidth && clickY >= 0 && clickY <= innerHeight) {
      const dataX = x.invert(clickX);
      const dataY = y.invert(clickY);

      onPointAdd({
        x: dataX,
        y: dataY,
        id: crypto.randomUUID(),
      });
    }
  }, [isPaintingMode, onPointAdd, xScale, yScale, dimensions.margin, innerWidth, innerHeight]);

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full min-h-[400px] bg-zinc-50 dark:bg-zinc-900 rounded-xl overflow-hidden"
    >
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        onClick={handleClick}
        className={`${isPaintingMode ? 'cursor-crosshair' : 'cursor-grab active:cursor-grabbing'}`}
      />

      {/* Hover tooltip */}
      {hoveredPoint && (
        <div className="absolute top-4 right-4 bg-zinc-800/90 dark:bg-zinc-100/90 text-white dark:text-zinc-900 px-3 py-2 rounded-lg text-sm font-mono shadow-lg backdrop-blur-sm">
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
