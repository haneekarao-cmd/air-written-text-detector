import React, { memo } from 'react';
import { PenLine, WandSparkles } from 'lucide-react';
import type { GestureState, Point } from '@/types';

interface AirWritingCanvasProps {
  trajectory?: Point[];
  strokes?: Point[][];
  gestureState: GestureState;
  recognizedWord: string;
  croppedImageBase64?: string;
}

export const AirWritingCanvas = memo(function AirWritingCanvas({
  strokes = [],
  gestureState,
  recognizedWord,
  croppedImageBase64,
}: AirWritingCanvasProps) {
  const isWriting = gestureState === 'writing';
  const isProcessing = gestureState === 'processing';
  const isActive = isWriting || isProcessing;

  // Convert points to continuous, anti-aliased quadratic Bezier curve
  const renderSmoothStrokePath = (pts: Point[]): string => {
    if (!pts || pts.length === 0) return '';
    if (pts.length === 1) {
      return `M ${pts[0].x} ${pts[0].y} l 0.1 0.1`;
    }
    if (pts.length === 2) {
      return `M ${pts[0].x} ${pts[0].y} L ${pts[1].x} ${pts[1].y}`;
    }

    let d = `M ${pts[0].x} ${pts[0].y}`;
    for (let i = 1; i < pts.length - 1; i++) {
      const xc = (pts[i].x + pts[i + 1].x) / 2;
      const yc = (pts[i].y + pts[i + 1].y) / 2;
      d += ` Q ${pts[i].x} ${pts[i].y}, ${xc} ${yc}`;
    }
    d += ` L ${pts[pts.length - 1].x} ${pts[pts.length - 1].y}`;
    return d;
  };

  const hasStrokes = strokes && strokes.length > 0 && strokes.some((s) => s.length > 0);
  const totalPoints = strokes.reduce((acc, s) => acc + s.length, 0);

  // Active fingertip coordinate for live target indicator
  const activeTip = isWriting && strokes.length > 0 && strokes[strokes.length - 1].length > 0
    ? strokes[strokes.length - 1][strokes[strokes.length - 1].length - 1]
    : null;

  return (
    <section className="writing-panel panel-surface">
      <div className="panel-heading">
        <div>
          <div className="eyebrow"><PenLine size={13} /> Continuous Handwriting Canvas</div>
          <h2>Trajectory canvas</h2>
        </div>
        <div className={`canvas-state ${isActive ? 'active' : ''}`}>
          <span /> {isWriting ? 'Writing in progress' : isProcessing ? 'Recognizing...' : 'Standing by'}
        </div>
      </div>
      
      <div className="trajectory-canvas">
        <div className="canvas-grid" />
        <div className="axis-label x">1280 x 720</div>
        <div className="axis-label y">Air Canvas</div>

        {hasStrokes && (
          <svg
            viewBox="0 0 1280 720"
            preserveAspectRatio="xMidYMid meet"
            className="trajectory-svg"
            aria-label="Air writing trajectory"
          >
            {strokes.map((s, idx) => {
              if (!s || s.length === 0) return null;
              const isSinglePoint = s.length === 1;
              const pathData = renderSmoothStrokePath(s);

              return (
                <g key={idx}>
                  {isSinglePoint ? (
                    <>
                      <circle
                        cx={s[0].x}
                        cy={s[0].y}
                        r={12}
                        fill="#00e5ff"
                        opacity={0.35}
                      />
                      <circle
                        cx={s[0].x}
                        cy={s[0].y}
                        r={6}
                        fill="#ffffff"
                      />
                    </>
                  ) : (
                    <>
                      {/* Outer neon glow */}
                      <path
                        d={pathData}
                        style={{
                          stroke: '#00e5ff',
                          strokeWidth: 10,
                          strokeLinecap: 'round',
                          strokeLinejoin: 'round',
                          fill: 'none',
                          opacity: 0.35,
                          animation: 'none',
                          strokeDasharray: 'none',
                        }}
                      />
                      {/* Solid crisp core pen stroke */}
                      <path
                        d={pathData}
                        style={{
                          stroke: '#ffffff',
                          strokeWidth: 4.5,
                          strokeLinecap: 'round',
                          strokeLinejoin: 'round',
                          fill: 'none',
                          animation: 'none',
                          strokeDasharray: 'none',
                        }}
                      />
                    </>
                  )}
                </g>
              );
            })}

            {/* Real-time fingertip cursor indicator directly on canvas */}
            {activeTip && (
              <g transform={`translate(${activeTip.x}, ${activeTip.y})`}>
                <circle r={18} fill="none" stroke="#00e5ff" strokeWidth={2} opacity={0.7}>
                  <animate attributeName="r" values="14;24;14" dur="1.2s" repeatCount="indefinite" />
                  <animate attributeName="opacity" values="0.8;0.2;0.8" dur="1.2s" repeatCount="indefinite" />
                </circle>
                <circle r={5} fill="#00e5ff" />
                <line x1={-12} y1={0} x2={12} y2={0} stroke="#00e5ff" strokeWidth={1.5} />
                <line x1={0} y1={-12} x2={0} y2={12} stroke="#00e5ff" strokeWidth={1.5} />
              </g>
            )}
          </svg>
        )}

        {!hasStrokes && croppedImageBase64 && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '20px' }}>
            <img
              src={`data:image/png;base64,${croppedImageBase64}`}
              alt="Handwriting Canvas Capture"
              style={{ maxHeight: '120px', maxWidth: '80%', borderRadius: '8px', border: '1px solid rgba(0, 229, 255, 0.3)', boxShadow: '0 0 20px rgba(0, 229, 255, 0.15)', background: '#fff' }}
            />
          </div>
        )}

        <div className="canvas-empty-copy">
          <WandSparkles size={17} />
          <span>
            {recognizedWord
              ? `Recognized “${recognizedWord}”`
              : hasStrokes
              ? `Captured ${strokes.length} stroke${strokes.length > 1 ? 's' : ''} (${totalPoints} pts)`
              : 'Write letters or words in the air (Index finger)'}
          </span>
        </div>
      </div>

      <div className="canvas-footer">
        <div><span className="metric-label">STROKES</span><strong>{strokes.length}</strong></div>
        <div><span className="metric-label">POINTS</span><strong>{totalPoints}</strong></div>
        <div><span className="metric-label">SMOOTHING</span><strong>1€ Adaptive Filter</strong></div>
        <div><span className="metric-label">MODE</span><strong>Microsoft TrOCR</strong></div>
      </div>
    </section>
  );
});
