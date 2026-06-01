/// <reference lib="webworker" />

/*
 * gaze-detection.worker.ts — Web Worker for live gaze analysis (interview webcam stream).
 *
 * The main thread sends packed MediaPipe FaceMesh landmarks (iris + eyelids) per frame.
 * This worker:
 *   - Derives horizontal iris ratio (H) and vertical iris ratio (V), then classifies gaze
 *     into 9 states: center / center_up / left / left_up / left_down /
 *                    right / right_up / right_down / down / away
 *   - Applies 3-frame median smoothing on H/V ratios to suppress single-frame noise.
 *   - Single-eye fallback: uses the valid eye when one eye's geometry is degenerate.
 *   - Iris quality gating: frames with irisQuality < 0.3 are classified as 'away'.
 *   - Tracks per-frame stats and emits look-read cycle events (lateral or down → center).
 *   - On { type: 'stop' }, aggregates the full session, runs computeStats + computeVerdict
 *     (cheating/suspicious/clear scoring) and returns that to the main thread.
 *
 * Gaze taxonomy:
 *   center_up                          ← v < GLOBAL_UP (0.32), h in center
 *   left_up   center   right_up        ← v in neutral band, h lateral / center
 *   left_down  down    right_down      ← v > GLOBAL_DOWN_SOFT (0.62), h lateral / center
 *   away                               ← no usable iris data
 *
 * Messages in:  { type:'reset', config? } | { type:'frame', ..., irisQuality? } | { type:'stop', t, sampleFps }
 * Messages out: { type:'resetDone' } | { type:'frameResult', ... } | { type:'session', ... }
 */

/* eslint-disable no-undef */

// ── Config ──────────────────────────────────────────────────────────────────

const CFG = {
  CHEAT: {
    lat:    10.0,   // eye-level lateral transitions / min
    lrc:     4.0,   // look-read cycles / min
    mld:     3.0,   // max eye-level lateral run (s)
    away:   25.0,   // away-gaze %
    cad:     3.5,   // center avg duration (s) — lower = worse
    down:   18.0,   // down-gaze %
    dnd:     2.5,   // max down run (s)
    latLow:  1.5,   // NEW: lower-lateral (phone/notes) runs / min
    mldLow:  2.0,   // NEW: max lower-lateral run (s)
    latUp:  15.0,   // NEW: upper-lateral (thinking) runs / min — very high to avoid FP
  },
};

const DEFAULT_GLOBAL_THRESHOLDS = {
  left:     0.42,
  right:    0.58,
  down:     0.72,   // strong-down hard override
  up:       0.32,   // NEW: v < this → upper band
  downSoft: 0.62,   // NEW: v in (0.62, 0.72] → lower band
};

const DEFAULT_CHEAT_THRESHOLDS = {
  lat: 10.0, lrc: 4.0, mld: 3.0, away: 25.0, cad: 3.5, down: 18.0, dnd: 2.5,
  latLow: 1.5, mldLow: 2.0, latUp: 15.0,
};

let globalThresholds = { left: 0.42, right: 0.58, down: 0.72, up: 0.32, downSoft: 0.62 };
let activeWeights    = { lat: 3, lrc: 3, mld: 2, away: 2, cad: 1, down: 2, dnd: 1, latLow: 4, mldLow: 3, latUp: 1 };
let maxScore = 22;

function recomputeMaxScore(): void {
  maxScore =
    activeWeights.lat  + activeWeights.lrc    + activeWeights.mld  +
    activeWeights.away + activeWeights.cad    + activeWeights.down  + activeWeights.dnd +
    activeWeights.latLow + activeWeights.mldLow + activeWeights.latUp;
}

function pickFiniteNum(v: any, fallback: number): number {
  return typeof v === 'number' && isFinite(v) ? v : fallback;
}

function pickFirst(candidates: any[], fallback: number): number {
  for (const v of candidates) {
    if (typeof v === 'number' && isFinite(v)) return v;
  }
  return fallback;
}

function applyWorkerConfig(config: any): void {
  if (!config) {
    globalThresholds = { ...DEFAULT_GLOBAL_THRESHOLDS };
    Object.assign(CFG.CHEAT, DEFAULT_CHEAT_THRESHOLDS);
    activeWeights = { lat: 3, lrc: 3, mld: 2, away: 2, cad: 1, down: 2, dnd: 1, latLow: 4, mldLow: 3, latUp: 1 };
    recomputeMaxScore();
    return;
  }

  if (config.thresholds != null && typeof config.thresholds === 'object') {
    const th = config.thresholds;
    globalThresholds.left     = pickFiniteNum(th.left,     DEFAULT_GLOBAL_THRESHOLDS.left);
    globalThresholds.right    = pickFiniteNum(th.right,    DEFAULT_GLOBAL_THRESHOLDS.right);
    globalThresholds.down     = pickFiniteNum(th.down,     DEFAULT_GLOBAL_THRESHOLDS.down);
    globalThresholds.up       = pickFiniteNum(th.up,       pickFiniteNum(th.upBand,       DEFAULT_GLOBAL_THRESHOLDS.up));
    globalThresholds.downSoft = pickFiniteNum(th.downSoft, pickFiniteNum(th.downSoftBand, DEFAULT_GLOBAL_THRESHOLDS.downSoft));

    CFG.CHEAT.lat    = pickFirst([th.lat,    th.lateralGazeFrequencyPerMin, th.lateralGazeFrequency],      DEFAULT_CHEAT_THRESHOLDS.lat);
    CFG.CHEAT.lrc    = pickFirst([th.lrc,    th.lookReadCyclesPerMin,       th.lookReadCycles],             DEFAULT_CHEAT_THRESHOLDS.lrc);
    CFG.CHEAT.mld    = pickFirst([th.mld,    th.maxLateralDurationSec,      th.maxLateralDuration],         DEFAULT_CHEAT_THRESHOLDS.mld);
    CFG.CHEAT.away   = pickFirst([th.away,   th.awayGazePercentage],                                        DEFAULT_CHEAT_THRESHOLDS.away);
    CFG.CHEAT.cad    = pickFirst([th.cad,    th.centerAttentionDurationSec, th.centerAttentionDuration],    DEFAULT_CHEAT_THRESHOLDS.cad);
    // NOTE: th.down is reserved for iris V-ratio classification (e.g. 0.72), not the verdict threshold.
    CFG.CHEAT.down   = pickFirst([th.downGazePercentage, th.downGazePct,   th.down_pct],                   DEFAULT_CHEAT_THRESHOLDS.down);
    CFG.CHEAT.dnd    = pickFirst([th.dnd,    th.maxDownDurationSec,         th.maxDownDuration],            DEFAULT_CHEAT_THRESHOLDS.dnd);
    CFG.CHEAT.latLow = pickFirst([th.latLow, th.lowerLateralFrequency],                                     DEFAULT_CHEAT_THRESHOLDS.latLow);
    CFG.CHEAT.mldLow = pickFirst([th.mldLow, th.maxLowerLateralDuration],                                   DEFAULT_CHEAT_THRESHOLDS.mldLow);
    CFG.CHEAT.latUp  = pickFirst([th.latUp,  th.upperLateralFrequency],                                     DEFAULT_CHEAT_THRESHOLDS.latUp);
  } else {
    globalThresholds = { ...DEFAULT_GLOBAL_THRESHOLDS };
    Object.assign(CFG.CHEAT, DEFAULT_CHEAT_THRESHOLDS);
  }

  if (config.weightage != null && typeof config.weightage === 'object') {
    const wg = config.weightage;
    activeWeights.lat    = pickFiniteNum(wg.lateralGazeFrequency,    3);
    activeWeights.lrc    = pickFiniteNum(wg.lookReadCycles,          3);
    activeWeights.mld    = pickFiniteNum(wg.maxLateralDuration,      2);
    activeWeights.away   = pickFiniteNum(wg.awayGazePercentage,      2);
    activeWeights.cad    = pickFiniteNum(wg.centerAttentionDuration,  1);
    activeWeights.down   = pickFiniteNum(wg.downGazePercentage,      2);
    activeWeights.dnd    = pickFiniteNum(wg.maxDownDuration,         1);
    activeWeights.latLow = pickFiniteNum(wg.lowerLateralFrequency,   4);
    activeWeights.mldLow = pickFiniteNum(wg.maxLowerLateralDuration, 3);
    activeWeights.latUp  = pickFiniteNum(wg.upperLateralFrequency,   1);
  } else {
    activeWeights = { lat: 3, lrc: 3, mld: 2, away: 2, cad: 1, down: 2, dnd: 1, latLow: 4, mldLow: 3, latUp: 1 };
  }

  recomputeMaxScore();
}

applyWorkerConfig(null);

// ── Landmark slot indices (must match service WORKER_LANDMARK_IDXS) ─────────
// Slots: leftOuter(33), leftInner(133), rightInner(362), rightOuter(263),
//        leftIris(468), rightIris(473), lUp(159), lDn(145), rUp(386), rDn(374)
const LM_LEFT_OUTER  = 0;
const LM_LEFT_INNER  = 1;
const LM_RIGHT_INNER = 2;
const LM_RIGHT_OUTER = 3;
const LM_LEFT_IRIS   = 4;
const LM_RIGHT_IRIS  = 5;
const LM_L_UP        = 6;
const LM_L_DN        = 7;
const LM_R_UP        = 8;
const LM_R_DN        = 9;

// ── Iris ratio computation ───────────────────────────────────────────────────

function getXY(buf: Float32Array, slot: number): { x: number; y: number } {
  const o = slot * 2;
  return { x: buf[o], y: buf[o + 1] };
}

// Single-eye fallback: if one eye's geometry is too small, returns the other eye's ratio.
function computeHRatio(buf: Float32Array): number | null {
  if (!buf || buf.length < 20) return null;
  const lo    = getXY(buf, LM_LEFT_OUTER).x;
  const li    = getXY(buf, LM_LEFT_INNER).x;
  const ro    = getXY(buf, LM_RIGHT_INNER).x;
  const ri    = getXY(buf, LM_RIGHT_OUTER).x;
  const lIris = getXY(buf, LM_LEFT_IRIS).x;
  const rIris = getXY(buf, LM_RIGHT_IRIS).x;
  const lw = li - lo;
  const rw = ri - ro;

  const lOk = lw >= 0.005;
  const rOk = rw >= 0.005;
  if (!lOk && !rOk) return null;

  const lR = lOk ? Math.max(0, Math.min(1, (lIris - lo) / lw)) : null;
  const rR = rOk ? Math.max(0, Math.min(1, (rIris - ro) / rw)) : null;

  const ratio = lR !== null && rR !== null ? (lR + rR) / 2 : (lR ?? rR!);
  return Math.round(ratio * 10000) / 10000;
}

// Single-eye fallback: if one eye's geometry is too small, returns the other eye's ratio.
function computeVRatio(buf: Float32Array): number | null {
  if (!buf || buf.length < 20) return null;
  const lUp = getXY(buf, LM_L_UP).y;
  const lDn = getXY(buf, LM_L_DN).y;
  const lI  = getXY(buf, LM_LEFT_IRIS).y;
  const rUp = getXY(buf, LM_R_UP).y;
  const rDn = getXY(buf, LM_R_DN).y;
  const rI  = getXY(buf, LM_RIGHT_IRIS).y;
  const lH  = lDn - lUp;
  const rH  = rDn - rUp;

  const lOk = lH >= 0.003;
  const rOk = rH >= 0.003;
  if (!lOk && !rOk) return null;

  const lV = lOk ? Math.max(0, Math.min(1, (lI - lUp) / lH)) : null;
  const rV = rOk ? Math.max(0, Math.min(1, (rI - rUp) / rH)) : null;

  const ratio = lV !== null && rV !== null ? (lV + rV) / 2 : (lV ?? rV!);
  return Math.round(ratio * 10000) / 10000;
}

// ── 3-frame median smoothing ─────────────────────────────────────────────────

const SMOOTH_WIN = 3;
let hBuf: (number | null)[] = [];
let vBuf: (number | null)[] = [];

function pushSmooth(buf: (number | null)[], val: number | null): void {
  buf.push(val);
  if (buf.length > SMOOTH_WIN) buf.shift();
}

function medianOf(buf: (number | null)[]): number | null {
  const vals = buf.filter((x): x is number => x !== null);
  if (!vals.length) return null;
  const sorted = [...vals].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

// ── Gaze state helpers ───────────────────────────────────────────────────────

const LATERAL_STATES = new Set(['left', 'left_up', 'left_down', 'right', 'right_up', 'right_down']);
const CENTER_STATES  = new Set(['center', 'center_up']);

function isLateralGaze(g: string): boolean { return LATERAL_STATES.has(g); }
function isCenterLike(g: string):  boolean { return CENTER_STATES.has(g);  }

// ── 9-state gaze classification ──────────────────────────────────────────────

function classifyGaze(
  h: number | null,
  v: number | null,
  lT: number,
  rT: number,
  upT: number,
  downSoftT: number,
  dT: number,
): string {
  if (h === null && v === null) return 'away';

  const isLateral = h !== null && (h < lT || h > rT);

  // Strong-down override only when NOT strongly lateral — preserves left_down / right_down
  if (v !== null && v >= dT && !isLateral) return 'down';

  if (h === null) return 'away';

  const lateralDir: string | null = h < lT ? 'left' : h > rT ? 'right' : null;

  const vBand = v === null      ? 'mid'
              : v < upT         ? 'up'
              : v >= downSoftT  ? 'down'
              : 'mid';

  if (lateralDir === null) {
    if (vBand === 'up')   return 'center_up';
    if (vBand === 'down') return 'down';
    return 'center';
  }

  if (vBand === 'up')   return lateralDir + '_up';
  if (vBand === 'down') return lateralDir + '_down';
  return lateralDir;
}

// ── Session stats ────────────────────────────────────────────────────────────

function computeStats(frames: any[], duration: number, irisVisible: number): any {
  const face = frames.filter((f) => f.face_detected);

  if (!face.length) {
    return {
      error: 'No face detected',
      gaze_pct: {},
      lateral_per_min: 0,
      look_read_per_min: 0,
      lateral_max_dur: 0,
      center_avg_dur: 0,
      face_detection_rate: 0,
      iris_visible_frames: 0,
      gaze_visible_rate: 0,
    };
  }

  const rawCnt: Record<string, number> = {};
  for (const f of face) rawCnt[f.gaze] = (rawCnt[f.gaze] || 0) + 1;
  const tf = face.length;

  const runs: any[] = [];
  let rs = face[0].time;
  let rg = face[0].gaze;
  for (let i = 1; i < face.length; i++) {
    if (face[i].gaze !== rg) {
      runs.push({ gaze: rg, start: rs, end: face[i].time, dur: +(face[i].time - rs).toFixed(2) });
      rs = face[i].time;
      rg = face[i].gaze;
    }
  }
  runs.push({ gaze: rg, start: rs, end: face[face.length - 1].time, dur: +(face[face.length - 1].time - rs).toFixed(2) });

  let latAllT = 0;
  for (let i = 1; i < face.length; i++) {
    const p = face[i - 1].gaze;
    const c = face[i].gaze;
    if (c !== p && (isLateralGaze(c) || isLateralGaze(p))) latAllT++;
  }

  const lrE: any[] = [];
  for (let i = 0; i < runs.length - 1; i++) {
    const g0 = runs[i].gaze;
    if (!isCenterLike(runs[i + 1].gaze)) continue;
    if (isLateralGaze(g0)) {
      const kind = g0.endsWith('_down') ? 'down_lateral' : g0.endsWith('_up') ? 'up_lateral' : 'lateral';
      lrE.push({ time: +runs[i].start.toFixed(1), dir: g0, away_dur: runs[i].dur, kind });
    } else if (g0 === 'down') {
      lrE.push({ time: +runs[i].start.toFixed(1), dir: 'down', away_dur: runs[i].dur, kind: 'down' });
    }
  }

  const cRuns    = runs.filter((r) => isCenterLike(r.gaze));
  const dRuns    = runs.filter((r) => r.gaze === 'down');
  const lAllRuns = runs.filter((r) => isLateralGaze(r.gaze));
  const lEyeRuns = runs.filter((r) => r.gaze === 'left' || r.gaze === 'right');
  const lUpRuns  = runs.filter((r) => r.gaze === 'left_up'   || r.gaze === 'right_up');
  const lDnRuns  = runs.filter((r) => r.gaze === 'left_down' || r.gaze === 'right_down');

  const dm     = Math.max(duration / 60, 0.01);
  const maxDur = (arr: any[]) => arr.length ? +Math.max(...arr.map((r) => r.dur)).toFixed(2) : 0;
  const avgDur = (arr: any[]) => arr.length ? +(arr.reduce((s, r) => s + r.dur, 0) / arr.length).toFixed(2) : 0;

  const cntCenter = (rawCnt['center'] || 0) + (rawCnt['center_up'] || 0);
  const cntLeft   = (rawCnt['left']   || 0) + (rawCnt['left_up']   || 0) + (rawCnt['left_down']  || 0);
  const cntRight  = (rawCnt['right']  || 0) + (rawCnt['right_up']  || 0) + (rawCnt['right_down'] || 0);
  const cntDown   = rawCnt['down'] || 0;
  const cntAway   = rawCnt['away'] || 0;

  return {
    total_frames:         frames.length,
    face_detected_frames: tf,
    face_detection_rate:  +(tf / frames.length).toFixed(3),
    iris_visible_frames:  irisVisible,
    gaze_visible_rate:    +(irisVisible / frames.length).toFixed(3),
    gaze_pct: {
      center: +(cntCenter / tf * 100).toFixed(1),
      left:   +(cntLeft   / tf * 100).toFixed(1),
      right:  +(cntRight  / tf * 100).toFixed(1),
      down:   +(cntDown   / tf * 100).toFixed(1),
      away:   +(cntAway   / tf * 100).toFixed(1),
      center_up:  +((rawCnt['center_up']  || 0) / tf * 100).toFixed(1),
      left_up:    +((rawCnt['left_up']    || 0) / tf * 100).toFixed(1),
      left_down:  +((rawCnt['left_down']  || 0) / tf * 100).toFixed(1),
      right_up:   +((rawCnt['right_up']   || 0) / tf * 100).toFixed(1),
      right_down: +((rawCnt['right_down'] || 0) / tf * 100).toFixed(1),
    },
    lateral_per_min:   +(latAllT / dm).toFixed(2),
    look_read_per_min: +(lrE.length / dm).toFixed(2),
    lateral_max_dur:   maxDur(lAllRuns),
    down_max_dur:      maxDur(dRuns),
    lateral_avg_dur:   avgDur(lAllRuns),
    center_avg_dur:    avgDur(cRuns),
    lateral_transitions: latAllT,
    look_read_cycles:    lrE.length,
    runs:     runs.slice(0, 500),
    lrEvents: lrE.slice(0, 100),
    lateral_pct: {
      eye_level: +(((rawCnt['left']      || 0) + (rawCnt['right']      || 0)) / tf * 100).toFixed(1),
      upper:     +(((rawCnt['left_up']   || 0) + (rawCnt['right_up']   || 0)) / tf * 100).toFixed(1),
      lower:     +(((rawCnt['left_down'] || 0) + (rawCnt['right_down'] || 0)) / tf * 100).toFixed(1),
    },
    lateral_eye_per_min:  +(lEyeRuns.length / dm).toFixed(2),
    lateral_up_per_min:   +(lUpRuns.length  / dm).toFixed(2),
    lateral_down_per_min: +(lDnRuns.length  / dm).toFixed(2),
    lateral_eye_max_dur:  maxDur(lEyeRuns),
    lateral_up_max_dur:   maxDur(lUpRuns),
    lateral_down_max_dur: maxDur(lDnRuns),
    look_read_down_per_min: +(lrE.filter((e) => e.kind === 'down' || e.kind === 'down_lateral').length / dm).toFixed(2),
    look_read_up_per_min:   +(lrE.filter((e) => e.kind === 'up_lateral').length / dm).toFixed(2),
  };
}

// ── Verdict scoring ──────────────────────────────────────────────────────────

function computeVerdict(stats: any): any {
  if (stats.error) {
    return { label: 'UNKNOWN', confidence: 0, color: 'orange', reason: stats.error, factors: [], calibrated: false };
  }

  const {
    lat: tLat, lrc: tLrc, mld: tMld, away: tAway, cad: tCad, down: tDown, dnd: tDnd,
    latLow: tLatLow, mldLow: tMldLow, latUp: tLatUp,
  } = CFG.CHEAT;
  const note = ' (global)';
  let score = 0;
  const factors: any[] = [];

  const addF = (name: string, val: any, thresh: string, trig: boolean, w: number, sev: string) => {
    if (trig) score += w;
    factors.push({ name, value: val, threshold: thresh + note, triggered: trig, severity: trig ? sev : 'none' });
  };

  const aw  = stats.gaze_pct.away || 0;
  const fdr = stats.face_detection_rate || 0;
  const usablePct = Math.max(0, 100 - aw);
  const qualityOK = fdr >= 0.75 && aw <= 55;

  const lat    = stats.lateral_eye_per_min  ?? stats.lateral_per_min ?? 0;
  const mld    = stats.lateral_eye_max_dur  ?? stats.lateral_max_dur ?? 0;
  const latLow = stats.lateral_down_per_min ?? 0;
  const mldLow = stats.lateral_down_max_dur ?? 0;
  const latUp  = stats.lateral_up_per_min   ?? 0;
  const lrc    = stats.look_read_per_min    || 0;
  const cad    = stats.center_avg_dur       || 0;
  const dn     = stats.gaze_pct.down        || 0;
  const dnd    = stats.down_max_dur         || 0;

  const trigLat    = qualityOK && lat    >= tLat;
  const trigLrc    = lrc >= tLrc;
  const trigMld    = qualityOK && mld    >= tMld;
  const trigLatLow = qualityOK && latLow >= tLatLow;
  const trigMldLow = qualityOK && mldLow >= tMldLow;
  const trigLatUp  = qualityOK && latUp  >= tLatUp;
  const hasReadingEvidence =
    trigLrc || (qualityOK && dn >= tDown) || trigLatLow || trigMldLow ||
    (qualityOK && mld >= Math.max(tMld, 6) && lat >= tLat * 1.1);
  const trigAway = aw >= tAway && hasReadingEvidence && aw <= 85;
  const trigCad  = qualityOK && cad > 0 && cad < tCad;
  const trigDown = qualityOK && dn  >= tDown;
  const trigDnd  = qualityOK && dnd >= tDnd;

  addF('Lateral Gaze Frequency',          lat    + '/min',  '>' + tLat    + '/min',  trigLat,    activeWeights.lat,    lat    >= tLat    * 1.4 ? 'high' : 'medium');
  addF('Look-Read Cycles',                lrc    + '/min',  '>' + tLrc    + '/min',  trigLrc,    activeWeights.lrc,    lrc    >= tLrc    * 1.4 ? 'high' : 'medium');
  addF('Sustained Lateral Gaze',          mld    + 's',     '>' + tMld    + 's',     trigMld,    activeWeights.mld,    mld    >= 8            ? 'high' : 'medium');
  addF('Away Gaze / Head Turning',        aw     + '%',     '>' + tAway   + '%',     trigAway,   activeWeights.away,   aw     >= tAway   * 1.3 ? 'high' : 'medium');
  addF('Short Attention Span',            cad    + 's avg', '<' + tCad    + 's avg', trigCad,    activeWeights.cad,    'medium');
  addF('Down Gaze (Reading Below)',       dn     + '%',     '>' + tDown   + '%',     trigDown,   activeWeights.down,   dn     >= tDown   * 1.3 ? 'high' : 'medium');
  addF('Sustained Down Gaze',             dnd    + 's',     '>' + tDnd    + 's',     trigDnd,    activeWeights.dnd,    dnd    >= 6            ? 'high' : 'medium');
  addF('Lower Lateral Gaze (Phone/Notes)',latLow + '/min',  '>' + tLatLow + '/min',  trigLatLow, activeWeights.latLow, latLow >= tLatLow * 2  ? 'high' : 'medium');
  addF('Sustained Lower Lateral Gaze',   mldLow + 's',     '>' + tMldLow + 's',     trigMldLow, activeWeights.mldLow, mldLow >= 4            ? 'high' : 'medium');
  addF('Upper Lateral Gaze (Thinking)',   latUp  + '/min',  '>' + tLatUp  + '/min',  trigLatUp,  activeWeights.latUp,  'low');

  const qualityPenalty = qualityOK ? 0 : Math.min(22, Math.round(Math.max(0, aw - 55) * 0.25 + (0.75 - Math.min(0.75, fdr)) * 30));
  const cRaw  = maxScore > 0 ? (score / maxScore) * 100 : 0;
  const nTrig = factors.filter((f) => f.triggered).length;

  const primaryEvidence =
    trigLrc || trigDown || trigDnd || trigLatLow || trigMldLow ||
    (qualityOK && mld >= Math.max(tMld, 6) && lat >= tLat * 1.2);

  let label: string, color: string, confidence: number;
  if (!qualityOK && !primaryEvidence) {
    label = 'CAN NOT EVALUATE'; color = 'orange'; confidence = 0;
  } else if (score >= 7 && primaryEvidence) {
    label = 'CHEATING DETECTED'; color = 'red';
    confidence = Math.min(98, Math.max(55, Math.round(60 + cRaw * 0.4 - qualityPenalty)));
  } else if (score >= 4 || (score >= 7 && !primaryEvidence)) {
    label = 'SUSPICIOUS BEHAVIOR'; color = 'orange';
    confidence = Math.max(30, Math.round(40 + cRaw * 0.35 - qualityPenalty));
  } else {
    label = 'NO CHEATING DETECTED'; color = 'green';
    confidence = Math.min(97, Math.max(55, Math.round(68 + (1 - cRaw / 100) * 29 - qualityPenalty)));
  }

  const nonCheatingConfidence = confidence;
  const cheatingConfidence =
    label === 'NO CHEATING DETECTED' || label === 'CAN NOT EVALUATE'
      ? Math.max(0, Math.min(100, 100 - nonCheatingConfidence))
      : nonCheatingConfidence;
  confidence = cheatingConfidence;

  const tNames = factors.filter((f) => f.triggered).map((f) => f.name);

  return {
    label, color, confidence,
    score, max_score: maxScore, triggered_count: nTrig,
    reason:
      nTrig + '/' + factors.length + ' indicators triggered' +
      (tNames.length ? ': ' + tNames.join(', ') : ' — none') +
      (qualityOK ? '' : ' · Tracking quality low (face ' + Math.round(fdr * 100) + '%, away ' + aw + '%, usable ' + usablePct + '%)'),
    factors,
    calibrated: false,
    thresholds: {
      lateral_per_min:       tLat,
      look_read_per_min:     tLrc,
      lateral_max_dur:       tMld,
      away_pct:              tAway,
      center_avg_dur:        tCad,
      down_pct:              tDown,
      down_max_dur:          tDnd,
      lower_lateral_per_min: tLatLow,
      lower_lateral_max_dur: tMldLow,
      upper_lateral_per_min: tLatUp,
    },
    confidence_mode:           'cheating',
    confidence_non_cheating:   nonCheatingConfidence,
  };
}

// ── Session state ────────────────────────────────────────────────────────────

let frames: any[]      = [];
let frameIndex         = 0;
let leftT              = globalThresholds.left;
let rightT             = globalThresholds.right;
let downT              = globalThresholds.down;
let upT                = globalThresholds.up;
let downSoftT          = globalThresholds.downSoft;

let gazeCounts = {
  center: 0, center_up: 0,
  left: 0, left_up: 0, left_down: 0,
  right: 0, right_up: 0, right_down: 0,
  down: 0, away: 0, no_face: 0,
};

let prevFaceGaze: string | null = null;
let lateralRunStart: number | null = null;
let irisVisibleFrames = 0;

function resetAll(): void {
  frames            = [];
  frameIndex        = 0;
  leftT             = globalThresholds.left;
  rightT            = globalThresholds.right;
  downT             = globalThresholds.down;
  upT               = globalThresholds.up;
  downSoftT         = globalThresholds.downSoft;
  gazeCounts        = { center: 0, center_up: 0, left: 0, left_up: 0, left_down: 0, right: 0, right_up: 0, right_down: 0, down: 0, away: 0, no_face: 0 };
  prevFaceGaze      = null;
  lateralRunStart   = null;
  irisVisibleFrames = 0;
  hBuf              = [];
  vBuf              = [];
}

function updateGazeCounts(g: string): void {
  if (g === 'no_face') { gazeCounts.no_face++; return; }
  if (g in gazeCounts) (gazeCounts as any)[g]++;
}

function buildStatsSnapshot(): any {
  return {
    total:  frameIndex,
    center: gazeCounts.center + gazeCounts.center_up,
    left:   gazeCounts.left   + gazeCounts.left_up   + gazeCounts.left_down,
    right:  gazeCounts.right  + gazeCounts.right_up  + gazeCounts.right_down,
    down:   gazeCounts.down,
    missed: gazeCounts.no_face,
    away:   gazeCounts.away,
  };
}

function processLookReadTransitions(gaze: string, t: number, faceDetected: boolean): any[] {
  const newEvents: any[] = [];

  if (!faceDetected) {
    prevFaceGaze    = null;
    lateralRunStart = null;
    return newEvents;
  }

  const isNonCenter = !isCenterLike(gaze) && gaze !== 'away';
  const prevIsNonCenter =
    prevFaceGaze !== null && !isCenterLike(prevFaceGaze) && prevFaceGaze !== 'away';

  if (isNonCenter) {
    if (!prevIsNonCenter || gaze !== prevFaceGaze) {
      lateralRunStart = t;
    }
  }

  if (isCenterLike(gaze) && prevIsNonCenter && prevFaceGaze !== null) {
    const awayDur = lateralRunStart !== null ? +(t - lateralRunStart).toFixed(2) : 0;
    const isDownKind = prevFaceGaze === 'down' || prevFaceGaze === 'left_down' || prevFaceGaze === 'right_down';
    const isUpKind   = prevFaceGaze === 'left_up' || prevFaceGaze === 'right_up';
    const kind = isDownKind ? 'down' : isUpKind ? 'up_lateral' : 'lateral';
    newEvents.push({
      type:     'lookRead',
      time:     lateralRunStart !== null ? +lateralRunStart.toFixed(1) : +t.toFixed(1),
      dir:      prevFaceGaze,
      away_dur: awayDur,
      kind,
      endTime:  t,
    });
  }

  prevFaceGaze = gaze;
  return newEvents;
}

// ── Message handler ──────────────────────────────────────────────────────────

addEventListener('message', (e: MessageEvent) => {
  const msg = e.data || {};

  if (msg.type === 'reset') {
    applyWorkerConfig(msg.config);
    resetAll();
    postMessage({ type: 'resetDone' });
    return;
  }

  if (msg.type === 'seed') {
    // Restore prior session frames saved to sessionStorage before a page reload.
    // Each entry carries n (frame count) so count-based stats (away_pct, down_pct,
    // face_detection_rate) are exact even though only change events were stored.
    const seedFrames: any[] = Array.isArray(msg.frames) ? msg.frames : [];
    for (const f of seedFrames) {
      const gaze  = typeof f.g === 'string' ? f.g : 'away';
      const count = typeof f.n === 'number' && f.n > 0 ? f.n : 1;
      for (let i = 0; i < count; i++) {
        frames.push({
          time:          +Number(f.t).toFixed(2),
          frame:         frameIndex,
          face_detected: !!f.fd,
          gaze,
          h_ratio:       null,
          v_ratio:       null,
        });
        frameIndex++;
        updateGazeCounts(gaze);
      }
    }
    if (typeof msg.irisVisibleFrames === 'number') {
      irisVisibleFrames = msg.irisVisibleFrames;
    }

    // Restore prevFaceGaze from the last face-detected seeded frame so the
    // first look-read transition after a page reload is not silently dropped.
    for (let i = frames.length - 1; i >= 0; i--) {
      if (frames[i].face_detected) {
        prevFaceGaze = frames[i].gaze;
        break;
      }
    }
    return;
  }

  if (msg.type === 'stop') {
    const t: number = Number(msg.t) || 0;
    let duration = t;
    if (frames.length && frames[frames.length - 1].time > duration) {
      duration = frames[frames.length - 1].time;
    }
    if (duration <= 0 && frames.length) {
      duration = frames[frames.length - 1].time;
    }

    const stats   = computeStats(frames, duration, irisVisibleFrames);
    const verdict = computeVerdict(stats);

    postMessage({
      type:               'session',
      stats:              buildStatsSnapshot(),
      gazeEvents:         [],
      activeEvent:        null,
      sessionDurationSec: duration,
      pocStats:           stats,
      verdict,
      finalThresholds:    { left: leftT, right: rightT, up: upT, downSoft: downSoftT, down: downT },
    });
    return;
  }

  if (msg.type !== 'frame') return;

  const t: number       = Number(msg.t) || 0;
  let hRatio: number | null = null;
  let vRatio: number | null = null;
  const faceDetected    = !!(msg.hasFace && msg.pts);
  let gaze              = 'no_face';

  if (faceDetected) {
    const irisQuality: number = typeof msg.irisQuality === 'number' ? msg.irisQuality : 1;
    const qualityOk = irisQuality >= 0.3;

    if (qualityOk) irisVisibleFrames++;

    const rawH = qualityOk ? computeHRatio(msg.pts as Float32Array) : null;
    const rawV = qualityOk ? computeVRatio(msg.pts as Float32Array) : null;

    pushSmooth(hBuf, rawH);
    pushSmooth(vBuf, rawV);
    hRatio = medianOf(hBuf);
    vRatio = medianOf(vBuf);

    gaze = classifyGaze(hRatio, vRatio, leftT, rightT, upT, downSoftT, downT);
  }

  frames.push({
    time:          +t.toFixed(2),
    frame:         frameIndex,
    face_detected: faceDetected,
    gaze,
    h_ratio:       hRatio,
    v_ratio:       vRatio,
  });
  frameIndex++;

  updateGazeCounts(gaze);

  const lrEvents = processLookReadTransitions(gaze, t, faceDetected);

  postMessage({
    type:        'frameResult',
    gaze,
    stats:       buildStatsSnapshot(),
    activeEvent: null,
    newEvents:   lrEvents,
    calibrated:  false,
  });
});

export {};
