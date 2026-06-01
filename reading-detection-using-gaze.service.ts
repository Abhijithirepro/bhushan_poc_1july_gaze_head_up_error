import { Injectable } from '@angular/core';
import { Observable, Subject } from 'rxjs';

// ── Public interfaces ─────────────────────────────────────────────────────────

export interface ReadingDetectionConfig {
  isEnabled: boolean;
  /** Frames per second sent to MediaPipe (2–4 is sufficient). Default: 3 */
  targetFPS: number;
  /** Worker sends a rolling report every N seconds. Default: 60 */
  rollingReportEverySec: number;
  /** Include raw report object in periodic log payload. Default: false */
  sendAllGazeDebugLogs: boolean;
  /** Push cumulative report to periodicGazeAnalysisResult$ every N minutes (0 = disabled). Default: 5 */
  sendGazeAnalysisEveryNMinutes: number;
  /** Passed through to consumer; the service itself does not act on this flag. */
  stopGazeDetectionOnInterviewerLeave?: boolean;
  /** Override path to @mediapipe/tasks-vision wasm bundle. */
  mediapipeWasmPath?: string;
  /** Override path to face_landmarker.task model file. */
  mediapipeModelPath?: string;
}

export interface RdgaFlag {
  severity: string;
  text: string;
}

export interface RdgaBreakdownItem {
  rule: string;
  detail: string;
  points: number;
}

export interface RdgaMatchedProfile {
  code: string;
  label: string;
}

export interface RdgaReport {
  riskScore: number;
  verdict: string;
  verdictBand: 'NORMAL' | 'SOFT' | 'WARN' | 'CRITICAL' | 'UNKNOWN';
  matchedProfile: RdgaMatchedProfile;
  durationSec: number;
  framesValid: number;
  framesTotal: number;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  stats: Record<string, any>;
  breakdown: RdgaBreakdownItem[];
  flags: RdgaFlag[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  raw: { sampleCount: number; sampleFps: number; rawSamples?: any[] };
  sampleFps?: number;
  generatedAt?: string;
}

export interface PeriodicGazeAnalysisPayload {
  analysisResult: RdgaReport;
  isFinal: boolean;
}

// ── Declare MediaPipe globals (loaded via <script> tag in index.html) ─────────
// eslint-disable-next-line @typescript-eslint/no-explicit-any
declare const window: any;

// ── Landmark indices for nose-based yaw/pitch estimation ─────────────────────
const LMIDX = {
  NOSE_TIP:        1,
  LEFT_EYE_OUTER:  33,
  RIGHT_EYE_OUTER: 263,
  CHIN:            152,
  FOREHEAD:        10,
};

@Injectable({ providedIn: 'root' })
export class ReadingDetectionUsingGaze {
  private readonly destroy$ = new Subject<void>();

  /** Emits every rolling report received from the worker. */
  private readonly gazeReport$ = new Subject<RdgaReport>();
  public readonly gazeReports$: Observable<RdgaReport> = this.gazeReport$.asObservable();

  /** Emits every sendGazeAnalysisEveryNMinutes (and on final stop). Subscribe and push to logService. */
  private readonly periodicAnalysisPayload$ = new Subject<PeriodicGazeAnalysisPayload>();
  public readonly periodicGazeAnalysisResult$: Observable<PeriodicGazeAnalysisPayload> =
    this.periodicAnalysisPayload$.asObservable();

  private config: ReadingDetectionConfig = {
    isEnabled: false,
    targetFPS: 3,
    rollingReportEverySec: 60,
    sendAllGazeDebugLogs: false,
    sendGazeAnalysisEveryNMinutes: 5,
    stopGazeDetectionOnInterviewerLeave: true,
  };

  private state = {
    started:              false,
    paused:               false,
    initializing:         false,
    stream:               null as MediaStream | null,
    video:                null as HTMLVideoElement | null,
    sessionStartTime:     0,
    frameCount:           0,
    frameProcessTimes:    [] as number[],
    activeSegmentStartMs: 0,   // wall-clock ms when current active segment began
    cumulativeActiveMs:   0,   // total ms spent in active (non-paused) detection
  };

  private captureCanvas = document.createElement('canvas');
  private captureCtx    = this.captureCanvas.getContext('2d', { willReadFrequently: false })!;

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private landmarkerPromise: Promise<any> | null = null;
  private faceModelReady = false;
  private gazeWorker: Worker | null = null;
  private lastReport: RdgaReport | null = null;
  private captureIntervalId: ReturnType<typeof setInterval> | null = null;
  private periodicAnalysisTimerId: ReturnType<typeof setTimeout> | undefined;

  private sampleBuffer: Array<{ t: number; yawD: number; pitchD: number; jawOpen: number; eyeLookDown?: number; eyeLookUp?: number; eyeLookOut?: number; eyeLookIn?: number; eyeBlink?: number }> = [];
  private checkpointIntervalId: ReturnType<typeof setInterval> | null = null;
  private readonly CHECKPOINT_KEY = 'rdga_checkpoint';
  private preserveCheckpointOnNextStop = false;
  private readonly CHECKPOINT_INTERVAL_MS = 5000;

  // ── Public API ────────────────────────────────────────────────────────────

  /**
   * Eagerly initialises the MediaPipe face landmarker so it is warm before start().
   * Safe to call multiple times — returns the same cached promise.
   */
  preload(): Promise<void> {
    return this.getFaceLandmarker()
      .then(() => { console.log('BB RDGA: face model ready.'); })
      .catch(e => { console.warn('BB RDGA: face model preload failed.', e); });
  }

  /**
   * Start gaze detection on the given MediaStream.
   * Returns an Observable that emits rolling RdgaReport objects from the worker.
   */
  start(stream: MediaStream, config: Partial<ReadingDetectionConfig>): Observable<RdgaReport> {
    this.config = { ...this.config, ...config };

    if (!this.config.isEnabled) return this.gazeReports$;
    if (this.state.started)     return this.gazeReports$;

    if (!this.faceModelReady) {
      this.getFaceLandmarker()
        .then(() => { if (!this.state.started) this.doStart(stream); })
        .catch(e => { console.warn('BB RDGA: cannot start — model failed to load.', e); });
    } else {
      this.doStart(stream);
    }

    return this.gazeReports$;
  }

  stop(preserveCheckpoint = false): void {
    if (!this.state.started) return;
    this.preserveCheckpointOnNextStop = preserveCheckpoint;
    // Accumulate the active segment that was running up to this stop
    if (!this.state.paused && this.state.activeSegmentStartMs > 0) {
      this.state.cumulativeActiveMs += Date.now() - this.state.activeSegmentStartMs;
      this.state.activeSegmentStartMs = 0;
    }
    if (preserveCheckpoint) {
      this.writeCheckpoint(); // flush latest samples before tearing down
    }
    console.log('BB RDGA: stop() called.');
    this.stopCaptureInterval();
    this.stopPeriodicTimer();
    if (this.gazeWorker) {
      this.gazeWorker.postMessage({ type: 'stop' });
    } else {
      if (!preserveCheckpoint) this.clearCheckpoint();
      this.cleanup(true);
    }
  }

  pause(): void {
    if (!this.state.started || this.state.paused) return;
    if (this.state.activeSegmentStartMs > 0) {
      this.state.cumulativeActiveMs += Date.now() - this.state.activeSegmentStartMs;
      this.state.activeSegmentStartMs = 0;
    }
    console.log('BB RDGA: paused.');
    this.state.paused = true;
    this.stopCaptureInterval();
    this.stopPeriodicTimer();
    this.stopCheckpointInterval();
  }

  resume(): void {
    if (!this.state.started || !this.state.paused) return;
    this.state.activeSegmentStartMs = Date.now();
    console.log('BB RDGA: resumed.');
    this.state.paused = false;
    this.startCaptureInterval();
    this.startPeriodicTimer();
    this.startCheckpointInterval();
  }

  /**
   * Returns total seconds during which reading detection was actively running
   * (excludes time paused while on coding/other tabs).
   */
  getActiveDetectionDurationSec(): number {
    let totalMs = this.state.cumulativeActiveMs;
    if (this.state.started && !this.state.paused && this.state.activeSegmentStartMs > 0) {
      totalMs += Date.now() - this.state.activeSegmentStartMs;
    }
    return totalMs / 1000;
  }

  reset(): void {
    this.stop();
    // landmarkerPromise and faceModelReady are intentionally preserved —
    // the pre-loaded landmarker is valid for the entire page lifetime.
  }

  // ── Internal: doStart ─────────────────────────────────────────────────────

  private doStart(stream: MediaStream): void {
    console.log('BB RDGA: starting.');
    this.state.started          = true;
    this.state.paused           = false;
    this.state.initializing     = true;
    this.state.stream           = stream;
    this.state.frameCount       = 0;
    this.state.frameProcessTimes = [];
    this.lastReport             = null;

    const checkpoint = this.loadCheckpoint();
    if (checkpoint && checkpoint.samples.length > 0) {
      this.state.sessionStartTime   = checkpoint.sessionStartTime;
      this.state.cumulativeActiveMs = checkpoint.cumulativeActiveMs ?? 0;
      this.sampleBuffer = checkpoint.samples.slice();
      console.log('BB RDGA: resuming from checkpoint — ' + checkpoint.samples.length + ' samples');
    } else {
      this.state.sessionStartTime   = Date.now();
      this.state.cumulativeActiveMs = 0;
      this.sampleBuffer = [];
    }
    this.state.activeSegmentStartMs = Date.now();

    this.captureCanvas.width  = 320;
    this.captureCanvas.height = 240;

    this.state.video    = this.ensureVideoElement(stream);
    this.initWorker();

    if (this.sampleBuffer.length > 0 && this.gazeWorker) {
      this.gazeWorker.postMessage({ type: 'replay', samples: this.sampleBuffer });
    }

    this.state.initializing = false;
    this.startCaptureInterval();
    this.startPeriodicTimer();
    this.startCheckpointInterval();
  }

  // ── Internal: video element ───────────────────────────────────────────────

  private ensureVideoElement(stream: MediaStream): HTMLVideoElement {
    const existing = document.getElementById('candidateStream') as HTMLVideoElement | null;
    if (existing && existing.tagName === 'VIDEO') {
      // Always sync srcObject — on a camera device change the element persists but
      // its srcObject still points to the old stopped stream, causing dead-track analysis.
      if (existing.srcObject !== stream) existing.srcObject = stream;
      return existing;
    }
    const v       = document.createElement('video');
    v.id          = 'rdga-hidden-video';
    v.muted       = true;
    v.autoplay    = true;
    v.playsInline = true;
    v.style.cssText = 'position:absolute;width:1px;height:1px;top:-9999px;left:-9999px;';
    v.srcObject   = stream;
    document.body.appendChild(v);
    v.play().catch(() => { /* autoplay may be blocked silently */ });
    return v;
  }

  private removeHiddenVideoIfCreated(): void {
    const hidden = document.getElementById('rdga-hidden-video');
    if (hidden && hidden.parentNode) {
      (hidden as HTMLVideoElement).srcObject = null;
      hidden.parentNode.removeChild(hidden);
    }
  }

  // ── Internal: Worker lifecycle ────────────────────────────────────────────

  private initWorker(): void {
    this.destroyWorker();
    try {
      this.gazeWorker = new Worker(
        new URL('../workers/reading-detection.worker', import.meta.url),
        { type: 'module' }
      );
      this.gazeWorker.onmessage = (e) => this.onWorkerMessage(e);
      this.gazeWorker.onerror   = (e) => { console.warn('BB RDGA: worker error.', e); };
      this.gazeWorker.postMessage({
        type: 'start',
        fps: this.config.targetFPS || 3,
        rollingReportEverySec: this.config.rollingReportEverySec || 60,
      });
    } catch (e) {
      console.warn('BB RDGA: failed to create worker.', e);
      this.gazeWorker = null;
    }
  }

  private destroyWorker(): void {
    if (this.gazeWorker) {
      try { this.gazeWorker.terminate(); } catch { /* ignore */ }
      this.gazeWorker = null;
    }
  }

  private onWorkerMessage(e: MessageEvent): void {
    const msg = e.data;
    if (!msg || !msg.type) return;

    if (msg.type === 'report') {
      this.lastReport = msg.report as RdgaReport;
      this.gazeReport$.next(this.lastReport);
      console.log('BB RDGA: rolling report — riskScore=' + this.lastReport.riskScore
        + ' verdict=' + this.lastReport.verdict);

    } else if (msg.type === 'final') {
      this.lastReport = (msg.report as RdgaReport) || null;
      if (this.lastReport) {
        console.log('BB RDGA: final report — riskScore=' + this.lastReport.riskScore
          + ' verdict=' + this.lastReport.verdict);
        this.periodicAnalysisPayload$.next({ analysisResult: this.lastReport, isFinal: true });
      } else {
        console.log('BB RDGA: session ended with no samples collected.');
      }
      if (!this.preserveCheckpointOnNextStop) {
        this.clearCheckpoint();
      }
      this.preserveCheckpointOnNextStop = false;
      this.cleanup(true);

    } else if (msg.type === 'error') {
      console.warn('BB RDGA: worker reported error: ' + msg.message);
      this.cleanup(true);
    }
  }

  // ── Internal: capture interval ────────────────────────────────────────────

  private startCaptureInterval(): void {
    this.stopCaptureInterval();
    const fps        = this.config.targetFPS || 3;
    const intervalMs = Math.round(1000 / fps);
    this.captureIntervalId = setInterval(() => this.captureAndProcess(), intervalMs);
  }

  private stopCaptureInterval(): void {
    if (this.captureIntervalId != null) {
      clearInterval(this.captureIntervalId);
      this.captureIntervalId = null;
    }
  }

  /**
   * Draws one frame onto the offscreen 320 px canvas, feeds it to MediaPipe,
   * extracts yaw/pitch/jaw, and sends the sample to the analysis worker.
   */
  private captureAndProcess(): void {
    if (!this.state.started || this.state.paused) return;
    const video = this.state.video;
    if (!video || video.readyState < 2) return;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return;

    const t      = (Date.now() - this.state.sessionStartTime) / 1000;
    const scale  = 320 / vw;
    const dw     = 320;
    const dh     = Math.round(vh * scale);

    if (this.captureCanvas.width  !== dw) this.captureCanvas.width  = dw;
    if (this.captureCanvas.height !== dh) this.captureCanvas.height = dh;

    this.captureCtx.drawImage(video, 0, 0, dw, dh);

    createImageBitmap(this.captureCanvas).then(bitmap => {
      return this.getFaceLandmarker().then(landmarker => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        let result: any;
        try {
          result = landmarker.detect(bitmap);
        } catch {
          result = { faceLandmarks: [], faceBlendshapes: [] };
        }
        bitmap.close();
        return result;
      }, (err) => { bitmap.close(); throw err; });
    }).then(result => {
      if (!this.state.started || this.state.paused) return;

      let yawDeg  = 0;
      let pitchDeg = 0;
      let jawOpen = 0;
      let face    = false;
      let eyeFields = { eyeLookDown: 0, eyeLookUp: 0, eyeLookOut: 0, eyeLookIn: 0, eyeBlink: 0 };

      if (result.faceLandmarks && result.faceLandmarks.length) {
        const lm = result.faceLandmarks[0];
        face     = true;
        const pose = this.estimateYawPitchNose(lm);
        yawDeg   = pose.yawDeg;
        pitchDeg = pose.pitchDeg;
        jawOpen  = this.jawOpenScore(result.faceBlendshapes);
        eyeFields = this.extractBlendshapes(result.faceBlendshapes);
      }

      if (this.gazeWorker) {
        this.gazeWorker.postMessage({ type: 'sample', sample: { t, yawDeg, pitchDeg, jawOpen, face, ...eyeFields } });
      }

      if (face) {
        this.sampleBuffer.push({
          t:           Math.round(t * 10) / 10,
          yawD:        Math.round(yawDeg * 10) / 10,
          pitchD:      Math.round(pitchDeg * 10) / 10,
          jawOpen:     Math.round(jawOpen * 1000) / 1000,
          eyeLookDown: Math.round(eyeFields.eyeLookDown * 1000) / 1000,
          eyeLookUp:   Math.round(eyeFields.eyeLookUp   * 1000) / 1000,
          eyeLookOut:  Math.round(eyeFields.eyeLookOut  * 1000) / 1000,
          eyeLookIn:   Math.round(eyeFields.eyeLookIn   * 1000) / 1000,
          eyeBlink:    Math.round(eyeFields.eyeBlink    * 1000) / 1000,
        });
      }

      this.state.frameCount++;
      this.state.frameProcessTimes.push(Date.now());
    }).catch(() => {
      // Silently swallow per-frame errors — a missed frame is not fatal
    });
  }

  // ── Internal: periodic timer ──────────────────────────────────────────────

  private startPeriodicTimer(): void {
    this.stopPeriodicTimer();
    const intervalMin = this.config.sendGazeAnalysisEveryNMinutes || 0;
    if (intervalMin <= 0) return;

    const intervalMs = intervalMin * 60 * 1000;
    const runAndReschedule = (): void => {
      if (!this.state.started) return;
      if (this.lastReport) {
        this.periodicAnalysisPayload$.next({ analysisResult: this.lastReport, isFinal: false });
      }
      this.periodicAnalysisTimerId = setTimeout(runAndReschedule, intervalMs);
    };
    this.periodicAnalysisTimerId = setTimeout(runAndReschedule, intervalMs);
  }

  private stopPeriodicTimer(): void {
    if (this.periodicAnalysisTimerId != null) {
      clearTimeout(this.periodicAnalysisTimerId);
      this.periodicAnalysisTimerId = undefined;
    }
  }

  // ── Internal: cleanup ─────────────────────────────────────────────────────

  private cleanup(clearData: boolean): void {
    this.stopCaptureInterval();
    this.stopPeriodicTimer();
    this.stopCheckpointInterval();
    this.destroyWorker();
    this.removeHiddenVideoIfCreated();

    this.state.started              = false;
    this.state.paused               = false;
    this.state.stream               = null;
    this.state.video                = null;
    this.state.frameCount           = 0;
    this.state.frameProcessTimes    = [];
    this.state.activeSegmentStartMs = 0;
    this.state.cumulativeActiveMs   = 0;

    if (clearData !== false) this.lastReport = null;
    console.log('BB RDGA: cleanup complete.');
  }

  // ── Internal: sessionStorage checkpoint ──────────────────────────────────

  private startCheckpointInterval(): void {
    this.stopCheckpointInterval();
    this.checkpointIntervalId = setInterval(() => this.writeCheckpoint(), this.CHECKPOINT_INTERVAL_MS);
  }

  private stopCheckpointInterval(): void {
    if (this.checkpointIntervalId != null) {
      clearInterval(this.checkpointIntervalId);
      this.checkpointIntervalId = null;
    }
  }

  private writeCheckpoint(): void {
    if (!this.state.started || !this.sampleBuffer.length) return;
    const snapshotActiveMs = this.state.cumulativeActiveMs +
      (!this.state.paused && this.state.activeSegmentStartMs > 0
        ? Date.now() - this.state.activeSegmentStartMs
        : 0);
    try {
      sessionStorage.setItem(this.CHECKPOINT_KEY, JSON.stringify({
        sessionStartTime:   this.state.sessionStartTime,
        cumulativeActiveMs: snapshotActiveMs,
        samples:            this.sampleBuffer,
      }));
    } catch (e) {
      console.warn('BB RDGA: checkpoint write failed (storage full?)', e);
    }
  }

  private loadCheckpoint(): { sessionStartTime: number; cumulativeActiveMs?: number; samples: any[] } | null {
    try {
      const raw = sessionStorage.getItem(this.CHECKPOINT_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (!parsed?.samples?.length || !parsed?.sessionStartTime) return null;
      return parsed;
    } catch {
      return null;
    }
  }

  private clearCheckpoint(): void {
    try {
      sessionStorage.removeItem(this.CHECKPOINT_KEY);
      this.sampleBuffer = [];
    } catch { /* ignore */ }
  }

  // ── Internal: MediaPipe face landmarker (lazy singleton) ──────────────────

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private getFaceLandmarker(): Promise<any> {
    if (this.landmarkerPromise) return this.landmarkerPromise;

    // Primary path: use the landmarker pre-loaded by index.html while window.Module was clean.
    // face_mesh (GazeDetectionService) sets non-configurable, non-writable poisoned getters on
    // window.Module after it initialises — by reusing the pre-loaded instance we bypass the
    // window.Module conflict entirely, on both first load and every rejoin.
    if (window.__rdgaLandmarkerPromise) {
      this.landmarkerPromise = (window.__rdgaLandmarkerPromise as Promise<any>)
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        .then((landmarker: any) => { this.faceModelReady = true; return landmarker; })
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        .catch((err: any) => {
          this.landmarkerPromise = null;
          (window as any).__rdgaLandmarkerPromise = null; // allow retry next call
          throw err;
        });
      return this.landmarkerPromise;
    }

    // Fallback: __rdgaLandmarkerPromise not available (index.html preload skipped or failed).
    // Attempt best-effort window.Module cleanup before loading tasks-vision wasm directly.
    let FaceLandmarker  = window.FaceLandmarker;
    let FilesetResolver = window.FilesetResolver;

    if (!FaceLandmarker && window.tasksVision) {
      FaceLandmarker  = window.tasksVision.FaceLandmarker;
      FilesetResolver = window.tasksVision.FilesetResolver;
    }

    if (!FaceLandmarker || !FilesetResolver) {
      return Promise.reject(new Error(
        'BB RDGA: FaceLandmarker / FilesetResolver not found on window. ' +
        'Make sure @mediapipe/tasks-vision is loaded via <script> before this service runs.'
      ));
    }

    // Best-effort cleanup of window.Module poisoned getters (may fail if non-configurable).
    try { delete (window as any).Module; } catch {
      try { (window as any).Module = {}; } catch {
        try {
          const mod = (window as any).Module;
          if (mod) {
            for (const key of Object.getOwnPropertyNames(mod)) {
              try {
                const desc = Object.getOwnPropertyDescriptor(mod, key);
                if (desc?.get) {
                  Object.defineProperty(mod, key, { value: undefined, configurable: true, writable: true });
                }
              } catch { /* non-configurable — skip */ }
            }
          }
        } catch { /* ignore */ }
      }
    }

    const cdnPrefix = window._hireproCdnDomain;
    let mediapipeWasmPath = cdnPrefix ? cdnPrefix + '/thirdParty/npm/@mediapipe/tasks-vision@0.10.14/wasm' : '/thirdParty/npm/@mediapipe/tasks-vision@0.10.14/wasm';
    let mediapipeModelPath = cdnPrefix ? cdnPrefix + '/thirdParty/models/face_landmarker/face_landmarker.task' : '/thirdParty/models/face_landmarker/face_landmarker.task';

    this.landmarkerPromise = FilesetResolver.forVisionTasks(mediapipeWasmPath)
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      .then((wasm: any) => FaceLandmarker.createFromOptions(wasm, {
        baseOptions: {
          modelAssetPath: mediapipeModelPath,
          delegate: 'CPU',
        },
        runningMode: 'IMAGE',
        numFaces: 1,
        outputFaceBlendshapes: true,
        minFaceDetectionConfidence: 0.4,
        minFacePresenceConfidence: 0.4,
      }))
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      .then((landmarker: any) => { this.faceModelReady = true; return landmarker; })
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      .catch((err: any) => { this.landmarkerPromise = null; throw err; });

    return this.landmarkerPromise;
  }

  // ── Internal: gaze math helpers ───────────────────────────────────────────
  //
  //   Nose-only formula calibrated on real Accenture interview videos:
  //   Ankita (cheat): yaw mean=6.4 std=3.4  pitch mean=7.2 std=2.3
  //   Ramya  (cheat): yaw mean=3.2 std=3.8  pitch mean=6.2 std=3.5
  //   Amit   (cheat): yaw mean=-1.0 std=2.4 pitch mean=8.4 std=1.7
  //   Niranjan (ok):  yaw mean=3.2 std=4.2  pitch mean=9.6 std=2.4
  //   Kavya  (ok):    yaw mean=5.7 std=3.2  pitch mean=5.3 std=1.7

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private estimateYawPitchNose(landmarks: any[]): { yawDeg: number; pitchDeg: number } {
    const le       = landmarks[LMIDX.LEFT_EYE_OUTER];
    const re       = landmarks[LMIDX.RIGHT_EYE_OUTER];
    const nose     = landmarks[LMIDX.NOSE_TIP];
    const chin     = landmarks[LMIDX.CHIN];
    const forehead = landmarks[LMIDX.FOREHEAD];

    if (!le || !re || !nose) return { yawDeg: 0, pitchDeg: 0 };

    const eyeMidX  = (le.x + re.x) / 2;
    const eyeW     = Math.abs(re.x - le.x) || 1e-6;
    const yawDeg   = ((nose.x - eyeMidX) / eyeW) * 85;

    const eyeMidY    = (le.y + re.y) / 2;
    const chinY      = chin      ? chin.y      : 0;
    const foreheadY  = forehead  ? forehead.y  : 0;
    const faceH      = Math.abs(chinY - foreheadY) || 1e-6;
    const pitchDeg   = ((nose.y - eyeMidY) / faceH - 0.12) * 75;

    return { yawDeg, pitchDeg };
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private jawOpenScore(blendshapes: any[]): number {
    if (!blendshapes || !blendshapes.length) return 0;
    const cats = blendshapes[0]?.categories ?? [];
    for (const cat of cats) {
      if (cat.categoryName === 'jawOpen') return cat.score;
    }
    return 0;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private extractBlendshapes(blendshapes: any[]): { eyeLookDown: number; eyeLookUp: number; eyeLookOut: number; eyeLookIn: number; eyeBlink: number } {
    const out = { eyeLookDown: 0, eyeLookUp: 0, eyeLookOut: 0, eyeLookIn: 0, eyeBlink: 0 };
    if (!blendshapes || !blendshapes.length) return out;
    const cats: any[] = blendshapes[0]?.categories ?? [];
    const scores: Record<string, number> = {};
    for (const cat of cats) { scores[cat.categoryName] = cat.score; }
    const avg = (a: string, b: string) => ((scores[a] || 0) + (scores[b] || 0)) / 2;
    out.eyeLookDown = avg('eyeLookDownLeft', 'eyeLookDownRight');
    out.eyeLookUp   = avg('eyeLookUpLeft',   'eyeLookUpRight');
    out.eyeLookOut  = avg('eyeLookOutLeft',  'eyeLookOutRight');
    out.eyeLookIn   = avg('eyeLookInLeft',   'eyeLookInRight');
    out.eyeBlink    = avg('eyeBlinkLeft',    'eyeBlinkRight');
    return out;
  }
}
