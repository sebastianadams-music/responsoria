import { SegmentSelector } from "./selector.js";

const CURVE_LEN = 64;
const BASE_FADE_IN = new Float32Array(CURVE_LEN);
const BASE_FADE_OUT = new Float32Array(CURVE_LEN);
const BASE_SINE_WINDOW = new Float32Array(CURVE_LEN);
for (let i = 0; i < CURVE_LEN; i++) {
  const t = i / (CURVE_LEN - 1);
  BASE_FADE_IN[i] = 0.5 * (1 - Math.cos(Math.PI * t));
  BASE_FADE_OUT[i] = 0.5 * (1 + Math.cos(Math.PI * t));
  BASE_SINE_WINDOW[i] = Math.sin(Math.PI * t);
}

const DEFAULT_FORMANT_BANDS = [180, 290, 440, 650, 920, 1280, 1750, 2350, 3150, 4200, 5600, 7400];
const BAND_MAKEUP = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.05, 1.12, 1.25, 1.45, 1.65, 1.8];
const CORPUS_JSON_PATH = "data/corpus_1000ms/corpus.json";

function scaledCurve(base, peak) {
  const out = new Float32Array(base.length);
  for (let i = 0; i < base.length; i++) {
    out[i] = Math.max(0.0001, base[i] * peak);
  }
  return out;
}

function decodeAudioCompat(ctx, arrayBuffer) {
  return new Promise((resolve, reject) => {
    try {
      const maybePromise = ctx.decodeAudioData(arrayBuffer, resolve, reject);
      if (maybePromise && typeof maybePromise.then === "function") {
        maybePromise.then(resolve, reject);
      }
    } catch (err) {
      reject(err);
    }
  });
}

function getRootPrefix() {
  if (typeof window !== "undefined" && window.location && /\/v\d+(\/|$)/.test(window.location.pathname)) {
    return "../";
  }
  return "";
}

/**
 * Generates a dark, warm stereo cathedral impulse response (~1.85s).
 */
function createCathedralImpulse(ctx, duration = 1.85, decay = 3.2) {
  const sr = ctx.sampleRate;
  const length = Math.floor(sr * duration);
  const impulse = ctx.createBuffer(2, length, sr);

  for (let ch = 0; ch < 2; ch++) {
    const data = impulse.getChannelData(ch);
    let lp = 0;
    const alpha = 0.22; // gentle high-frequency damping for dark acoustic space
    for (let i = 0; i < length; i++) {
      const t = i / length;
      const env = Math.pow(1 - t, decay) * Math.min(1, i / (sr * 0.015));
      const white = Math.random() * 2 - 1;
      lp = lp + alpha * (white - lp);
      data[i] = lp * env;
    }
  }
  return impulse;
}

export class ResponsoriaAudioEngine {
  constructor({ onModeChange, onNoteTriggered } = {}) {
    this.onModeChange = onModeChange || (() => {});
    this.onNoteTriggered = onNoteTriggered || (() => {});

    this.ctx = null;
    this.masterGain = null;
    this.dryGain = null;
    this.wetGain = null;
    this.convolver = null;
    this.compressor = null;
    this.grainBus = null;
    this.directGain = null;
    this.formantSum = null;
    this.bandFilters = [];
    this.bandGains = [];

    this.corpus = null;
    this.timeline = null;
    this.formantData = null;
    this.envFps = 20;
    this.envBands = DEFAULT_FORMANT_BANDS;
    this.envQ = 3.5;
    this.envFrames = 0;
    this.envStride = 1 + DEFAULT_FORMANT_BANDS.length;

    this.selector = null;
    this.spriteBuffers = new Map();
    this.loadedSprites = new Set();
    this.spriteFetchPromises = [];
    this._spriteLoadPromise = null;
    this._unlockedPrimed = false;

    this.mode = "user";
    this.userPos = { x: 0.5, y: 0.5 };

    this.isDataReady = false;
    this.isRunning = false;
    this.startTime = null;
    this.loopCount = 0;
    this.nextNoteIdx = 0;
    this.nextPhraseIdx = 0;
    this.envLoopCount = 0;
    this.nextEnvFrameIdx = 0;
    this.lastEnvFrameTime = -1;
    this.activeSustain = null;
    this.schedulerTimer = null;
    this.dataPromise = null;
  }

  setUserPosition(x, y) {
    this.userPos.x = Math.min(1, Math.max(0, x));
    this.userPos.y = Math.min(1, Math.max(0, y));
    if (this.selector) {
      this.selector.updateLastUserPos(this.userPos.x, this.userPos.y);
    }
  }

  async preloadMetadata() {
    if (this.dataPromise) return this.dataPromise;
    this.dataPromise = (async () => {
      const prefix = getRootPrefix();
      const [corpusRes, timelineRes, formantRes] = await Promise.all([
        fetch(`${prefix}${CORPUS_JSON_PATH}`),
        fetch(`${prefix}data/timeline.json`),
        fetch(`${prefix}data/formants.bin`),
      ]);
      this.corpus = await corpusRes.json();
      this.timeline = await timelineRes.json();
      this.formantData = new Uint8Array(await formantRes.arrayBuffer());

      this.envFps = this.timeline.envFps || 20;
      this.envBands = this.timeline.envBands || DEFAULT_FORMANT_BANDS;
      this.envQ = this.timeline.envQ || 3.5;
      this.envStride = 1 + this.envBands.length;
      this.envFrames =
        this.timeline.envFrames || Math.floor(this.formantData.length / this.envStride);

      this.selector = new SegmentSelector(this.corpus.segments);
      this.selector.updateLastUserPos(this.userPos.x, this.userPos.y);

      // Pre-fetch sprite bytes immediately in background so playback starts instantly on click
      this.spriteFetchPromises = this.corpus.sprites.map((url) =>
        fetch(`${prefix}${url}`)
          .then((r) => (r.ok ? r.arrayBuffer() : null))
          .catch(() => null)
      );

      this.isDataReady = true;
    })();
    return this.dataPromise;
  }

  _ensureContext() {
    if (this.ctx) return;
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    this.ctx = new AudioCtx();

    this.compressor = this.ctx.createDynamicsCompressor();
    this.compressor.threshold.value = -12;
    this.compressor.knee.value = 18;
    this.compressor.ratio.value = 4;
    this.compressor.attack.value = 0.005;
    this.compressor.release.value = 0.25;

    this.masterGain = this.ctx.createGain();
    this.masterGain.gain.value = 0.9;

    this.dryGain = this.ctx.createGain();
    this.dryGain.gain.value = 0.78;

    this.wetGain = this.ctx.createGain();
    this.wetGain.gain.value = 0.26;

    this.convolver = this.ctx.createConvolver();
    this.convolver.buffer = createCathedralImpulse(this.ctx);

    this.grainBus = this.ctx.createGain();
    this.grainBus.gain.value = 1.0;

    this.directGain = this.ctx.createGain();
    this.directGain.gain.value = 0.0001;

    this.formantSum = this.ctx.createGain();
    this.formantSum.gain.value = 1.85;

    this.grainBus.connect(this.directGain);
    this.directGain.connect(this.dryGain);
    this.directGain.connect(this.wetGain);

    this.formantSum.connect(this.dryGain);
    this.formantSum.connect(this.wetGain);

    this.dryGain.connect(this.compressor);
    this.wetGain.connect(this.convolver);
    this.convolver.connect(this.compressor);
    this.compressor.connect(this.masterGain);
    this.masterGain.connect(this.ctx.destination);
  }

  _ensureFormantBank() {
    if (!this.ctx || !this.grainBus || !this.formantSum || this.bandGains.length > 0) return;
    for (let b = 0; b < this.envBands.length; b++) {
      const filter = this.ctx.createBiquadFilter();
      filter.type = "bandpass";
      filter.frequency.value = this.envBands[b];
      filter.Q.value = this.envQ;

      const gain = this.ctx.createGain();
      gain.gain.value = 0.0001;

      this.grainBus.connect(filter);
      filter.connect(gain);
      gain.connect(this.formantSum);

      this.bandFilters.push(filter);
      this.bandGains.push(gain);
    }
  }

  unlockFromGesture() {
    try {
      if (typeof navigator !== "undefined" && navigator.audioSession) {
        navigator.audioSession.type = "playback";
      }
    } catch {}

    this._ensureContext();

    if (this.ctx.state !== "running") {
      try {
        const p = this.ctx.resume();
        if (p && typeof p.catch === "function") p.catch(() => {});
      } catch {}
    }

    if (!this._unlockedPrimed) {
      try {
        const buf = this.ctx.createBuffer(1, 1, this.ctx.sampleRate || 44100);
        const src = this.ctx.createBufferSource();
        src.buffer = buf;
        src.connect(this.ctx.destination);
        src.start(0);
        this._unlockedPrimed = true;
      } catch {}
    }
  }

  async _loadSprites() {
    if (!this.corpus) return;
    if (this._spriteLoadPromise) return this._spriteLoadPromise;

    const prefix = getRootPrefix();
    const decodeSprite = async (url, idx) => {
      let arr = this.spriteFetchPromises[idx] ? await this.spriteFetchPromises[idx] : null;
      if (!arr) {
        const res = await fetch(`${prefix}${url}`);
        if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
        arr = await res.arrayBuffer();
      }
      const decoded = await decodeAudioCompat(this.ctx, arr.slice(0));
      this.spriteBuffers.set(idx, decoded);
      this.loadedSprites.add(idx);
      return idx;
    };

    const tasks = this.corpus.sprites.map((url, idx) =>
      decodeSprite(url, idx).catch((err) => {
        console.error(`Failed loading sprite ${idx}:`, err);
        throw err;
      })
    );

    this._spriteLoadPromise = Promise.any(tasks)
      .catch(() => {})
      .finally(() => {
        if (this.loadedSprites.size === 0) {
          this._spriteLoadPromise = null;
        }
      });

    return this._spriteLoadPromise;
  }

  async start() {
    this.unlockFromGesture();

    await this.preloadMetadata();
    if (this.loadedSprites.size === 0) {
      await this._loadSprites();
    }

    if (!this.isRunning && this.loadedSprites.size > 0) {
      this._ensureFormantBank();
      this.isRunning = true;
      this.startTime = null;
      if (!this.schedulerTimer) {
        this.schedulerTimer = setInterval(() => this._schedule(), 25);
      }
    }
    return this.isRunning;
  }

  _schedule() {
    if (!this.isRunning || !this.ctx || this.ctx.state !== "running" || this.loadedSprites.size === 0) {
      return;
    }

    if (this.startTime === null) {
      this.startTime = this.ctx.currentTime + 0.02;
      this.loopCount = 0;
      this.nextNoteIdx = 0;
      this.nextPhraseIdx = 0;
      this.envLoopCount = 0;
      this.nextEnvFrameIdx = 0;
      this.lastEnvFrameTime = this.startTime - 0.001;
      this.activeSustain = null;

      this.directGain.gain.setValueAtTime(0.0001, this.startTime);
      for (let b = 0; b < this.bandGains.length; b++) {
        this.bandGains[b].gain.setValueAtTime(0.0001, this.startTime);
      }
    }

    const lookaheadSec = 0.12;
    const now = this.ctx.currentTime;
    const horizon = now + lookaheadSec;
    const duration = this.timeline.duration;
    const notes = this.timeline.notes;
    const phraseEnds = this.timeline.phraseEnds;

    if (duration <= 0 || notes.length === 0) return;

    this._scheduleEnvelope(horizon);

    // Process events chronologically across loops until next event exceeds horizon
    let guard = 0;
    while (guard++ < 128) {
      const loopBase = this.startTime + this.loopCount * duration;
      const nextNoteTime =
        this.nextNoteIdx < notes.length ? loopBase + notes[this.nextNoteIdx][0] : Infinity;
      const nextPhraseTime =
        this.nextPhraseIdx < phraseEnds.length ? loopBase + phraseEnds[this.nextPhraseIdx] : Infinity;
      const nextSustainTime = this.activeSustain ? this.activeSustain.nextGrainTime : Infinity;

      const nextEventTime = Math.min(nextNoteTime, nextPhraseTime, nextSustainTime);
      if (nextEventTime > horizon) {
        if (
          !this.activeSustain &&
          this.nextNoteIdx >= notes.length &&
          this.nextPhraseIdx >= phraseEnds.length
        ) {
          this.loopCount += 1;
          this.nextNoteIdx = 0;
          this.nextPhraseIdx = 0;
          continue;
        }
        break;
      }

      // Handle phrase transitions, primary notes, and same-sample granular sustain in exact time order
      if (nextPhraseTime <= nextNoteTime && nextPhraseTime <= nextSustainTime) {
        this.nextPhraseIdx += 1;
        this._toggleModeAt(nextPhraseTime);
      } else if (nextNoteTime <= nextSustainTime) {
        const note = notes[this.nextNoteIdx];
        this.nextNoteIdx += 1;
        const followingNoteTime =
          this.nextNoteIdx < notes.length ? loopBase + notes[this.nextNoteIdx][0] : loopBase + duration;
        const nextBoundaryTime = Math.min(followingNoteTime, nextPhraseTime);
        this._triggerNote(note, nextNoteTime, nextBoundaryTime);
      } else {
        const sustain = this.activeSustain;
        if (
          !sustain ||
          nextSustainTime >= sustain.endTime - 0.04 ||
          nextSustainTime >= nextNoteTime - 0.03 ||
          nextSustainTime >= nextPhraseTime - 0.03
        ) {
          this.activeSustain = null;
        } else {
          this._triggerSustainGrain(nextSustainTime);
        }
      }
    }
  }

  _scheduleEnvelope(horizon) {
    if (!this.formantData || this.envFrames <= 0 || this.bandGains.length === 0) return;
    const duration = this.timeline.duration;
    const envStep = 1 / this.envFps;
    const bandCount = this.bandGains.length;

    let guard = 0;
    while (guard++ < 64) {
      const frameTime =
        this.startTime + this.envLoopCount * duration + this.nextEnvFrameIdx * envStep;
      if (frameTime > horizon) break;

      const safeTime = Math.max(this.lastEnvFrameTime + 0.001, frameTime);
      this.lastEnvFrameTime = safeTime;

      const base = this.nextEnvFrameIdx * this.envStride;
      const bb = this.formantData[base] / 255;
      const directTarget = Math.max(0.0001, 0.30 * Math.pow(bb, 0.9));
      this.directGain.gain.linearRampToValueAtTime(directTarget, safeTime);

      for (let b = 0; b < bandCount; b++) {
        const bVal = this.formantData[base + 1 + b] / 255;
        const makeup = BAND_MAKEUP[b] ?? 1.0;
        const bandTarget = Math.max(0.0001, bVal * makeup);
        this.bandGains[b].gain.linearRampToValueAtTime(bandTarget, safeTime);
      }

      this.nextEnvFrameIdx += 1;
      if (this.nextEnvFrameIdx >= this.envFrames) {
        this.nextEnvFrameIdx = 0;
        this.envLoopCount += 1;
      }
    }
  }

  _toggleModeAt(when) {
    this.activeSustain = null;
    this.mode = this.mode === "user" ? "answer" : "user";
    if (this.mode === "answer") {
      this.selector.beginAnswerPhrase();
    } else {
      this.selector.beginUserPhrase();
    }
    const modeSnapshot = this.mode;
    const delayMs = Math.max(0, (when - this.ctx.currentTime) * 1000);
    setTimeout(() => {
      this.onModeChange(modeSnapshot);
    }, delayMs);
  }

  _playGrain({ seg, when, playDur, targetGain, fadeIn, fadeOut }) {
    const buffer = this.spriteBuffers.get(seg.spriteIdx);
    if (!buffer) return null;

    const safeWhen = Math.max(when, this.ctx.currentTime + 0.005);
    const dur = Math.max(0.05, Math.min(seg.dur, playDur));

    let fi = Math.max(0.005, fadeIn);
    let fo = Math.max(0.005, fadeOut);
    if (fi + fo > dur * 0.95) {
      const scale = (dur * 0.95) / (fi + fo);
      fi = Math.max(0.004, fi * scale);
      fo = Math.max(0.004, fo * scale);
    }

    const src = this.ctx.createBufferSource();
    src.buffer = buffer;

    const env = this.ctx.createGain();
    env.gain.value = 0.0001;
    const fadeOutStart = Math.max(safeWhen + fi + 0.002, safeWhen + dur - fo);
    const fadeOutDur = Math.max(0.003, safeWhen + dur - fadeOutStart);
    env.gain.setValueCurveAtTime(scaledCurve(BASE_FADE_IN, targetGain), safeWhen, fi);
    env.gain.setValueCurveAtTime(scaledCurve(BASE_FADE_OUT, targetGain), fadeOutStart, fadeOutDur);

    src.connect(env);
    env.connect(this.grainBus);

    src.start(safeWhen, seg.start, dur);
    return { safeWhen, dur };
  }

  _playWindowedGrain({ spriteIdx, offset, when, dur, peakGain }) {
    const buffer = this.spriteBuffers.get(spriteIdx);
    if (!buffer) return null;

    const safeWhen = Math.max(when, this.ctx.currentTime + 0.005);
    const safeDur = Math.max(0.04, dur);

    const src = this.ctx.createBufferSource();
    src.buffer = buffer;

    const env = this.ctx.createGain();
    env.gain.value = 0.0001;
    env.gain.setValueCurveAtTime(scaledCurve(BASE_SINE_WINDOW, peakGain), safeWhen, safeDur);

    src.connect(env);
    env.connect(this.grainBus);

    src.start(safeWhen, offset, safeDur);
    return { safeWhen, dur: safeDur };
  }

  _triggerNote([, amp, dur], when, nextBoundaryTime) {
    this.activeSustain = null;
    const activeMode = this.mode;
    const pick =
      activeMode === "user"
        ? this.selector.selectUser(this.userPos.x, this.userPos.y, 5, this.loadedSprites, { record: true })
        : this.selector.selectAnswer(5, this.loadedSprites, { advance: true });

    if (!pick) return;
    const { seg } = pick;

    const safeWhen = Math.max(when, this.ctx.currentTime + 0.005);
    const sustainSpan = Math.max(0.2, Math.min(dur, nextBoundaryTime - when));
    const targetGain = (0.72 + 0.28 * Math.pow(amp, 0.5)) * seg.gainComp * 0.65;

    let played;
    if (seg.dur >= sustainSpan + 0.08) {
      played = this._playGrain({
        seg,
        when: safeWhen,
        playDur: sustainSpan + 0.14,
        targetGain,
        fadeIn: 0.035,
        fadeOut: 0.16,
      });
      if (!played) return;
    } else {
      const initialDur = Math.max(0.06, Math.min(seg.dur, seg.dur * 0.90));
      const fadeIn = Math.min(0.035, initialDur * 0.28);
      const fadeOut = Math.min(0.14, initialDur * 0.44);
      played = this._playGrain({
        seg,
        when: safeWhen,
        playDur: initialDur,
        targetGain,
        fadeIn,
        fadeOut,
      });
      if (!played) return;

      const coreStartFrac = seg.dur < 0.16 ? 0.12 : 0.20;
      const coreEndFrac = seg.dur < 0.16 ? 0.92 : 0.86;
      const coreStart = seg.start + seg.dur * coreStartFrac;
      const coreEnd = seg.start + seg.dur * coreEndFrac;
      const coreSpan = Math.max(0.05, coreEnd - coreStart);
      const grainDur = Math.min(0.18, Math.max(0.052, coreSpan * 0.68), coreSpan * 0.92);
      const travel = Math.max(0, coreSpan - grainDur);
      const nextGrainTime = played.safeWhen + Math.max(0.03, played.dur - fadeOut * 0.85);

      if (nextGrainTime < played.safeWhen + sustainSpan - 0.04) {
        this.activeSustain = {
          seg,
          coreStart,
          travel,
          grainDur,
          targetGain: targetGain * 0.96,
          endTime: played.safeWhen + sustainSpan,
          nextGrainTime,
          scanPhase: 0.25,
          scanDir: 1,
        };
      }
    }

    const delayMs = Math.max(0, (played.safeWhen - this.ctx.currentTime) * 1000);
    setTimeout(() => {
      this.onNoteTriggered({
        mode: activeMode,
        x: seg.x,
        y: seg.y,
        amplitude: amp,
        when: played.safeWhen,
      });
    }, delayMs);
  }

  _triggerSustainGrain(when) {
    const sustain = this.activeSustain;
    if (!sustain) return;

    const jitter = (Math.random() - 0.5) * 0.28;
    const normPos = Math.min(1, Math.max(0, sustain.scanPhase + jitter));
    const offset = sustain.coreStart + normPos * sustain.travel;

    sustain.scanPhase += sustain.scanDir * 0.34;
    if (sustain.scanPhase >= 1) {
      sustain.scanPhase = Math.max(0, 2 - sustain.scanPhase);
      sustain.scanDir = -1;
    } else if (sustain.scanPhase <= 0) {
      sustain.scanPhase = Math.min(1, -sustain.scanPhase);
      sustain.scanDir = 1;
    }

    const maxSafeFromSeg = Math.max(0.045, sustain.seg.start + sustain.seg.dur - offset);
    const gDur = Math.max(
      0.045,
      Math.min(sustain.grainDur, Math.max(0.055, sustain.endTime + 0.10 - when), maxSafeFromSeg)
    );

    const played = this._playWindowedGrain({
      spriteIdx: sustain.seg.spriteIdx,
      offset,
      when,
      dur: gDur,
      peakGain: sustain.targetGain,
    });
    if (!played) {
      this.activeSustain = null;
      return;
    }

    const hop = played.dur * (0.47 + Math.random() * 0.06);
    const nextGrainTime = played.safeWhen + Math.max(0.025, hop);
    if (nextGrainTime < sustain.endTime - 0.04) {
      sustain.nextGrainTime = nextGrainTime;
    } else {
      this.activeSustain = null;
    }
  }
}
