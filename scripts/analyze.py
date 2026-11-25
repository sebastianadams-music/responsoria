import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RAW_DIR = Path("raw_audio")
OUT_DIR = Path("public/data")

SR_ANALYSIS = 22050
SR_SPRITE = 32000
HOP_LENGTH = 512
FRAME_LENGTH = 2048
INTER_TRACK_GAP = 2.0

ENV_FPS = 20
ENV_HOP = int(round(SR_ANALYSIS / ENV_FPS))
FORMANT_CENTERS = [180, 290, 440, 650, 920, 1280, 1750, 2350, 3150, 4200, 5600, 7400]
FORMANT_Q = 3.5

NUM_SPRITES = 5
TARGET_SEGMENTS = 800
TARGET_PACKED_SEC = 350.0


def zscore(x: np.ndarray) -> np.ndarray:
    std = np.std(x)
    if std < 1e-8:
        return np.zeros_like(x)
    return (x - np.mean(x)) / std


def winsorize_and_map(values: np.ndarray, low_pct=2.0, high_pct=98.0, out_min=0.02, out_max=0.98) -> np.ndarray:
    lo = np.percentile(values, low_pct)
    hi = np.percentile(values, high_pct)
    if hi - lo < 1e-9:
        return np.full_like(values, 0.5 * (out_min + out_max), dtype=np.float64)
    clipped = np.clip(values, lo, hi)
    normed = (clipped - lo) / (hi - lo)
    return out_min + normed * (out_max - out_min)


def compute_track_analysis(wav_path: Path, analysis_only: bool = False, trim_leading_silence: bool = False):
    # Load stereo for sprite extraction & mono for DSP
    y_stereo, sr_orig = sf.read(str(wav_path), always_2d=True, dtype="float32")
    y_mono_orig = np.mean(y_stereo, axis=1)

    if sr_orig != SR_ANALYSIS:
        y_mono = librosa.resample(y_mono_orig, orig_sr=sr_orig, target_sr=SR_ANALYSIS)
    else:
        y_mono = y_mono_orig

    if analysis_only:
        y_st = None
    elif sr_orig != SR_SPRITE:
        y_st_l = librosa.resample(y_stereo[:, 0], orig_sr=sr_orig, target_sr=SR_SPRITE)
        y_st_r = librosa.resample(y_stereo[:, 1], orig_sr=sr_orig, target_sr=SR_SPRITE)
        y_st = np.stack([y_st_l, y_st_r], axis=1)
    else:
        y_st = y_stereo

    if trim_leading_silence:
        rms_init = librosa.feature.rms(y=y_mono, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
        rms_init_fast = uniform_filter1d(rms_init, size=3, mode="nearest")
        active_init = rms_init_fast[rms_init_fast > 1e-6]
        ref_init = max(float(np.percentile(active_init, 95)) if len(active_init) > 0 else 1e-3, 1e-5)
        db_init = 20.0 * np.log10(np.maximum(rms_init_fast, 1e-7) / ref_init)
        hits = np.where(db_init >= -24.0)[0]
        if len(hits) > 0:
            trim_sec = max(0.0, float(hits[0] * HOP_LENGTH / SR_ANALYSIS) - 0.12)
            if trim_sec > 0.05:
                y_mono = y_mono[int(round(trim_sec * SR_ANALYSIS)):]
                if y_st is not None:
                    y_st = y_st[int(round(trim_sec * SR_SPRITE)):]
                print(f"    Trimmed {trim_sec:.2f}s of leading silence from {wav_path.name}")

    duration = len(y_mono) / SR_ANALYSIS
    fps = SR_ANALYSIS / HOP_LENGTH

    # RMS energy envelope
    rms = librosa.feature.rms(y=y_mono, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    rms_fast = uniform_filter1d(rms, size=3, mode="nearest")   # ~70 ms
    rms_slow = uniform_filter1d(rms, size=7, mode="nearest")   # ~162 ms
    rms_phrase = uniform_filter1d(rms, size=max(1, int(round(0.40 * fps))), mode="nearest")  # ~400 ms

    active_rms = rms_fast[rms_fast > 1e-6]
    ref_rms = np.percentile(active_rms, 95) if len(active_rms) > 0 else 1e-3
    ref_rms = max(float(ref_rms), 1e-5)

    rms_fast_db = 20.0 * np.log10(np.maximum(rms_fast, 1e-7) / ref_rms)
    rms_slow_db = 20.0 * np.log10(np.maximum(rms_slow, 1e-7) / ref_rms)
    rms_phrase_db = 20.0 * np.log10(np.maximum(rms_phrase, 1e-7) / ref_rms)
    rms_norm = np.clip(rms_fast / ref_rms, 0.0, 1.0)

    # Superflux onset strength (vibrato-resistant)
    S_mel = librosa.feature.melspectrogram(
        y=y_mono, sr=SR_ANALYSIS, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH, n_mels=128, fmin=60, fmax=9000
    )
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)
    superflux = librosa.onset.onset_strength(S=S_mel_db, sr=SR_ANALYSIS, hop_length=HOP_LENGTH, lag=1, max_size=3)

    # Positive dB derivative for note envelope
    db_diff = np.maximum(0.0, np.diff(rms_fast_db, prepend=rms_fast_db[0]))

    # 20 Hz broadband + 12-band vocal-tract formant envelopes for cross-synthesis
    S_env = np.abs(librosa.stft(y_mono, n_fft=FRAME_LENGTH, hop_length=ENV_HOP))
    freqs_env = librosa.fft_frequencies(sr=SR_ANALYSIS, n_fft=FRAME_LENGTH)
    f_safe = np.maximum(freqs_env, 1.0)
    band_envs = np.zeros((len(FORMANT_CENTERS), S_env.shape[1]), dtype=np.float32)
    for b, fc in enumerate(FORMANT_CENTERS):
        H = 1.0 / np.sqrt(1.0 + (FORMANT_Q * (f_safe / fc - fc / f_safe)) ** 2)
        H[0] = 0.0
        band_envs[b] = np.sqrt(np.sum((S_env * H[:, None]) ** 2, axis=0) / (np.sum(H ** 2) + 1e-8))
        band_envs[b] = uniform_filter1d(band_envs[b], size=2, mode="nearest")

    bb_raw = librosa.feature.rms(y=y_mono, frame_length=FRAME_LENGTH, hop_length=ENV_HOP)[0]
    n_env_frames = min(len(bb_raw), band_envs.shape[1])
    env_broadband = uniform_filter1d(bb_raw[:n_env_frames], size=2, mode="nearest").astype(np.float32)
    env_bands = band_envs[:, :n_env_frames]

    if analysis_only:
        hf_flux = None
        mfcc_12 = None
        mfcc_delta = None
        centroid = None
    else:
        # STFT for HF flux & spectral centroid
        S_mag = np.abs(librosa.stft(y_mono, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH))
        freqs = librosa.fft_frequencies(sr=SR_ANALYSIS, n_fft=FRAME_LENGTH)
        hf_mask = (freqs >= 2000.0) & (freqs <= 8000.0)
        S_hf = np.log1p(S_mag[hf_mask, :])
        hf_flux = np.sum(np.maximum(0.0, np.diff(S_hf, axis=1, prepend=S_hf[:, :1])), axis=0)

        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(S=S_mag, sr=SR_ANALYSIS)[0]
        centroid = np.maximum(centroid, 60.0)

        # MFCCs [1:13]
        mfcc_all = librosa.feature.mfcc(S=S_mel_db, n_mfcc=13)
        mfcc_12 = mfcc_all[1:13, :]  # shape (12, T)
        mfcc_delta = np.linalg.norm(np.diff(mfcc_12, axis=1, prepend=mfcc_12[:, :1]), axis=0)

    return {
        "y_st": y_st,
        "y_mono": y_mono,
        "duration": duration,
        "rms_fast": rms_fast,
        "rms_fast_db": rms_fast_db,
        "rms_slow_db": rms_slow_db,
        "rms_phrase_db": rms_phrase_db,
        "rms_norm": rms_norm,
        "ref_rms": ref_rms,
        "superflux": superflux,
        "db_diff": db_diff,
        "env_broadband": env_broadband,
        "env_bands": env_bands,
        "hf_flux": hf_flux,
        "mfcc_12": mfcc_12,
        "mfcc_delta": mfcc_delta,
        "centroid": centroid,
    }


def detect_phrases_dp(track_data, note_times, offset_sec: float, min_dur: float = 15.0, max_dur: float = 40.0):
    fps = SR_ANALYSIS / HOP_LENGTH
    rms_phrase_db = track_data["rms_phrase_db"]
    duration = track_data["duration"]
    movement_end = duration + 0.5 * INTER_TRACK_GAP

    # Find local valleys in the 400ms-smoothed dB envelope
    valleys, props = find_peaks(-rms_phrase_db, distance=max(1, int(round(1.5 * fps))), prominence=2.0)

    raw_cands = []
    for v, prom in zip(valleys, props["prominences"]):
        t_v = float(v / fps)
        if t_v < min_dur - 1.0 or t_v > movement_end - (min_dur - 1.0):
            continue
        cadence_score = float(prom + 0.55 * max(0.0, -float(rms_phrase_db[v]) - 8.0))
        raw_cands.append((t_v, cadence_score))

    # Deduplicate candidates closer than 0.5s keeping higher score
    raw_cands.sort(key=lambda x: x[0])
    dedup = []
    for t_c, sc in raw_cands:
        if dedup and (t_c - dedup[-1][0]) < 0.5:
            if sc > dedup[-1][1]:
                dedup[-1] = (t_c, sc)
        else:
            dedup.append((t_c, sc))

    def run_dp(cands_list):
        nodes = [(0.0, 0.0)] + sorted(cands_list, key=lambda x: x[0]) + [(movement_end, 20.0)]
        n = len(nodes)
        dp = [-1e18] * n
        parent = [-1] * n
        dp[0] = 0.0

        for i in range(n):
            if dp[i] < -1e17:
                continue
            t_i = nodes[i][0]
            for j in range(i + 1, n):
                t_j, score_j = nodes[j]
                dt = t_j - t_i
                if dt < min_dur:
                    continue
                if dt > max_dur:
                    break
                reward = score_j - 0.04 * ((dt - 24.0) ** 2) - 3.5
                cand_val = dp[i] + reward
                if cand_val > dp[j]:
                    dp[j] = cand_val
                    parent[j] = i

        return nodes, dp, parent

    nodes, dp, parent = run_dp(dedup)

    # Fallback if any stretch lacks a valley inside [15s, 40s]
    fallback_count = 0
    while dp[-1] < -1e17 and fallback_count < 32:
        # Find furthest reachable node whose gap to movement_end exceeds max_dur
        best_i = 0
        for i in range(len(nodes) - 1):
            if dp[i] > -1e17 and nodes[i][0] > nodes[best_i][0]:
                best_i = i
        t_start = nodes[best_i][0]
        lo_t = t_start + min_dur
        hi_t = min(movement_end - min_dur, t_start + max_dur)
        if hi_t < lo_t:
            hi_t = min(movement_end, t_start + max_dur)
        f0 = max(0, int(round(lo_t * fps)))
        f1 = min(len(rms_phrase_db), max(f0 + 1, int(round(hi_t * fps))))
        f_min = f0 + int(np.argmin(rms_phrase_db[f0:f1]))
        t_fb = float(f_min / fps)
        dedup.append((t_fb, 0.0))
        dedup.sort(key=lambda x: x[0])
        nodes, dp, parent = run_dp(dedup)
        fallback_count += 1

    # Reconstruct path
    chosen_rel = []
    cur = len(nodes) - 1
    if dp[cur] > -1e17:
        while cur > 0:
            chosen_rel.append(nodes[cur][0])
            cur = parent[cur]
        chosen_rel.reverse()
    else:
        # Extreme fallback: uniform slices inside [min_dur, max_dur]
        t_cur = 0.0
        while movement_end - t_cur > max_dur:
            t_cur += 24.0
            chosen_rel.append(t_cur)
        chosen_rel.append(movement_end)

    # Snap internal boundaries to midpoint between surrounding notes if it preserves [14.5, 40.5]
    snapped = list(chosen_rel)
    for k in range(len(snapped) - 1):
        t_b = snapped[k]
        if t_b < duration and len(note_times) > 1:
            idx_after = int(np.searchsorted(note_times, t_b))
            if 0 < idx_after < len(note_times):
                mid = 0.5 * (note_times[idx_after - 1] + note_times[idx_after])
            elif idx_after >= len(note_times):
                mid = max(t_b, note_times[-1] + 0.3)
            else:
                mid = t_b

            prev_b = 0.0 if k == 0 else snapped[k - 1]
            next_b = snapped[k + 1]
            if 14.5 <= (mid - prev_b) <= 40.5 and 14.5 <= (next_b - mid) <= 40.5:
                snapped[k] = mid

    return snapped, fallback_count


def extract_notes_and_phrases(track_data, offset_sec: float):
    fps = SR_ANALYSIS / HOP_LENGTH
    rms_fast_db = track_data["rms_fast_db"]
    rms_norm = track_data["rms_norm"]
    superflux = track_data["superflux"]
    db_diff = track_data["db_diff"]
    duration = track_data["duration"]

    # Combined note onset curve gated by active energy (> -34 dB)
    onset_curve = 0.55 * np.maximum(0.0, zscore(superflux)) + 0.45 * np.maximum(0.0, zscore(db_diff))
    active_gate = rms_fast_db > -34.0
    onset_gated = np.where(active_gate, onset_curve, 0.0)

    min_note_frames = max(1, int(round(0.24 * fps)))
    peaks, _ = find_peaks(onset_gated, height=0.25, distance=min_note_frames, prominence=0.18)

    note_times = [float(p / fps) for p in peaks]
    if len(note_times) == 0:
        active_idx = np.where(active_gate)[0]
        if len(active_idx) > 0:
            note_times = [float(active_idx[0] / fps)]

    # Detect 15-40s phrase endings via DP over real amplitude envelope valleys
    phrase_ends_rel, fallback_count = detect_phrases_dp(track_data, note_times, offset_sec, min_dur=15.0, max_dur=40.0)

    # Build note records with true active sustain duration up to next note or phrase end
    min_rest_frames = max(1, int(round(0.20 * fps)))
    notes = []
    p_ptr = 0
    for i, t_rel in enumerate(note_times):
        f_idx = min(len(rms_norm) - 1, max(0, int(round(t_rel * fps))))
        f_end_amp = min(len(rms_norm), f_idx + int(round(0.22 * fps)) + 1)
        amp = float(np.max(rms_norm[f_idx:f_end_amp]))
        amp = float(np.clip(max(amp, 0.08), 0.08, 1.0))

        while p_ptr < len(phrase_ends_rel) and phrase_ends_rel[p_ptr] <= t_rel:
            p_ptr += 1
        next_phrase_t = phrase_ends_rel[p_ptr] if p_ptr < len(phrase_ends_rel) else duration
        next_note_t = note_times[i + 1] if i + 1 < len(note_times) else duration
        limit_t = min(next_note_t, next_phrase_t)

        # Scan rms_fast_db forward; truncate sustain only if genuine rest (< -32 dB for >= 0.20s)
        f_scan_start = min(len(rms_fast_db), int(round((t_rel + 0.08) * fps)))
        f_scan_end = min(len(rms_fast_db), int(round(limit_t * fps)))
        sustain_end = limit_t
        if f_scan_end > f_scan_start + min_rest_frames:
            run_len = 0
            run_start = f_scan_start
            for f in range(f_scan_start, f_scan_end):
                if rms_fast_db[f] < -32.0:
                    if run_len == 0:
                        run_start = f
                    run_len += 1
                    if run_len >= min_rest_frames:
                        sustain_end = float(run_start / fps)
                        break
                else:
                    run_len = 0

        dur = float(np.clip(sustain_end - t_rel, 0.22, 8.0))
        notes.append([round(offset_sec + t_rel, 3), round(amp, 4), round(dur, 3)])

    phrase_ends_abs = [round(offset_sec + t, 3) for t in phrase_ends_rel]
    return notes, phrase_ends_abs, fallback_count


def segment_syllables(track_idx: int, track_data):
    fps = SR_ANALYSIS / HOP_LENGTH
    superflux = track_data["superflux"]
    mfcc_delta = track_data["mfcc_delta"]
    hf_flux = track_data["hf_flux"]
    rms_fast = track_data["rms_fast"]
    rms_fast_db = track_data["rms_fast_db"]
    mfcc_12 = track_data["mfcc_12"]
    centroid = track_data["centroid"]
    duration = track_data["duration"]

    # Multi-cue syllable boundary novelty
    novelty = (
        0.45 * zscore(superflux)
        + 0.35 * zscore(mfcc_delta)
        + 0.20 * zscore(hf_flux)
    )

    min_sep_frames = max(1, int(round(0.15 * fps)))
    peaks, _ = find_peaks(novelty, height=0.05, distance=min_sep_frames, prominence=0.12)

    # Refine each boundary within +-35ms toward local RMS minimum (biased slightly before burst)
    win_half = max(1, int(round(0.035 * fps)))
    refined_times = []
    for p in peaks:
        lo = max(0, p - win_half)
        hi = min(len(rms_fast), p + max(1, win_half // 2) + 1)
        local_rms = rms_fast[lo:hi]
        best_f = lo + int(np.argmin(local_rms))
        refined_times.append(float(best_f / fps))

    # Ensure sorted unique boundaries
    boundaries = sorted(set([0.0] + refined_times + [duration]))

    # Build segments clamped to [0.15s, 2.0s], subdividing >2.0s at secondary novelty peaks
    raw_spans = []
    for i in range(len(boundaries) - 1):
        t0, t1 = boundaries[i], boundaries[i + 1]
        span = t1 - t0
        if span < 0.15:
            continue
        if span <= 2.0:
            raw_spans.append((t0, t1))
        else:
            # Subdivide long sustained spans at secondary peaks or max 1.8s chunks
            cur = t0
            while t1 - cur > 2.0:
                f0 = int(round((cur + 0.35) * fps))
                f1 = min(len(novelty) - 1, int(round((cur + 1.85) * fps)))
                if f1 > f0 + 1:
                    sub_p = f0 + int(np.argmax(novelty[f0:f1]))
                    split_t = float(sub_p / fps)
                else:
                    split_t = cur + 1.5
                raw_spans.append((cur, split_t))
                cur = split_t
            if t1 - cur >= 0.15:
                raw_spans.append((cur, t1))

    segments = []
    for t0, t1 in raw_spans:
        f0 = max(0, int(round(t0 * fps)))
        f1 = min(len(rms_fast), int(round(t1 * fps)))
        if f1 <= f0:
            continue

        seg_rms_frames = rms_fast[f0:f1]
        seg_db = 20.0 * np.log10(max(float(np.mean(seg_rms_frames)), 1e-7) / track_data["ref_rms"])
        if seg_db < -34.0:
            continue

        # Voiced core: exclude outer 10% frames if segment is >= 8 frames
        core_pad = (f1 - f0) // 10 if (f1 - f0) >= 10 else 0
        c0, c1 = f0 + core_pad, f1 - core_pad
        weights = np.maximum(rms_fast[c0:c1], 1e-7)
        weights_sum = float(np.sum(weights))
        if weights_sum <= 0:
            continue

        mfcc_vec = np.sum(mfcc_12[:, c0:c1] * weights[None, :], axis=1) / weights_sum
        log_cent = float(np.sum(np.log(centroid[c0:c1]) * weights) / weights_sum)
        seg_rms = float(np.sqrt(np.mean(seg_rms_frames ** 2)))

        segments.append({
            "track_idx": track_idx,
            "t_start": t0,
            "t_end": t1,
            "dur": t1 - t0,
            "mfcc": mfcc_vec,
            "log_centroid": log_cent,
            "rms": seg_rms,
        })

    return segments


def select_stratified_corpus(segments, target_count=TARGET_SEGMENTS, target_sec=TARGET_PACKED_SEC):
    # 20 x 20 grid stratification across (x, y)
    grid_n = 20
    bins = {}
    for idx, s in enumerate(segments):
        gx = min(grid_n - 1, max(0, int(s["x"] * grid_n)))
        gy = min(grid_n - 1, max(0, int(s["y"] * grid_n)))
        bins.setdefault((gx, gy), []).append(idx)

    rng = np.random.default_rng(2026)
    selected_indices = []
    used = set()

    # Track representation tracking
    # Round-robin across non-empty grid cells so every occupied region (including extreme corners) is sampled
    cell_keys = sorted(bins.keys())
    for k in cell_keys:
        rng.shuffle(bins[k])

    # Sort each cell preferring balanced track coverage on pass 1
    total_dur = 0.0
    round_num = 0
    while len(selected_indices) < target_count and total_dur < target_sec:
        added_this_round = False
        for k in cell_keys:
            cell_list = bins[k]
            if round_num < len(cell_list):
                cand_idx = cell_list[round_num]
                if cand_idx not in used:
                    seg_dur = segments[cand_idx]["dur"] + 0.06  # incl pre/post pad
                    if total_dur + seg_dur <= target_sec + 15.0:
                        used.add(cand_idx)
                        selected_indices.append(cand_idx)
                        total_dur += seg_dur
                        added_this_round = True
                        if len(selected_indices) >= target_count or total_dur >= target_sec:
                            break
        if not added_this_round:
            break
        round_num += 1

    return [segments[i] for i in selected_indices]


def pack_sprites(selected_segments, tracks_stereo):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Shuffle selected segments across sprites so sprite_0 alone already spans the full 2D plane and all movements
    rng = np.random.default_rng(2026)
    order = rng.permutation(len(selected_segments))
    buckets = [[] for _ in range(NUM_SPRITES)]
    for pos, seg_i in enumerate(order):
        buckets[pos % NUM_SPRITES].append(selected_segments[seg_i])

    pre_roll = 0.015
    post_tail = 0.03
    silence_pad = 0.12
    silence_samples = int(round(silence_pad * SR_SPRITE))
    silence_block = np.zeros((silence_samples, 2), dtype=np.float32)

    # Target RMS for gentle gain compensation
    rms_vals = np.array([s["rms"] for s in selected_segments], dtype=np.float64)
    target_rms = float(np.median(rms_vals))

    sprite_paths = []
    corpus_segments = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for sprite_idx, bucket in enumerate(buckets):
            chunks = [silence_block]  # leading digital silence absorbs any MP3 encoder priming
            cursor_samples = silence_samples

            for seg in bucket:
                y_st = tracks_stereo[seg["track_idx"]]
                max_samples = len(y_st)

                s0 = max(0, int(round((seg["t_start"] - pre_roll) * SR_SPRITE)))
                s1 = min(max_samples, int(round((seg["t_end"] + post_tail) * SR_SPRITE)))
                if s1 <= s0 + 64:
                    continue

                clip = y_st[s0:s1].copy()
                # Subtle 3ms micro-fade at WAV boundaries to avoid discontinuities before MP3 encoding
                fade_len = min(int(0.003 * SR_SPRITE), len(clip) // 4)
                if fade_len > 1:
                    ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)[:, None]
                    clip[:fade_len] *= ramp
                    clip[-fade_len:] *= ramp[::-1]

                # Offset into the playable start (accounting for actual pre_roll extracted)
                actual_pre = (seg["t_start"] * SR_SPRITE - s0) / SR_SPRITE
                pre_trim = max(0.0, actual_pre - 0.004)
                clip_dur_sec = (s1 - s0) / SR_SPRITE
                play_start_sec = (cursor_samples / SR_SPRITE) + pre_trim
                play_dur_sec = min(clip_dur_sec - pre_trim, max(0.15, clip_dur_sec - pre_trim))

                gain_comp = float(np.clip((target_rms / max(seg["rms"], 1e-6)) ** 0.55, 0.45, 2.2))

                corpus_segments.append([
                    sprite_idx,
                    round(play_start_sec, 4),
                    round(play_dur_sec, 4),
                    round(float(seg["x"]), 4),
                    round(float(seg["y"]), 4),
                    round(gain_comp, 4),
                ])

                chunks.append(clip)
                chunks.append(silence_block)
                cursor_samples += len(clip) + silence_samples

            full_wav = np.concatenate(chunks, axis=0)
            wav_path = tmp / f"sprite_{sprite_idx}.wav"
            mp3_rel = f"data/sprite_{sprite_idx}.mp3"
            mp3_path = OUT_DIR / f"sprite_{sprite_idx}.mp3"
            sf.write(str(wav_path), full_wav, SR_SPRITE, subtype="PCM_16")

            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-i", str(wav_path),
                    "-c:a", "libmp3lame", "-b:a", "128k", "-ac", "2", "-ar", str(SR_SPRITE),
                    str(mp3_path),
                ],
                check=True,
            )
            sprite_paths.append(mp3_rel)

    return sprite_paths, corpus_segments


def build_formant_binary(tracks, total_duration: float) -> int:
    # Global broadband and per-band normalization references across active frames of all 9 movements
    all_bb = np.concatenate([td["env_broadband"] for td in tracks])
    active_mask = all_bb > 1e-5
    ref_bb = float(np.percentile(all_bb[active_mask], 95)) if np.any(active_mask) else 1e-3
    ref_bb = max(ref_bb, 1e-5)

    n_bands = len(FORMANT_CENTERS)
    ref_bands = np.zeros(n_bands, dtype=np.float64)
    for b in range(n_bands):
        all_b = np.concatenate([td["env_bands"][b] for td in tracks])
        rb = float(np.percentile(all_b[active_mask], 98)) if np.any(active_mask) else 1e-3
        ref_bands[b] = max(rb, 1e-5)

    # Resample each track onto exact 1/ENV_FPS timeline grid so there is zero cumulative drift
    chunks = []
    offset_sec = 0.0
    dt_orig = ENV_HOP / SR_ANALYSIS
    for idx, td in enumerate(tracks):
        dur = td["duration"]
        f_start = int(round(offset_sec * ENV_FPS))
        f_track_end = int(round((offset_sec + dur) * ENV_FPS))
        n_track_frames = max(1, f_track_end - f_start)

        t_orig = np.arange(len(td["env_broadband"]), dtype=np.float64) * dt_orig
        t_grid = np.arange(n_track_frames, dtype=np.float64) / ENV_FPS

        bb_interp = np.interp(t_grid, t_orig, td["env_broadband"])
        bb_norm = np.clip(bb_interp / ref_bb, 0.0, 1.0)
        gate = bb_norm >= 0.015

        frame_block = np.zeros((n_track_frames, 1 + n_bands), dtype=np.uint8)
        frame_block[:, 0] = np.clip(np.round(bb_norm * 255.0), 0, 255).astype(np.uint8)

        for b in range(n_bands):
            b_interp = np.interp(t_grid, t_orig, td["env_bands"][b])
            b_norm = np.clip(b_interp / ref_bands[b], 0.0, 1.0) ** 1.35
            b_norm = np.where(gate, b_norm, 0.0)
            frame_block[:, 1 + b] = np.clip(np.round(b_norm * 255.0), 0, 255).astype(np.uint8)

        chunks.append(frame_block)

        if idx < len(tracks) - 1:
            f_next_start = int(round((offset_sec + dur + INTER_TRACK_GAP) * ENV_FPS))
            n_gap = max(0, f_next_start - f_track_end)
            if n_gap > 0:
                chunks.append(np.zeros((n_gap, 1 + n_bands), dtype=np.uint8))
        offset_sec += dur + INTER_TRACK_GAP

    packed = np.vstack(chunks)
    total_frames = int(packed.shape[0])
    out_path = OUT_DIR / "formants.bin"
    out_path.write_bytes(packed.tobytes())

    active_rows = packed[packed[:, 0] > 4]
    print(f"\n--- Formant Envelope Diagnostics ---")
    print(f"formants.bin   : {out_path.stat().st_size / 1024:.1f} KB, {total_frames} frames ({total_frames / ENV_FPS:.2f}s @ {ENV_FPS} Hz)")
    if len(active_rows) > 0:
        bb_mean = float(np.mean(active_rows[:, 0]))
        bb_p95 = float(np.percentile(active_rows[:, 0], 95))
        print(f"  Broadband    : mean={bb_mean:.1f}, p95={bb_p95:.1f}")
        for b, fc in enumerate(FORMANT_CENTERS):
            col = active_rows[:, 1 + b]
            print(f"  Band {b:2d} ({fc:4d}Hz): mean={np.mean(col):5.1f}, p95={np.percentile(col, 95):5.1f}")
    return total_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeline-only", action="store_true", help="Only regenerate timeline.json and formants.bin using existing corpus/sprites")
    args = parser.parse_args()

    wav_files = sorted(RAW_DIR.glob("0[1-9].wav"))
    if len(wav_files) != 9:
        raise RuntimeError(f"Expected 9 WAV files in {RAW_DIR}, found {len(wav_files)}")

    print(f"Loading and analyzing {len(wav_files)} Feria Quinta movements...")
    tracks = []
    for idx, wf in enumerate(wav_files):
        print(f"  Analyzing {wf.name}...")
        tracks.append(
            compute_track_analysis(
                wf,
                analysis_only=args.timeline_only,
                trim_leading_silence=(idx == 0),
            )
        )

    all_notes = []
    all_phrase_ends = []
    total_fallbacks = 0
    offset = 0.0
    for idx, td in enumerate(tracks):
        notes, p_ends, fb = extract_notes_and_phrases(td, offset)
        all_notes.extend(notes)
        all_phrase_ends.extend(p_ends)
        total_fallbacks += fb
        offset += td["duration"] + INTER_TRACK_GAP

    total_duration = offset - INTER_TRACK_GAP

    phrase_durs = [pe - prev for prev, pe in zip([0.0] + all_phrase_ends[:-1], all_phrase_ends)]
    below_15 = sum(1 for d in phrase_durs if d < 15.0)
    above_40 = sum(1 for d in phrase_durs if d > 40.0)
    print(f"\n--- Timeline Diagnostics ---")
    print(f"Total duration : {total_duration:.2f}s ({total_duration/60:.2f} min)")
    print(f"Total notes    : {len(all_notes)} ({len(all_notes)/total_duration:.2f} notes/sec)")
    print(f"Total phrases  : {len(all_phrase_ends)} (fallbacks={total_fallbacks})")
    print(
        f"Phrase dur (s) : min={min(phrase_durs):.2f}, p25={np.percentile(phrase_durs, 25):.2f}, "
        f"median={np.median(phrase_durs):.2f}, p75={np.percentile(phrase_durs, 75):.2f}, max={max(phrase_durs):.2f}"
    )
    print(f"Phrase bounds  : <15s={below_15}, >40s={above_40}")
    print(f"First 15 durs  : {[round(d, 2) for d in phrase_durs[:15]]}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_env_frames = build_formant_binary(tracks, total_duration)

    timeline_json = {
        "duration": round(total_duration, 3),
        "notes": all_notes,
        "phraseEnds": all_phrase_ends,
        "envFps": ENV_FPS,
        "envBands": FORMANT_CENTERS,
        "envQ": FORMANT_Q,
        "envFrames": total_env_frames,
    }
    (OUT_DIR / "timeline.json").write_text(json.dumps(timeline_json, separators=(",", ":")))

    if args.timeline_only:
        print(f"\n--- Output Summary (timeline-only) ---")
        print(f"timeline.json: {len(all_notes)} notes, {len(all_phrase_ends)} phraseEnds, {(OUT_DIR / 'timeline.json').stat().st_size / 1024:.1f} KB")
        return

    # Segment syllables across all 9 tracks
    all_segments = []
    for idx, td in enumerate(tracks):
        segs = segment_syllables(idx, td)
        all_segments.extend(segs)
        print(f"  Track {idx+1:02d}: {len(segs)} syllable segments")

    print(f"Total valid syllable segments: {len(all_segments)}")

    # Compute 2D Feature Space: X = MFCC, Y = Spectral Centroid
    log_cents = np.array([s["log_centroid"] for s in all_segments], dtype=np.float64)
    mfcc_mat = np.vstack([s["mfcc"] for s in all_segments])

    y_coords = winsorize_and_map(log_cents, 2.0, 98.0, 0.02, 0.98)

    mfcc_scaled = StandardScaler().fit_transform(mfcc_mat)
    pc1_raw = PCA(n_components=1).fit_transform(mfcc_scaled)[:, 0]
    r_raw = float(np.corrcoef(pc1_raw, log_cents)[0, 1])

    if abs(r_raw) > 0.55:
        # Project out linear log_centroid component from standardized MFCC[1:13]
        c_z = zscore(log_cents)[:, None]
        denom = float(np.sum(c_z ** 2))
        if denom > 1e-8:
            proj = (c_z.T @ mfcc_scaled) / denom
            mfcc_orth = mfcc_scaled - c_z @ proj
        else:
            mfcc_orth = mfcc_scaled
        pc1 = PCA(n_components=1).fit_transform(mfcc_orth)[:, 0]
        print(f"  Raw MFCC PC1 corr with log_centroid was r={r_raw:.3f} (>0.55); orthogonalized MFCC space.")
    else:
        pc1 = pc1_raw

    x_coords = winsorize_and_map(pc1, 2.0, 98.0, 0.02, 0.98)
    r_final = float(np.corrcoef(x_coords, y_coords)[0, 1])

    for i, s in enumerate(all_segments):
        s["x"] = float(x_coords[i])
        s["y"] = float(y_coords[i])

    # Grid occupancy diagnostics (10x10)
    grid = np.zeros((10, 10), dtype=int)
    for s in all_segments:
        gx = min(9, max(0, int(s["x"] * 10)))
        gy = min(9, max(0, int(s["y"] * 10)))
        grid[gx, gy] += 1
    occupied_bins = int(np.sum(grid > 0))
    print(f"Feature space Pearson r(x, y) = {r_final:.4f}; 10x10 occupied bins = {occupied_bins}/100")

    # Select stratified corpus and pack sprites
    selected = select_stratified_corpus(all_segments, TARGET_SEGMENTS, TARGET_PACKED_SEC)
    print(f"Selected {len(selected)} stratified segments (~{sum(s['dur'] for s in selected):.1f}s pure audio)")

    tracks_stereo = [td["y_st"] for td in tracks]
    sprite_paths, corpus_segments = pack_sprites(selected, tracks_stereo)

    corpus_json = {
        "sprites": sprite_paths,
        "segments": corpus_segments,
    }
    timeline_json = {
        "duration": round(total_duration, 3),
        "notes": all_notes,
        "phraseEnds": all_phrase_ends,
    }

    (OUT_DIR / "corpus.json").write_text(json.dumps(corpus_json, separators=(",", ":")))
    (OUT_DIR / "timeline.json").write_text(json.dumps(timeline_json, separators=(",", ":")))

    total_mp3_bytes = sum((OUT_DIR / f"sprite_{i}.mp3").stat().st_size for i in range(NUM_SPRITES))
    print(f"\n--- Output Summary ---")
    print(f"corpus.json  : {len(corpus_segments)} segments, {(OUT_DIR / 'corpus.json').stat().st_size / 1024:.1f} KB")
    print(f"timeline.json: {len(all_notes)} notes, {len(all_phrase_ends)} phraseEnds, {(OUT_DIR / 'timeline.json').stat().st_size / 1024:.1f} KB")
    print(f"5 MP3 sprites: {total_mp3_bytes / (1024 * 1024):.2f} MB total")


if __name__ == "__main__":
    main()
