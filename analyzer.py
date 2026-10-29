import numpy as np
from scipy.signal import stft

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

CHORD_TEMPLATES = {}
for i, name in enumerate(NOTES):
    maj = np.zeros(12)
    maj[[i, (i + 4) % 12, (i + 7) % 12]] = 1.0
    CHORD_TEMPLATES[name] = maj / np.linalg.norm(maj)
    min_t = np.zeros(12)
    min_t[[i, (i + 3) % 12, (i + 7) % 12]] = 1.0
    CHORD_TEMPLATES[name + "m"] = min_t / np.linalg.norm(min_t)

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
MAJOR_PROFILE = (MAJOR_PROFILE - MAJOR_PROFILE.mean()) / MAJOR_PROFILE.std()
MINOR_PROFILE = (MINOR_PROFILE - MINOR_PROFILE.mean()) / MINOR_PROFILE.std()

def _extract_chroma(y, sr, nperseg=4096):
    if len(y) < nperseg:
        return np.zeros(12)
    f, _, zxx = stft(y, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
    mag = np.abs(zxx)
    chroma = np.zeros(12)
    valid_bins = np.where((f >= 65.0) & (f <= 2100.0))[0]
    for b in valid_bins:
        pc = int(round(12.0 * np.log2(f[b] / 440.0) + 69.0)) % 12
        chroma[pc] += np.sum(mag[b])
    norm = np.linalg.norm(chroma)
    return chroma / norm if norm > 1e-4 else np.zeros(12)

def detect_bpm(audio_mono, sr):
    hop = 512
    nperseg = 1024
    if len(audio_mono) < nperseg * 2:
        return 120.0
    _, _, zxx = stft(audio_mono, fs=sr, nperseg=nperseg, noverlap=nperseg - hop)
    diff = np.maximum(0, np.diff(np.abs(zxx), axis=1))
    onset = np.sum(diff, axis=0)
    onset -= np.mean(onset)
    corr = np.correlate(onset, onset, mode="full")[len(onset) - 1:]
    fps = float(sr) / float(hop)
    min_lag = max(1, int(fps * 60.0 / 210.0))
    max_lag = min(len(corr) - 1, int(fps * 60.0 / 55.0))
    if max_lag <= min_lag:
        return 120.0
    best_lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
    if 0 < best_lag < len(corr) - 1:
        y0, ym1, yp1 = corr[best_lag], corr[best_lag - 1], corr[best_lag + 1]
        denom = 2.0 * (2.0 * y0 - ym1 - yp1)
        if abs(denom) > 1e-6:
            best_lag += (yp1 - ym1) / denom
    bpm = round((fps * 60.0) / float(best_lag), 1)
    return max(60.0, min(200.0, bpm))

def detect_key(audio_mono, sr):
    step = sr * 3
    chromas = []
    for i in range(0, len(audio_mono), step):
        chunk = audio_mono[i:i + step]
        if len(chunk) >= 2048:
            chromas.append(_extract_chroma(chunk, sr, nperseg=2048))
    if not chromas:
        return "C Major"
    avg_chroma = np.mean(chromas, axis=0)
    norm = np.linalg.norm(avg_chroma)
    if norm < 1e-4:
        return "C Major"
    avg_chroma /= norm
    best_score, best_key = -999.0, "C Major"
    for i in range(12):
        shifted = np.roll(avg_chroma, -i)
        s_std = shifted.std()
        shifted = (shifted - shifted.mean()) / (s_std if s_std > 1e-4 else 1.0)
        score_maj = float(np.dot(shifted, MAJOR_PROFILE))
        score_min = float(np.dot(shifted, MINOR_PROFILE))
        if score_maj > best_score:
            best_score, best_key = score_maj, f"{NOTES[i]} Major"
        if score_min > best_score:
            best_score, best_key = score_min, f"{NOTES[i]} Minor"
    return best_key

def _match_chord(chroma, rms):
    if rms < 0.005 or np.linalg.norm(chroma) < 1e-4:
        return ""
    best_sim, best_chord = -1.0, ""
    for name, tmpl in CHORD_TEMPLATES.items():
        sim = float(np.dot(chroma, tmpl))
        if sim > best_sim:
            best_sim, best_chord = sim, name
    return best_chord if best_sim >= 0.70 else ""

def detect_chords_and_measures(audio_mono, sr, bpm):
    beat_dur = 60.0 / bpm
    bar_dur = beat_dur * 4.0
    total_dur = len(audio_mono) / float(sr)
    num_bars = max(1, int(np.ceil(total_dur / bar_dur)))
    measures = []

    for b in range(num_bars):
        b_start = b * bar_dur
        b_end = min(total_dur, (b + 1) * bar_dur)
        half_dur = (b_end - b_start) / 2.0
        
        # Sub-intervals: first half & second half
        sub_chords = []
        for h in range(2):
            s_t = b_start + h * half_dur
            e_t = min(total_dur, s_t + half_dur)
            s_idx = int(s_t * sr)
            e_idx = int(e_t * sr)
            chunk = audio_mono[s_idx:e_idx]
            rms = np.sqrt(np.mean(chunk ** 2)) if len(chunk) > 0 else 0.0
            chroma = _extract_chroma(chunk, sr, nperseg=min(2048, max(512, len(chunk))))
            c_name = _match_chord(chroma, rms)
            sub_chords.append((s_t, e_t, c_name))
        
        c1, c2 = sub_chords[0][2], sub_chords[1][2]
        if c1 and c2 and c1 != c2:
            summary = f"{c1} → {c2}"
        elif c1 or c2:
            summary = c1 or c2
        else:
            summary = "-"
        measures.append({"bar": b + 1, "start": b_start, "end": b_end, "subs": sub_chords, "summary": summary})
    return measures

def detect_song_form(audio_mono, sr, bpm):
    beat_dur = 60.0 / bpm
    bar_dur = beat_dur * 4.0
    sec_bars = 4 if bpm < 100 else 8
    sec_dur = bar_dur * sec_bars
    total_dur = len(audio_mono) / float(sr)
    n_chunks = max(1, int(np.ceil(total_dur / sec_dur)))

    feats = []
    times = []
    for i in range(n_chunks):
        st_t = i * sec_dur
        en_t = min(total_dur, (i + 1) * sec_dur)
        chunk = audio_mono[int(st_t * sr):int(en_t * sr)]
        c = _extract_chroma(chunk, sr, nperseg=min(4096, max(512, len(chunk))))
        rms = np.sqrt(np.mean(chunk ** 2)) if len(chunk) > 0 else 0.0
        vec = np.append(c, rms * 10.0)
        norm = np.linalg.norm(vec)
        feats.append(vec / norm if norm > 1e-4 else np.zeros(13))
        times.append((st_t, en_t))

    clusters = []
    raw_labels = []
    for f in feats:
        matched = False
        for c_idx, c_vec in enumerate(clusters):
            if np.dot(f, c_vec) > 0.85:
                raw_labels.append(chr(65 + c_idx))
                matched = True
                break
        if not matched:
            raw_labels.append(chr(65 + len(clusters)))
            clusters.append(f)

    # Merge consecutive identical sections
    merged = []
    cur_lbl = raw_labels[0]
    cur_st = times[0][0]
    cur_en = times[0][1]
    for lbl, (st, en) in zip(raw_labels[1:], times[1:]):
        if lbl == cur_lbl:
            cur_en = en
        else:
            merged.append({"name": f"구간 {cur_lbl}", "start": cur_st, "end": cur_en})
            cur_lbl = lbl
            cur_st = st
            cur_en = en
    merged.append({"name": f"구간 {cur_lbl}", "start": cur_st, "end": cur_en})
    return merged
