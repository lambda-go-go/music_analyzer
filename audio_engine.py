import os
import subprocess
import tempfile
import numpy as np
import soundfile as sf
import sounddevice as sd
import imageio_ffmpeg

class AudioEngine:
    def __init__(self):
        self.data = None
        self.sr = 44100
        self.duration = 0.0
        self.pos_sample = 0.0
        self.speed = 1.0
        self.is_playing = False
        self.loop_enabled = False
        self.loop_start = 0.0
        self.loop_end = 0.0
        self.stream = None
        self._temp_files = []

    def load_file(self, filepath):
        self.pause()
        ext = os.path.splitext(filepath)[1].lower()
        load_path = filepath
        if ext == ".mp4":
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp.close()
            self._temp_files.append(tmp.name)
            cmd = [
                imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", filepath,
                "-vn", "-acodec", "libmp3lame", "-q:a", "2", tmp.name
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            load_path = tmp.name

        data, sr = sf.read(load_path, dtype="float32")
        if data.ndim == 1:
            data = np.column_stack([data, data])
        elif data.shape[1] > 2:
            data = data[:, :2]

        self.data = data
        self.sr = sr
        self.duration = float(len(data)) / float(sr)
        self.pos_sample = 0.0
        self.loop_start = 0.0
        self.loop_end = self.duration
        self.loop_enabled = False

        if self.stream is not None:
            self.stream.stop()
            self.stream.close()

        self.stream = sd.OutputStream(
            samplerate=self.sr,
            channels=2,
            callback=self._audio_callback,
            blocksize=2048
        )
        self.stream.start()

    def _audio_callback(self, outdata, frames, time_info, status):
        if not self.is_playing or self.data is None:
            outdata.fill(0)
            return

        total_samples = len(self.data)
        out_indices = self.pos_sample + np.arange(frames, dtype=np.float64) * self.speed
        
        if self.loop_enabled:
            s_idx = max(0, int(self.loop_start * self.sr))
            e_idx = min(total_samples, max(s_idx + 1, int(self.loop_end * self.sr)))
            span = max(1, e_idx - s_idx)
            wrapped = s_idx + np.mod(out_indices - s_idx, span)
            idx0 = np.clip(np.floor(wrapped).astype(int), 0, total_samples - 1)
            idx1 = np.clip(idx0 + 1, 0, total_samples - 1)
            frac = (wrapped - idx0)[:, None]
            outdata[:] = self.data[idx0] * (1.0 - frac) + self.data[idx1] * frac
            self.pos_sample = s_idx + np.mod(self.pos_sample + frames * self.speed - s_idx, span)
        else:
            valid_mask = (out_indices >= 0) & (out_indices < total_samples - 1)
            valid_len = int(np.sum(valid_mask))
            if valid_len > 0:
                v_idx = out_indices[valid_mask]
                idx0 = np.floor(v_idx).astype(int)
                idx1 = idx0 + 1
                frac = (v_idx - idx0)[:, None]
                outdata[:valid_len] = self.data[idx0] * (1.0 - frac) + self.data[idx1] * frac
            if valid_len < frames:
                outdata[valid_len:].fill(0)
                self.is_playing = False
            self.pos_sample += frames * self.speed
            if self.pos_sample >= total_samples:
                self.pos_sample = total_samples
                self.is_playing = False

    def play(self):
        if self.data is not None:
            if not self.loop_enabled and self.pos_sample >= len(self.data):
                self.pos_sample = 0.0
            self.is_playing = True

    def pause(self):
        self.is_playing = False

    def seek(self, sec):
        if self.data is not None:
            clamped_sec = max(0.0, min(self.duration, float(sec)))
            self.pos_sample = clamped_sec * self.sr

    def get_position(self):
        if self.data is None or self.sr == 0:
            return 0.0
        return self.pos_sample / self.sr

    def set_speed(self, speed):
        self.speed = max(0.1, min(2.0, float(speed)))

    def set_loop(self, start_sec, end_sec, enabled=True):
        self.loop_start = max(0.0, min(self.duration, float(start_sec)))
        self.loop_end = max(self.loop_start, min(self.duration, float(end_sec)))
        self.loop_enabled = enabled

    def save_segment(self, start_sec, end_sec, out_path):
        if self.data is None:
            return False
        s_idx = max(0, int(start_sec * self.sr))
        e_idx = min(len(self.data), int(end_sec * self.sr))
        if e_idx <= s_idx:
            return False
        segment = self.data[s_idx:e_idx]
        sf.write(out_path, segment, self.sr, format="MP3")
        return True

    def close(self):
        self.is_playing = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        for tmp in self._temp_files:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
