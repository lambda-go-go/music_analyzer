import os, tkinter as tk
from tkinter import filedialog as fd, messagebox as mb
import numpy as np
from audio_engine import AudioEngine
from analyzer import detect_bpm, detect_key, detect_chords_and_measures, detect_song_form

class MusicPracticeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("음악 연습 보조 프로그램")
        self.root.geometry("640x740")
        self.engine = AudioEngine()
        self.measures, self.song_forms = [], []
        self.loop_active, self.has_segmented = False, False
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(50, self._update_ui)

    def _build_ui(self):
        f_top = tk.Frame(self.root, pady=4); f_top.pack(fill="x", padx=10)
        tk.Button(f_top, text="음악 파일 선택 (MP3/MP4)", font=("맑은 고딕", 10, "bold"), command=self.load_file).pack(side="left")
        self.lbl_file = tk.Label(f_top, text="선택된 파일 없음", anchor="w", fg="gray"); self.lbl_file.pack(side="left", padx=8, fill="x")

        f_info = tk.LabelFrame(self.root, text="음악 정보 (BPM / Key)", font=("맑은 고딕", 9, "bold"), padx=10, pady=5)
        f_info.pack(fill="x", padx=10, pady=4)
        self.lbl_bpm = tk.Label(f_info, text="BPM: --", font=("맑은 고딕", 11, "bold"), fg="blue"); self.lbl_bpm.pack(side="left", expand=True)
        self.lbl_key = tk.Label(f_info, text="Key: --", font=("맑은 고딕", 11, "bold"), fg="darkgreen"); self.lbl_key.pack(side="left", expand=True)

        f_pb = tk.LabelFrame(self.root, text="재생 컨트롤", font=("맑은 고딕", 9, "bold"), padx=10, pady=5); f_pb.pack(fill="x", padx=10, pady=4)
        f_btn = tk.Frame(f_pb); f_btn.pack(fill="x")
        self.btn_play = tk.Button(f_btn, text="재생 ▶", width=10, command=self.toggle_play); self.btn_play.pack(side="left")
        self.lbl_time = tk.Label(f_btn, text="00:00 / 00:00", font=("맑은 고딕", 9)); self.lbl_time.pack(side="right")
        self.slider_pos = tk.Scale(f_pb, from_=0, to=100, orient="horizontal", showvalue=0, command=self._on_seek); self.slider_pos.pack(fill="x", pady=2)

        f_spd = tk.LabelFrame(self.root, text="재생속도 설정 (0.1 ~ 2.0)", font=("맑은 고딕", 9, "bold"), padx=10, pady=5); f_spd.pack(fill="x", padx=10, pady=4)
        self.slider_spd = tk.Scale(f_spd, from_=0.1, to=2.0, resolution=0.1, orient="horizontal", command=self._on_speed_slider)
        self.slider_spd.set(1.0); self.slider_spd.pack(side="left", fill="x", expand=True)
        self.ent_spd = tk.Entry(f_spd, width=5, font=("맑은 고딕", 10)); self.ent_spd.insert(0, "1.0"); self.ent_spd.pack(side="left", padx=6)
        tk.Button(f_spd, text="입력", command=self._apply_speed_entry).pack(side="left")

        f_loop = tk.LabelFrame(self.root, text="반복구간 설정", font=("맑은 고딕", 9, "bold"), padx=10, pady=5); f_loop.pack(fill="x", padx=10, pady=4)
        f_lbtn = tk.Frame(f_loop); f_lbtn.pack(fill="x")
        self.btn_loop = tk.Button(f_lbtn, text="반복구간 켜기", command=self.toggle_loop); self.btn_loop.pack(side="left")
        self.lbl_loop_info = tk.Label(f_lbtn, text="구간: 00:00 ~ 00:00", fg="gray"); self.lbl_loop_info.pack(side="right")
        tk.Label(f_loop, text="시작점 슬라이더").pack(anchor="w")
        self.scale_lstart = tk.Scale(f_loop, from_=0, to=100, resolution=0.1, orient="horizontal", command=lambda v: self._on_loop_change(True)); self.scale_lstart.pack(fill="x")
        tk.Label(f_loop, text="마지막점 슬라이더").pack(anchor="w")
        self.scale_lend = tk.Scale(f_loop, from_=0, to=100, resolution=0.1, orient="horizontal", command=lambda v: self._on_loop_change(False)); self.scale_lend.pack(fill="x")

        f_sf = tk.LabelFrame(self.root, text="송폼 분리", font=("맑은 고딕", 9, "bold"), padx=10, pady=5); f_sf.pack(fill="x", padx=10, pady=4)
        tk.Button(f_sf, text="송폼 분리 실행", command=self.run_song_form).pack(anchor="w", pady=2)
        self.lst_forms = tk.Listbox(f_sf, height=4, font=("맑은 고딕", 9)); self.lst_forms.pack(fill="x", pady=2)
        self.lst_forms.bind("<<ListboxSelect>>", self._on_select_form)

        f_ch = tk.LabelFrame(self.root, text="실시간 코드 & 마디 표시", font=("맑은 고딕", 9, "bold"), padx=10, pady=5); f_ch.pack(fill="x", padx=10, pady=4)
        self.lbl_measure = tk.Label(f_ch, text="마디: -", font=("맑은 고딕", 10)); self.lbl_measure.pack(side="left", padx=10)
        self.lbl_chord = tk.Label(f_ch, text="현재 코드: -", font=("맑은 고딕", 12, "bold"), fg="crimson"); self.lbl_chord.pack(side="left", padx=20)
        self.lbl_prog = tk.Label(f_ch, text="마디 코드: -", font=("맑은 고딕", 10), fg="purple"); self.lbl_prog.pack(side="left", padx=10)

    def _fmt(self, sec):
        return f"{int(sec)//60:02d}:{int(sec)%60:02d}"

    def load_file(self):
        p = fd.askopenfilename(filetypes=[("음악/비디오 파일", "*.mp3 *.mp4"), ("모든 파일", "*.*")])
        if not p: return
        self.lbl_file.config(text=os.path.basename(p), fg="black")
        self.engine.load_file(p)
        dur = self.engine.duration
        self.slider_pos.config(to=dur)
        self.scale_lstart.config(to=dur); self.scale_lstart.set(0)
        self.scale_lend.config(to=dur); self.scale_lend.set(dur)
        mono, sr = np.mean(self.engine.data, axis=1), self.engine.sr
        bpm, key = detect_bpm(mono, sr), detect_key(mono, sr)
        self.lbl_bpm.config(text=f"BPM: {bpm}"); self.lbl_key.config(text=f"Key: {key}")
        self.measures = detect_chords_and_measures(mono, sr, bpm)
        self.song_forms, self.has_segmented = [], False
        self.lst_forms.delete(0, "end")

    def toggle_play(self):
        if self.engine.is_playing:
            self.engine.pause(); self.btn_play.config(text="재생 ▶")
        else:
            self.engine.play(); self.btn_play.config(text="일시정지 ❚❚")

    def _on_seek(self, val):
        if not self.engine.is_playing: self.engine.seek(float(val))

    def _on_speed_slider(self, val):
        spd = round(float(val), 1)
        self.engine.set_speed(spd)
        self.ent_spd.delete(0, "end"); self.ent_spd.insert(0, str(spd))

    def _apply_speed_entry(self):
        try:
            val = max(0.1, min(2.0, round(float(self.ent_spd.get()), 1)))
            self.slider_spd.set(val); self.engine.set_speed(val)
        except ValueError: pass

    def toggle_loop(self):
        self.loop_active = not self.loop_active
        st, en = float(self.scale_lstart.get()), float(self.scale_lend.get())
        self.engine.set_loop(st, en, self.loop_active)
        self.btn_loop.config(text="반복구간 끄기" if self.loop_active else "반복구간 켜기")

    def _on_loop_change(self, is_start):
        st, en = float(self.scale_lstart.get()), float(self.scale_lend.get())
        if st > en:
            if is_start: self.scale_lend.set(st); en = st
            else: self.scale_lstart.set(en); st = en
        self.lbl_loop_info.config(text=f"구간: {self._fmt(st)} ~ {self._fmt(en)}")
        self.engine.set_loop(st, en, self.loop_active)
        self.engine.seek(st if is_start else max(st, en - 1.0))
        self.engine.play(); self.btn_play.config(text="일시정지 ❚❚")

    def run_song_form(self):
        if self.engine.data is None: return
        mono = np.mean(self.engine.data, axis=1)
        txt = self.lbl_bpm.cget("text")
        bpm = float(txt.split()[1]) if "BPM: --" not in txt else 120.0
        self.song_forms = detect_song_form(mono, self.engine.sr, bpm)
        self.has_segmented = True
        self.lst_forms.delete(0, "end")
        for s in self.song_forms:
            self.lst_forms.insert("end", f"{s['name']}: {self._fmt(s['start'])} ~ {self._fmt(s['end'])}")

    def _on_select_form(self, evt):
        sel = self.lst_forms.curselection()
        if not sel or sel[0] >= len(self.song_forms): return
        sec = self.song_forms[sel[0]]
        st, en = sec["start"], sec["end"]
        self.scale_lstart.set(st); self.scale_lend.set(en)
        self.loop_active = True
        self.btn_loop.config(text="반복구간 끄기")
        self.engine.set_loop(st, en, True)
        self.engine.seek(st); self.engine.play(); self.btn_play.config(text="일시정지 ❚❚")

    def _update_ui(self):
        pos, dur = self.engine.get_position(), self.engine.duration
        self.lbl_time.config(text=f"{self._fmt(pos)} / {self._fmt(dur)}")
        if self.engine.is_playing:
            self.slider_pos.set(pos)
            self.btn_play.config(text="일시정지 ❚❚")
        elif self.btn_play.cget("text") == "일시정지 ❚❚":
            self.btn_play.config(text="재생 ▶")

        if self.measures:
            c_bar, c_chord, c_prog = "-", "-", "-"
            for m in self.measures:
                if m["start"] <= pos < m["end"]:
                    c_bar, c_prog = f"{m['bar']} 마디", m["summary"]
                    for s_st, s_en, c in m["subs"]:
                        if s_st <= pos < s_en and c:
                            c_chord = c; break
                    break
            self.lbl_measure.config(text=f"마디: {c_bar}")
            self.lbl_chord.config(text=f"현재 코드: {c_chord}")
            self.lbl_prog.config(text=f"마디 코드: {c_prog}")
        self.root.after(50, self._update_ui)

    def _on_close(self):
        if self.loop_active or self.has_segmented:
            ans = mb.askyesnocancel("구간 저장 확인", "설정한 반복 구간 또는 송폼 구간을 mp3 파일로 저장하시겠습니까?")
            if ans is None: return
            if ans:
                sp = fd.asksaveasfilename(defaultextension=".mp3", filetypes=[("MP3 파일", "*.mp3")], title="구간 저장")
                if sp:
                    st, en = float(self.scale_lstart.get()), float(self.scale_lend.get())
                    self.engine.save_segment(st, en, sp)
                    mb.showinfo("저장 완료", f"성공적으로 저장되었습니다:\n{sp}")
        self.engine.close()
        self.root.destroy()

if __name__ == "__main__":
    tk_root = tk.Tk()
    app = MusicPracticeApp(tk_root)
    tk_root.mainloop()
