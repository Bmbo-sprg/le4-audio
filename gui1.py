from enum import Enum
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import threading
import time
import tkinter as tk
import tkinter.filedialog
import ttkbootstrap as ttk
import ttkbootstrap.constants as ttk_const
import pyaudio
import scipy.io.wavfile
import wave as pywave

from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils import (
    Note,
    CHORDS_L,
    wave2specgram,
    extract_f0,
    frame2spec,
    frame2vol,
    spec2chroma,
)


class PlayStatus(Enum):
    STOP = 0
    PLAY = 1
    PAUSE = 2


class AudioVisualizer(ttk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.filepath: str = ""
        self.wavefile = None
        self.sr: int = None
        self.fft_size: int = None
        self.fft_shift: int = None
        self.vol_threshold: int = None
        self.vc_freq: float = None
        self.vc_depth: float = None
        self.tremolo_freq: float = None
        self.tremolo_depth: float = None
        self.wave: np.ndarray = None
        self.spec: np.ndarray = None
        self.specgram: np.ndarray = None
        self.img_wave = None
        self.img_specgram = None
        self.img_spec = None
        self.f0_out: np.ndarray = None
        self.vol_in: float = None
        self.f0_in: np.ndarray = None
        self.chroma_in: int = None
        self.chordgram: np.ndarray = None
        self.p_out = pyaudio.PyAudio()
        self.p_in = pyaudio.PyAudio()
        self.play_status: PlayStatus = PlayStatus.STOP
        self.play_ms: int = 0
        self.stacked_input = np.array([])

        self.WAVE_RES_X = 10000
        self.SPECGRAM_RES_X = 100
        self.SPEC_RES_X = 1000
        self.P_OUT_CHUNK = 1024

        self.master = master
        self.pack()

        self.lframe = ttk.Frame(self, borderwidth=2)
        self.lframe.pack(side=tk.LEFT)

        self.fft_size_frame = ttk.Frame(self.lframe, padding=(0, 20, 0, 10))
        self.fft_size_frame.pack(side=tk.TOP)
        self.fft_size_label = ttk.Label(self.fft_size_frame, text='FFT size')
        self.fft_size_label.pack(side=tk.LEFT)
        self.fft_size_entry = ttk.Combobox(
            self.fft_size_frame, values=[2 ** i for i in range(9, 15)])
        self.fft_size_entry.set('4096')
        self.fft_size_entry.pack(side=tk.LEFT, anchor=tk.W)

        self.fft_shift_frame = ttk.Frame(self.lframe, padding=(0, 10))
        self.fft_shift_frame.pack(side=tk.TOP)
        self.fft_shift_label = ttk.Label(self.fft_shift_frame, text='FFT shift')
        self.fft_shift_label.pack(side=tk.LEFT)
        self.fft_shift_entry = ttk.Entry(self.fft_shift_frame)
        self.fft_shift_entry.insert(tk.END, '1280')
        self.fft_shift_entry.pack(side=tk.LEFT, anchor=tk.W)

        self.vol_threshold_frame = ttk.Frame(self.lframe, padding=(0, 10))
        self.vol_threshold_frame.pack(side=tk.TOP)
        self.vol_threshold_label = ttk.Label(self.vol_threshold_frame, text='Volume threshold')
        self.vol_threshold_label.pack(side=tk.LEFT)
        self.vol_threshold_scale = ttk.Scale(
            self.vol_threshold_frame,
            from_=-60,
            to=0,
            value=-30,
            length=200,
            orient=tk.HORIZONTAL,
        )
        self.vol_threshold_scale.pack(side=tk.LEFT, anchor=tk.W)

        self.open_file_button = ttk.Button(
            self.lframe,
            text='Open file',
            bootstyle=ttk_const.PRIMARY,
            command=self.open_file,
        )
        self.open_file_button.pack(side=tk.TOP)

        self.filename_label = ttk.Label(self.lframe, text='')
        self.filename_label.pack(side=tk.TOP)
        self.duration_label = ttk.Label(self.lframe, text='')
        self.duration_label.pack(side=tk.TOP)
        self.sr_label = ttk.Label(self.lframe, text='')
        self.sr_label.pack(side=tk.TOP)

        self.vc_freq_frame = ttk.Frame(self.lframe, padding=(0, 10))
        self.vc_freq_frame.pack(side=tk.TOP)
        self.vc_freq_label = ttk.Label(self.vc_freq_frame, text='Voice Change frequency')
        self.vc_freq_label.pack(side=tk.LEFT)
        self.vc_freq_scale = ttk.Scale(
            self.vc_freq_frame,
            from_=0,
            to=30,
            value=0,
            length=200,
            orient=tk.HORIZONTAL,
        )
        self.vc_freq_scale.pack(side=tk.LEFT, anchor=tk.W)

        self.vc_depth_frame = ttk.Frame(self.lframe, padding=(0, 10))
        self.vc_depth_frame.pack(side=tk.TOP)
        self.vc_depth_label = ttk.Label(self.vc_depth_frame, text='Voice Change depth')
        self.vc_depth_label.pack(side=tk.LEFT)
        self.vc_depth_scale = ttk.Scale(
            self.vc_depth_frame,
            from_=0,
            to=100,
            value=0,
            length=200,
            orient=tk.HORIZONTAL,
        )
        self.vc_depth_scale.pack(side=tk.LEFT, anchor=tk.W)

        self.vc_button = ttk.Button(
            self.lframe,
            text='Apply Voice Change',
            bootstyle=ttk_const.PRIMARY,
            command=self.apply_vc,
        )
        self.vc_button.pack(side=tk.TOP)

        self.tremolo_freq_frame = ttk.Frame(self.lframe, padding=(0, 10))
        self.tremolo_freq_frame.pack(side=tk.TOP)
        self.tremolo_freq_label = ttk.Label(self.tremolo_freq_frame, text='Tremolo frequency')
        self.tremolo_freq_label.pack(side=tk.LEFT)
        self.tremolo_freq_scale = ttk.Scale(
            self.tremolo_freq_frame,
            from_=0,
            to=30,
            value=0,
            length=200,
            orient=tk.HORIZONTAL,
        )
        self.tremolo_freq_scale.pack(side=tk.LEFT, anchor=tk.W)

        self.tremolo_depth_frame = ttk.Frame(self.lframe, padding=(0, 10))
        self.tremolo_depth_frame.pack(side=tk.TOP)
        self.tremolo_depth_label = ttk.Label(self.tremolo_depth_frame, text='Tremolo depth')
        self.tremolo_depth_label.pack(side=tk.LEFT)
        self.tremolo_depth_scale = ttk.Scale(
            self.tremolo_depth_frame,
            from_=0,
            to=100,
            value=0,
            length=200,
            orient=tk.HORIZONTAL,
        )
        self.tremolo_depth_scale.pack(side=tk.LEFT, anchor=tk.W)

        self.tremolo_button = ttk.Button(
            self.lframe,
            text='Apply Tremolo',
            bootstyle=ttk_const.PRIMARY,
            command=self.apply_tremolo,
        )
        self.tremolo_button.pack(side=tk.TOP)

        self.quit_button = ttk.Button(
            self.lframe,
            text='Quit app',
            bootstyle=ttk_const.DANGER,
            command=self.quit,
        )
        self.quit_button.pack(side=tk.TOP)

        self.mframe = ttk.Frame(self, borderwidth=2)
        self.mframe.pack(side=tk.LEFT)

        self.frame_wave = ttk.Frame(self.mframe)
        self.frame_wave.pack(side=tk.TOP)
        self.fig_wave = plt.figure(figsize=(10, 3))
        self.fig_wave.subplots_adjust(0.05, 0.01, 0.95, 0.99)
        self.canvas_wave = FigureCanvasTkAgg(self.fig_wave, master=self.frame_wave)
        self.canvas_wave.get_tk_widget().pack(side=tk.TOP)

        self.frame_specgram = ttk.Frame(self.mframe)
        self.frame_specgram.pack(side=tk.TOP)
        self.fig_specgram = plt.figure(figsize=(10, 5))
        self.fig_specgram.subplots_adjust(0.05, 0.05, 0.95, 0.99)
        self.canvas_specgram = FigureCanvasTkAgg(self.fig_specgram, master=self.frame_specgram)
        self.canvas_specgram.get_tk_widget().pack(side=tk.TOP)
        self.animation_wave = FuncAnimation(
            self.fig_wave,
            self.update_img_wave,
            interval=500,
            blit=True,
        )
        self.animation_specgram = FuncAnimation(
            self.fig_specgram,
            self.update_img_specgram,
            interval=250,
            blit=True,
        )

        self.ctrl_frame = ttk.Frame(self.mframe)
        self.ctrl_frame.pack(side=tk.TOP)
        self.time_label = ttk.Label(self.ctrl_frame, text=self._format_time(0))
        self.time_label.pack(side=tk.LEFT)
        self.play_button = ttk.Button(
            self.ctrl_frame,
            text='Play',
            bootstyle=ttk_const.SUCCESS,
            command=self.play,
        )
        self.play_button.pack(side=tk.LEFT)
        self.pause_button = ttk.Button(
            self.ctrl_frame,
            text='Pause',
            bootstyle=ttk_const.SECONDARY,
            command=self.pause,
        )
        self.pause_button.pack(side=tk.LEFT)
        self.stop_button = ttk.Button(
            self.ctrl_frame,
            text='Stop',
            bootstyle=ttk_const.DANGER,
            command=self.stop,
        )
        self.stop_button.pack(side=tk.LEFT)

        self.rframe = ttk.Frame(self, borderwidth=2)
        self.rframe.pack(side=tk.RIGHT)

        self.frame_spec = ttk.Frame(self.rframe)
        self.frame_spec.pack(side=tk.TOP)
        self.fig_spec = plt.figure(figsize=(5, 5))
        self.canvas_spec = FigureCanvasTkAgg(self.fig_spec, master=self.frame_spec)
        self.canvas_spec.get_tk_widget().pack(side=tk.TOP)
        self.animation_spec = FuncAnimation(
            self.fig_spec,
            self.update_img_spec,
            interval=250,
            blit=True,
        )

        self.chord_labelframe = ttk.LabelFrame(self.rframe, text='Chord')
        self.chord_labelframe.pack(side=tk.TOP)
        self.chord_label = ttk.Label(self.chord_labelframe, text='', font=('Helvetica', 20))
        self.chord_label.pack(side=tk.TOP)

        self.chroma_labelframe = ttk.LabelFrame(self.rframe, text='Chroma')
        self.chroma_labelframe.pack(side=tk.TOP)
        self.chroma_label = ttk.Label(self.chroma_labelframe, text='', font=('Helvetica', 20))
        self.chroma_label.pack(side=tk.TOP)

        self.master.protocol("WM_DELETE_WINDOW", self.quit)

    def reload_fig(self):
        self.stop()
        self.fig_wave.clf()
        self.fig_specgram.clf()
        self.fig_spec.clf()

        ax = self.fig_wave.add_subplot(111)
        ax.set_xlim([0, self.WAVE_RES_X])
        ax.xaxis.set_visible(False)
        wave = [self.wave[i * len(self.wave) // self.WAVE_RES_X] for i in range(self.WAVE_RES_X)]
        self.img_wave = ax.plot(wave, color='#4582ec', linewidth=1)

        ax = self.fig_specgram.add_subplot(111)
        ax.set_yscale('log')
        ax.set_ylim([60, self.sr / 2])
        specgram = self.specgram[:, [i * self.specgram.shape[1] // self.SPECGRAM_RES_X for i in range(self.SPECGRAM_RES_X)]]
        self.img_specgram = ax.imshow(
            specgram,
            extent=[0, specgram.shape[1], 0, self.sr / 2],
            aspect='auto',
            interpolation='nearest',
            cmap='inferno',
        )
        f0_out = [self.f0_out[i * len(self.f0_out) // self.SPECGRAM_RES_X] for i in range(self.SPECGRAM_RES_X)]
        f0_in = [0 for i in range(self.SPECGRAM_RES_X)]
        self.img_f0 = ax.plot(f0_out, color='#d9534f')
        self.img_f1 = ax.plot(f0_in, color='#02b875')

        ax = self.fig_spec.add_subplot(111)
        ax.set_xlim([0, self.sr // 2])
        spec = self.spec[0]
        spec = [spec[i * len(spec) // self.SPEC_RES_X] for i in range(self.SPEC_RES_X)]
        x_spec = np.linspace(0, self.sr // 2, len(spec))
        self.img_spec = ax.plot(x_spec, spec, color='#f0ad4e', linewidth=1)

        self.canvas_wave.draw()
        self.canvas_specgram.draw()
        self.canvas_spec.draw()

    def play(self):
        if self.wavefile is None:
            return

        if self.play_status == PlayStatus.PLAY:
            pass
        elif self.play_status == PlayStatus.PAUSE:
            self.play_status = PlayStatus.PLAY
            self.play_button.configure(bootstyle=ttk_const.SECONDARY)
            self.pause_button.configure(bootstyle=ttk_const.WARNING)
        elif self.play_status == PlayStatus.STOP:
            self.stream_out = self.p_out.open(
                format=self.p_out.get_format_from_width(self.wavefile.getsampwidth()),
                channels=self.wavefile.getnchannels(),
                rate=self.wavefile.getframerate(),
                output=True,
            )
            self.stream_in = self.p_in.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sr,
                input=True,
                frames_per_buffer=self.fft_shift,
                stream_callback=self.play_in,
            )
            self.play_status = PlayStatus.PLAY
            self.play_button.configure(bootstyle=ttk_const.SECONDARY)
            self.pause_button.configure(bootstyle=ttk_const.WARNING)
            self.t_play_out = threading.Thread(target=self.play_out, daemon=True)
            self.t_play_out.start()
            self.t_update_gui = threading.Thread(target=self.update_gui, daemon=True)
            self.t_update_gui.start()

    def play_out(self):
        data = self.wavefile.readframes(self.P_OUT_CHUNK)
        self.play_ms += self.P_OUT_CHUNK / self.sr * 1000

        while data != b'':
            if self.play_status == PlayStatus.PLAY:
                self.stream_out.write(data)
                data = self.wavefile.readframes(self.P_OUT_CHUNK)
                self.play_ms += self.P_OUT_CHUNK / self.sr * 1000
            elif self.play_status == PlayStatus.PAUSE:
                time.sleep(0.1)
            elif self.play_status == PlayStatus.STOP:
                return

        self.stop()

    def play_in(self, in_data, frame_count, time_info, status):
        frame = np.frombuffer(in_data, dtype=np.float32)
        self.stacked_input = np.hstack((self.stacked_input, frame))
        if not len(self.stacked_input) >= self.fft_size:
            return (in_data, pyaudio.paContinue)
        self.stacked_input = self.stacked_input[-self.fft_size:]

        frame = self.stacked_input
        frame_hammed = frame * np.hamming(self.fft_size)
        self.vol_in = frame2vol(frame)
        self.f0_in[min(len(self.f0_in) - 1, int(self.play_ms * self.sr / self.fft_shift / 1000))] = \
            extract_f0(frame_hammed, self.sr, 261, 524) if self.vol_in > self.vol_threshold else 0
        self.chroma_in = np.argmax(spec2chroma(np.abs(frame2spec(frame)), self.sr))
        return (in_data, pyaudio.paContinue)

    def update_gui(self):
        while True:
            if self.play_status == PlayStatus.PLAY:
                time.sleep(0.1)
                chord = self.chordgram[min(len(self.chordgram) - 1, int(self.play_ms * self.sr / self.fft_shift / 1000))]
                self.time_label.configure(text=self._format_time(self.play_ms))
                self.chord_label.configure(text=CHORDS_L[chord][0] if chord >= 0 else '')
                self.chroma_label.configure(text=Note(self.chroma_in).name if (self.vol_in or -1000) > self.vol_threshold else '')
            elif self.play_status == PlayStatus.PAUSE:
                time.sleep(0.1)
            elif self.play_status == PlayStatus.STOP:
                return

    def pause(self):
        if self.play_status == PlayStatus.PLAY:
            self.play_status = PlayStatus.PAUSE
            self.play_button.configure(bootstyle=ttk_const.SUCCESS)
            self.pause_button.configure(bootstyle=ttk_const.SECONDARY)

    def stop(self):
        if self.play_status == PlayStatus.PLAY:
            self.pause()
        if self.play_status == PlayStatus.PAUSE:  # PLAY also ends up here
            self.play_status = PlayStatus.STOP
            self.stream_out.close()
            self.stream_in.close()
            self.wavefile.rewind()
            self.play_ms = 0
            self.play_button.configure(bootstyle=ttk_const.SUCCESS)
            self.pause_button.configure(bootstyle=ttk_const.SECONDARY)
            self.time_label.configure(text=self._format_time(self.play_ms))
            self.chord_label.configure(text='')

    def update_img_wave(self, frame_idx):
        if self.img_wave is None:
            return tuple()
        wave = [self.wave[i * len(self.wave) // self.WAVE_RES_X] for i in range(self.WAVE_RES_X)]
        self.img_wave[0].set_ydata(wave)
        return (self.img_wave[0],)

    def update_img_specgram(self, frame_idx):
        if self.img_specgram is None:
            return tuple()
        specgram = self.specgram[:, [i * self.specgram.shape[1] // self.SPECGRAM_RES_X for i in range(self.SPECGRAM_RES_X)]]
        self.img_specgram.set_data(specgram)
        self.img_f0[0].set_ydata([self.f0_out[i * len(self.f0_out) // self.SPECGRAM_RES_X] for i in range(self.SPECGRAM_RES_X)])
        self.img_f1[0].set_ydata([self.f0_in[i * len(self.f0_in) // self.SPECGRAM_RES_X] for i in range(self.SPECGRAM_RES_X)])
        return self.img_specgram, self.img_f0[0], self.img_f1[0]

    def update_img_spec(self, frame_idx):
        if self.img_spec is None:
            return tuple()
        spec = self.spec[min(len(self.spec) - 1, int(self.play_ms * self.sr / self.fft_shift / 1000))]
        spec = [spec[i * len(spec) // self.SPEC_RES_X] for i in range(self.SPEC_RES_X)]
        self.img_spec[0].set_ydata(spec)
        return (self.img_spec[0],)

    def open_file(self, filepath=None):
        self.stop()
        if filepath is None:
            fTyp = [("wav file", "*.wav")]
            iDir = os.path.abspath(os.path.dirname(__file__))
            self.filepath = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
            if self.filepath == '':
                return
        else:
            print(filepath)
            self.filepath = filepath

        self.fig_specgram.clf()
        self.fig_wave.clf()
        self.fig_spec.clf()
        self.filename_label.configure(text='Loading...')

        self.wavefile = None
        self.wave = None
        self.specgram = None
        self.spec = None
        self.f0_out = None
        self.chordgram = None
        self.play_ms = 0

        self.fft_size = int(self.fft_size_entry.get())
        self.fft_shift = int(self.fft_shift_entry.get())
        self.vol_threshold = int(self.vol_threshold_scale.get())

        self.wave, self.sr = librosa.load(self.filepath, sr=None)
        self.wavefile = pywave.open(self.filepath, 'r')
        specgram = wave2specgram(self.wave, self.fft_size, self.fft_shift)
        log_specgram = np.log(np.abs(specgram) + 1e-10)  # avoid log(0)
        self.spec = log_specgram
        self.specgram = np.flipud(np.array(log_specgram).T)

        f0 = []
        chordgram = []
        for i in np.arange(0, len(self.wave) - self.fft_size, self.fft_shift):
            idx = int(i)
            frame = self.wave[idx:idx + self.fft_size]
            frame_hammed = frame * np.hamming(self.fft_size)
            vol = frame2vol(frame)
            f0.append(
                extract_f0(frame_hammed, self.sr, 261, 524)  # C4 ~ C5
                if vol > self.vol_threshold else 0
            )
            chroma = spec2chroma(np.abs(frame2spec(frame)), self.sr)
            score_l = []
            for chord in CHORDS_L:
                score_l.append(
                    1.0 * chroma[chord[1].value] +
                    0.5 * chroma[chord[2].value] +
                    0.8 * chroma[chord[3].value]
                )
            chordgram.append(
                np.argmax(score_l)
                if vol > self.vol_threshold else -1
            )
        self.f0_out = np.asarray(f0)
        self.f0_in = [0 for _ in range(len(self.f0_out))]
        self.chordgram = np.asarray(chordgram)

        self.filename_label.configure(text=self.filepath)
        self.duration_label.configure(
            text=f'Duration: {self._format_time(len(self.wave) / self.sr * 1000)}')
        self.sr_label.configure(text=f'Sampling rate: {self.sr} Hz')
        self.reload_fig()

    def apply_vc(self):
        self.stop()
        self.vc_freq = self.vc_freq_scale.get()
        self.vc_depth = self.vc_depth_scale.get()
        wave = self.wave * (self.vc_depth / 100.0 * np.sin(
            2.0 * np.pi * self.vc_freq * np.arange(len(self.wave)) / self.sr))
        wave = (wave * 32768.0).astype(np.int16)
        filepath = self.filepath.replace(".wav", "_vc.wav")
        scipy.io.wavfile.write(filepath, self.sr, wave)
        self.open_file(filepath=filepath)

    def apply_tremolo(self):
        self.stop()
        self.tremolo_freq = self.tremolo_freq_scale.get()
        self.tremolo_depth = self.tremolo_depth_scale.get()
        wave = self.wave * (1.0 + self.tremolo_depth / 100.0 * np.sin(
            2.0 * np.pi * self.tremolo_freq * np.arange(len(self.wave)) / self.sr))
        wave = (wave * 32768.0).astype(np.int16)
        filepath = self.filepath.replace(".wav", "_tremolo.wav")
        scipy.io.wavfile.write(filepath, self.sr, wave)
        self.open_file(filepath=filepath)

    def quit(self):
        self.stop()
        self.master.quit()
        self.master.destroy()

    def _format_time(self, ms):
        m = int(ms // 1000 // 60)
        s = int(ms // 1000 % 60)
        ms = int(ms // 10 % 100)
        return f'{m:02}:{s:02}:{ms:02}'


if __name__ == '__main__':
    root = tk.Tk()
    root.title('Audio Visualizer')
    app = AudioVisualizer(master=root)
    app.mainloop()
