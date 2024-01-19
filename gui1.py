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
import wave as pywave

from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils import (
    CHORDS_L,
    wave2specgram,
    extract_f0,
    frame2spec,
    frame2vol,
    spec2chroma,
)


class AudioVisualizer(ttk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.filepath: str = ""
        self.wavefile = None
        self.sr: int = None
        self.wave: np.ndarray = None
        self.specgram: np.ndarray = None
        self.f0: np.ndarray = None
        self.chromagram: np.ndarray = None
        self.chordgram: np.ndarray = None
        self.is_playing: bool = False
        self.play_ms: int = 0

        self.WAVE_RES_X = 10000
        self.SPECGRAM_RES_X = 100

        self.master = master
        self.pack()

        self.lframe = ttk.Frame(self, borderwidth=2)
        self.lframe.pack(side=tk.LEFT)

        self.open_file_button = ttk.Button(
            self.lframe,
            text='Open file',
            bootstyle=ttk_const.PRIMARY,
            command=self.open_file,
        )
        self.open_file_button.pack(side=tk.TOP)

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
        self.vol_threshold_entry = ttk.Entry(self.vol_threshold_frame)
        self.vol_threshold_entry.insert(tk.END, '-30')
        self.vol_threshold_entry.pack(side=tk.LEFT, anchor=tk.W)

        self.filename_label = ttk.Label(self.lframe, text='')
        self.filename_label.pack(side=tk.TOP)
        self.duration_label = ttk.Label(self.lframe, text='')
        self.duration_label.pack(side=tk.TOP)
        self.sr_label = ttk.Label(self.lframe, text='')
        self.sr_label.pack(side=tk.TOP)

        self.quit_button = ttk.Button(
            self.lframe,
            text='Quit app',
            bootstyle=ttk_const.DANGER,
            command=self.quit,
        )
        self.quit_button.pack(side=tk.TOP)

        self.frame_wave = ttk.Frame(self)
        self.frame_wave.pack(side=tk.TOP)
        self.fig_wave = plt.figure(figsize=(12, 3))
        self.fig_wave.subplots_adjust(0.05, 0.01, 0.95, 0.99)
        self.canvas_wave = FigureCanvasTkAgg(self.fig_wave, master=self.frame_wave)
        self.canvas_wave.get_tk_widget().pack(side=tk.TOP)

        self.frame_specgram = ttk.Frame(self)
        self.frame_specgram.pack(side=tk.TOP)
        self.fig_specgram = plt.figure(figsize=(12, 5))
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
            interval=100,
            blit=True,
        )

        self.ctrl_frame = ttk.Frame(self)
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

    def reload_fig(self):
        self.fig_wave.clf()
        ax = self.fig_wave.add_subplot(111)
        ax.set_xlim([0, self.WAVE_RES_X])
        ax.xaxis.set_visible(False)
        wave = [self.wave[i * len(self.wave) // self.WAVE_RES_X] for i in range(self.WAVE_RES_X)]
        self.img_wave = ax.plot(wave, color='#4582ec', linewidth=1)

        self.fig_specgram.clf()
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
        f0 = [self.f0[i * len(self.f0) // self.SPECGRAM_RES_X] for i in range(self.SPECGRAM_RES_X)]
        self.img_f0 = ax.plot(f0, color='#d9534f')
        ax = ax.twinx()
        ax.set_ylim([0, 23])
        chordgram = [self.chordgram[i * len(self.chordgram) // self.SPECGRAM_RES_X] for i in range(self.SPECGRAM_RES_X)]
        self.img_chordgram = ax.plot(chordgram, color='#4582ec')

        self.canvas_wave.draw()
        self.canvas_specgram.draw()

    def play(self):
        if self.is_playing or self.wavefile is None:
            return
        self.p_out = pyaudio.PyAudio()
        self.stream_out = self.p_out.open(
            format=self.p_out.get_format_from_width(self.wavefile.getsampwidth()),
            channels=self.wavefile.getnchannels(),
            rate=self.wavefile.getframerate(),
            output=True,
        )
        self.is_playing = True
        self.play_button.configure(bootstyle=ttk_const.SECONDARY)
        self.pause_button.configure(bootstyle=ttk_const.WARNING)
        if self.play_ms == 0:
            self.t_play_out = threading.Thread(target=self.play_out, daemon=True)
            self.t_play_out.start()
            self.t_update_gui = threading.Thread(target=self.update_gui, daemon=True)
            self.t_update_gui.start()

    def update_img_wave(self, frame_idx):
        if not self.is_playing:
            return tuple()
        wave = [self.wave[i * len(self.wave) // self.WAVE_RES_X] for i in range(self.WAVE_RES_X)]
        self.img_wave[0].set_ydata(wave)
        return (self.img_wave[0],)

    def update_img_specgram(self, frame_idx):
        if not self.is_playing:
            return tuple()
        specgram = self.specgram[:, [i * self.specgram.shape[1] // self.SPECGRAM_RES_X for i in range(self.SPECGRAM_RES_X)]]
        self.img_specgram.set_data(specgram)
        self.img_f0[0].set_ydata([self.f0[i * len(self.f0) // self.SPECGRAM_RES_X] for i in range(self.SPECGRAM_RES_X)])
        self.img_chordgram[0].set_ydata([self.chordgram[i * len(self.chordgram) // self.SPECGRAM_RES_X] for i in range(self.SPECGRAM_RES_X)])
        return self.img_specgram, self.img_f0[0], self.img_chordgram[0]

    def play_out(self):
        CHUNK = 1024
        data = self.wavefile.readframes(CHUNK)
        self.play_ms += CHUNK / self.sr * 1000
        while data != b'':
            if not self.is_playing:
                continue
            self.stream_out.write(data)
            data = self.wavefile.readframes(CHUNK)
            self.play_ms += CHUNK / self.sr * 1000
        self.stop()

    def update_gui(self):
        while True:
            time.sleep(0.1)
            if not self.is_playing:
                continue
            self.time_label.configure(text=self._format_time(self.play_ms))

    def pause(self):
        if not self.is_playing:
            return
        self.is_playing = False
        self.play_button.configure(bootstyle=ttk_const.SUCCESS)
        self.pause_button.configure(bootstyle=ttk_const.SECONDARY)

    def stop(self):
        self.pause()
        self.stream_out.close()
        self.p_out.terminate()
        self.wavefile.rewind()
        self.play_ms = 0
        self.play_button.configure(bootstyle=ttk_const.SUCCESS)
        self.pause_button.configure(bootstyle=ttk_const.SECONDARY)
        self.time_label.configure(text=self._format_time(self.play_ms))

    def open_file(self):
        fTyp = [("wav file", "*.wav")]
        iDir = os.path.abspath(os.path.dirname(__file__))
        self.filepath = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
        if self.filepath == '':
            return

        self.fig_specgram.clf()
        self.fig_wave.clf()
        self.filename_label.configure(text='Loading...')

        self.wavefile = None
        self.wave = None
        self.specgram = None
        self.f0 = None
        self.chromagram = None
        self.chordgram = None
        self.is_playing = False
        self.play_ms = 0

        fft_size = int(self.fft_size_entry.get())
        fft_shift = int(self.fft_shift_entry.get())
        vol_threshold = int(self.vol_threshold_entry.get())

        self.wave, self.sr = librosa.load(self.filepath, sr=None)
        self.wavefile = pywave.open(self.filepath, 'r')
        specgram = wave2specgram(self.wave, fft_size, fft_shift)
        log_specgram = np.log(np.abs(specgram) + 1e-10)  # avoid log(0)
        self.specgram = np.flipud(np.array(log_specgram).T)

        f0 = []
        chordgram = []
        for i in np.arange(0, len(self.wave) - fft_size, fft_shift):
            idx = int(i)
            frame = self.wave[idx:idx + fft_size]
            frame_hammed = frame * np.hamming(fft_size)
            vol = frame2vol(frame)
            f0.append(
                extract_f0(frame_hammed, self.sr)
                if vol > vol_threshold else 0
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
                if vol > vol_threshold else 0
            )
        self.f0 = np.asarray(f0)
        self.chordgram = np.asarray(chordgram)

        self.filename_label.configure(text=self.filepath)
        self.duration_label.configure(
            text=f'Duration: {self._format_time(len(self.wave) / self.sr * 1000)}')
        self.sr_label.configure(text=f'Sampling rate: {self.sr} Hz')
        self.reload_fig()

    def quit(self):
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
