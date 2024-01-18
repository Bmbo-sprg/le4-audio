import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk
import tkinter.filedialog
import ttkbootstrap as ttk
import ttkbootstrap.constants as ttk_const

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils import (
    CHORDS_L,
    wave2specgram,
    extract_f0,
    frame2spec,
    frame2vol,
    spec2chroma,
)

'''
AudioVisualizer
    |--- __init__ <- ガチの最初に呼ばれる (ウィジェットをとりあえずpackし、figは作らない (音声がNoneなので))
    |--- load_file <- ファイルを読んで、figを初期化する (表示範囲は最大)
    |--- initialize_fig <- いる？
    |--- reload_fig <- 表示範囲を更新する
    |--- タイマー関連 <- 100msをchunkとして、サンプルコードをパクる
    |--- 音声再生関連
'''

class AudioVisualizer(ttk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.filename: str = ""
        self.sr: int = None
        self.wave: np.ndarray = None
        self.specgram: np.ndarray = None
        self.f0: np.ndarray = None
        self.chromagram: np.ndarray = None
        self.chordgram: np.ndarray = None
        self.is_playing: bool = False
        self.play_ms: int = 0

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
        self.fft_size_entry = ttk.Entry(self.fft_size_frame)
        self.fft_size_entry.insert(tk.END, '4096')
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

        self.frame_wave = ttk.Frame(self)
        self.frame_wave.pack(side=tk.TOP)
        self.fig_wave = plt.figure(figsize=(16, 3))
        self.fig_wave.subplots_adjust(0.05, 0.01, 0.95, 0.99)
        self.canvas_wave = FigureCanvasTkAgg(self.fig_wave, master=self.frame_wave)
        self.canvas_wave.get_tk_widget().pack(side=tk.TOP)

        self.frame_specgram = ttk.Frame(self)
        self.frame_specgram.pack(side=tk.TOP)
        self.fig_specgram = plt.figure(figsize=(16, 5))
        self.fig_specgram.subplots_adjust(0.05, 0.05, 0.95, 0.99)
        self.canvas_specgram = FigureCanvasTkAgg(self.fig_specgram, master=self.frame_specgram)
        self.canvas_specgram.get_tk_widget().pack(side=tk.TOP)

        self.ctrl_frame = ttk.Frame(self)
        self.ctrl_frame.pack(side=tk.TOP)
        self.time_label = ttk.Label(self.ctrl_frame, text='00:00:00')
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

    def load_file(self):
        if self.filename == '':
            return

        self.fig_specgram.clf()
        self.fig_wave.clf()
        self.filename_label.configure(text='Loading...')

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

        self.wave, self.sr = librosa.load(self.filename, sr=None)
        self.specgram = wave2specgram(self.wave, fft_size, fft_shift)

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

        self.filename_label.configure(text=self.filename)
        self.duration_label.configure(text=f'Duration: {len(self.wave) / self.sr:.2f} sec')
        self.sr_label.configure(text=f'Sampling rate: {self.sr} Hz')
        self.initialize_fig()
        # self.update_plot()

    def initialize_fig(self):
        self.fig_wave.clf()
        self.fig_specgram.clf()
        ax = self.fig_wave.add_subplot()
        ax.xaxis.set_visible(False)
        self.plot_wave = ax.plot(self.wave)
        ax = self.fig_specgram.add_subplot()
        ax.set_yscale('log')
        self.plot_specgram = ax.pcolormesh(
            np.flipud(np.transpose(np.log(np.abs(self.specgram)))),
            shading='nearest',
            cmap='inferno',
        )

    def play(self):
        if self.is_playing:
            return
        self.is_playing = True
        self.play_button.configure(bootstyle=ttk_const.SECONDARY)
        self.pause_button.configure(bootstyle=ttk_const.WARNING)
        self.increment()

    def increment(self):
        if not self.is_playing:
            return
        self.play_ms += 100
        m = self.play_ms // 1000 // 60
        s = self.play_ms // 1000 % 60
        ms = self.play_ms // 10 % 100
        self.time_label.configure(text=f'{m:02}:{s:02}:{ms:02}')
        self.master.after(100, self.increment)

    def pause(self):
        if not self.is_playing:
            return
        self.is_playing = False
        self.play_button.configure(bootstyle=ttk_const.SUCCESS)
        self.pause_button.configure(bootstyle=ttk_const.SECONDARY)

    def stop(self):
        self.is_playing = False
        self.play_ms = 0
        self.play_button.configure(bootstyle=ttk_const.SUCCESS)
        self.pause_button.configure(bootstyle=ttk_const.SECONDARY)
        self.time_label.configure(text='00:00:00')

    def open_file(self):
        fTyp = [("wav file", "*.wav")]
        iDir = os.path.abspath(os.path.dirname(__file__))
        self.filename = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
        self.load_file()

    def update_plot(self):
        if self.wave is not None:
            ax = self.fig_wave.add_subplot(111)
            ax.set_xlim([0, len(self.wave)])
            ax.xaxis.set_visible(False)
            ax.plot(self.wave)
            self.canvas_wave.draw()

        if self.specgram is not None:
            log_specgram = np.log(np.abs(self.specgram) + 1e-10)  # avoid log(0)
            ax = self.fig_specgram.add_subplot(111)
            ax.set_yscale('log')
            ax.set_ylim([60, self.sr / 2])
            ax.imshow(
                np.flipud(np.array(log_specgram).T),
                extent=[0, len(log_specgram), 0, self.sr/2],
                aspect='auto',
                interpolation='nearest',
                cmap='inferno',
            )

            if self.f0 is not None:
                ax.plot(self.f0, label='f0', color='red')

            if self.chordgram is not None:
                ax2 = ax.twinx()
                ax2.set_ylim([0, 23])
                ax2.plot(self.chordgram, label='chord', color='blue')
            self.canvas_specgram.draw()


if __name__ == '__main__':
    root = tk.Tk()
    root.title('Audio Visualizer')
    app = AudioVisualizer(master=root)
    app.mainloop()
