import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk
import tkinter.filedialog
from ttkbootstrap import (
    Button,
    Entry,
    Frame,
    Label,
    Scale,
)
from ttkbootstrap.constants import (
    PRIMARY,
    SUCCESS,
    SECONDARY,
    WARNING,
    DANGER,
)

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils import (
    CHORDS_L,
    wave2specgram,
    extract_f0,
    frame2spec,
    frame2vol,
    spec2chroma,
)


class AudioVisualizer(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        plt.rcParams['image.cmap'] = "inferno"
        self.filename: str = ""
        self.wave: np.ndarray = None
        self.is_playing: bool = False
        self.play_ms: int = 0

        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.lframe = Frame(self, borderwidth=2)
        self.lframe.pack(side=tk.LEFT)

        self.open_file_button = Button(
            self.lframe,
            text='Open file',
            bootstyle=PRIMARY,
            command=self.open_file,
        )
        self.open_file_button.pack(side=tk.TOP)

        self.fft_size_frame = Frame(self.lframe, padding=(0, 20, 0, 10))
        self.fft_size_frame.pack(side=tk.TOP)
        self.fft_size_label = Label(self.fft_size_frame, text='FFT size')
        self.fft_size_label.pack(side=tk.LEFT)
        self.fft_size_entry = Entry(self.fft_size_frame)
        self.fft_size_entry.insert(tk.END, '4096')
        self.fft_size_entry.pack(side=tk.LEFT, anchor=tk.W)

        self.fft_shift_frame = Frame(self.lframe, padding=(0, 10))
        self.fft_shift_frame.pack(side=tk.TOP)
        self.fft_shift_label = Label(self.fft_shift_frame, text='FFT shift')
        self.fft_shift_label.pack(side=tk.LEFT)
        self.fft_shift_entry = Entry(self.fft_shift_frame)
        self.fft_shift_entry.insert(tk.END, '1280')
        self.fft_shift_entry.pack(side=tk.LEFT)

        self.vol_threshold_frame = Frame(self.lframe, padding=(0, 10))
        self.vol_threshold_frame.pack(side=tk.TOP)
        self.vol_threshold_label = Label(self.vol_threshold_frame, text='Volume threshold')
        self.vol_threshold_label.pack(side=tk.LEFT)
        self.vol_threshold_entry = Entry(self.vol_threshold_frame)
        self.vol_threshold_entry.insert(tk.END, '-30')
        self.vol_threshold_entry.pack(side=tk.LEFT)

        self.frame_waveform = Frame(self)
        self.frame_waveform.pack(side=tk.TOP)
        self.fig_waveform = plt.figure(figsize=(16, 3))
        self.canvas_waveform = FigureCanvasTkAgg(self.fig_waveform, master=self.frame_waveform)
        self.canvas_waveform.get_tk_widget().pack(side=tk.TOP)

        self.frame_specgram = Frame(self)
        self.frame_specgram.pack(side=tk.TOP)
        self.fig_specgram = plt.figure(figsize=(16, 5))
        self.canvas_specgram = FigureCanvasTkAgg(self.fig_specgram, master=self.frame_specgram)
        self.canvas_specgram.get_tk_widget().pack(side=tk.TOP)

        self.ctrl_frame = Frame(self)
        self.ctrl_frame.pack(side=tk.TOP)
        self.time_label = Label(self.ctrl_frame, text='00:00:00')
        self.time_label.pack(side=tk.LEFT)
        self.play_button = Button(
            self.ctrl_frame,
            text='Play',
            bootstyle=SUCCESS,
            command=self.play,
        )
        self.play_button.pack(side=tk.LEFT)
        self.pause_button = Button(
            self.ctrl_frame,
            text='Pause',
            bootstyle=SECONDARY,
            command=self.pause,
        )
        self.pause_button.pack(side=tk.LEFT)
        self.stop_button = Button(
            self.ctrl_frame,
            text='Stop',
            bootstyle=DANGER,
            command=self.stop,
        )
        self.stop_button.pack(side=tk.LEFT)

    def play(self):
        if self.is_playing:
            return
        self.is_playing = True
        self.play_button.configure(bootstyle=SECONDARY)
        self.pause_button.configure(bootstyle=WARNING)
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
        self.play_button.configure(bootstyle=SUCCESS)
        self.pause_button.configure(bootstyle=SECONDARY)

    def stop(self):
        self.is_playing = False
        self.play_ms = 0
        self.play_button.configure(bootstyle=SUCCESS)
        self.pause_button.configure(bootstyle=SECONDARY)
        self.time_label.configure(text='00:00:00')

    def open_file(self):
        fTyp = [("wav file", "*.wav")]
        iDir = os.path.abspath(os.path.dirname(__file__))
        self.filename = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
        self.load_file(self.filename)

    def load_file(self, filename):
        if filename == '':
            return

        self.fig_specgram.clf()
        self.fig_waveform.clf()

        fft_size = int(self.fft_size_entry.get())
        fft_shift = int(self.fft_shift_entry.get())
        vol_threshold = int(self.vol_threshold_entry.get())

        self.wave, sr = librosa.load(filename, sr=None)
        wave = self.wave
        specgram = wave2specgram(wave, fft_size, fft_shift)
        log_specgram = np.log(np.abs(specgram))

        f0 = []
        chordgram = []
        for i in np.arange(0, len(wave) - fft_size, fft_shift):
            idx = int(i)
            frame = wave[idx:idx + fft_size]
            frame_hammed = frame * np.hamming(fft_size)
            vol = frame2vol(frame)
            f0.append(
                extract_f0(frame_hammed, sr)
                if vol > vol_threshold else 0
            )
            chroma = spec2chroma(np.abs(frame2spec(frame)), sr)
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

        ax = self.fig_waveform.add_subplot(111)
        ax.set_xlim([0, len(wave)])
        ax.xaxis.set_visible(False)
        ax.plot(wave)
        self.fig_waveform.subplots_adjust(0.05, 0.01, 0.95, 0.99)
        self.canvas_waveform.draw()

        ax = self.fig_specgram.add_subplot(111)
        ax.set_yscale('log')
        ax.set_ylim([60, sr / 2])
        ax.imshow(
            np.flipud(np.array(log_specgram).T),
            extent=[0, len(log_specgram), 0, sr/2],
            aspect='auto',
            interpolation='nearest',
        )
        ax.plot(f0, label='f0', color='red')
        ax2 = ax.twinx()
        ax2.set_ylim([0, 23])
        ax2.plot(chordgram, label='chord', color='blue')
        self.fig_specgram.subplots_adjust(0.05, 0.05, 0.95, 0.99)
        self.canvas_specgram.draw()


if __name__ == '__main__':
    root = tk.Tk()
    root.title('Audio Visualizer')
    app = AudioVisualizer(master=root)
    app.mainloop()
