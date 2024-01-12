import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter
import tkinter.filedialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from utils import (
    Note,
    SpeechStatus,
    wave2specgram,
    extract_f0,
    frame2spec,
    frame2vol,
    spec2chroma,
    detect_speech,
)

root = tkinter.Tk()
root.title('Audio Visualizer')

filename = ''

CHORDS_L = [
  # name, root, third, fifth
  ['C Major', Note.C, Note.E, Note.G],
  ['C Minor', Note.C, Note.DS, Note.G],
  ['C# Major', Note.CS, Note.F, Note.GS],
  ['C# Minor', Note.CS, Note.E, Note.GS],
  ['D Major', Note.D, Note.FS, Note.A],
  ['D Minor', Note.D, Note.F, Note.A],
  ['D# Major', Note.DS, Note.G, Note.AS],
  ['D# Minor', Note.DS, Note.FS, Note.AS],
  ['E Major', Note.E, Note.GS, Note.B],
  ['E Minor', Note.E, Note.G, Note.B],
  ['F Major', Note.F, Note.A, Note.C],
  ['F Minor', Note.F, Note.GS, Note.C],
  ['F# Major', Note.FS, Note.AS, Note.CS],
  ['F# Minor', Note.FS, Note.A, Note.CS],
  ['G Major', Note.G, Note.B, Note.D],
  ['G Minor', Note.G, Note.AS, Note.D],
  ['G# Major', Note.GS, Note.C, Note.DS],
  ['G# Minor', Note.GS, Note.B, Note.DS],
  ['A Major', Note.A, Note.CS, Note.E],
  ['A Minor', Note.A, Note.C, Note.E],
  ['A# Major', Note.AS, Note.D, Note.F],
  ['A# Minor', Note.AS, Note.CS, Note.F],
  ['B Major', Note.B, Note.DS, Note.FS],
  ['B Minor', Note.B, Note.D, Note.FS],
]

def open_file(_):
    fTyp = [("wav file", "*.wav")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    global filename
    filename = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)

    load_file(filename)

def load_file(filename):
    if filename == '':
        return

    sr = int(sr_entry.get())
    frame_size = int(frame_size_entry.get())
    shift_size = int(shift_size_entry.get())
    vol_threshold = int(vol_threshold_entry.get())

    wave = librosa.load(filename, sr=sr)[0]
    specgram = wave2specgram(wave, frame_size, shift_size)
    log_specgram = np.log(np.abs(specgram))

    f0 = []
    chordgram = []
    for i in np.arange(0, len(wave) - frame_size, shift_size):
        idx = int(i)
        frame = wave[idx : idx + frame_size]
        frame_hammed = frame * np.hamming(frame_size)
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

    ax = fig_waveform.add_subplot(111)
    ax.set_xlim([0, len(wave)])
    ax.xaxis.set_visible(False)
    ax.plot(wave)
    canvas_waveform.draw()

    ax = fig_specgram.add_subplot(111)
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
    canvas_specgram.draw()

plt.rcParams['image.cmap'] = "inferno"

sr_entry = tkinter.Entry(root)
sr_entry.insert(tkinter.END, '44100')
sr_entry.pack(side=tkinter.TOP)

frame_size_entry = tkinter.Entry(root)
frame_size_entry.insert(tkinter.END, '4096')
frame_size_entry.pack(side=tkinter.TOP)

shift_size_entry = tkinter.Entry(root)
shift_size_entry.insert(tkinter.END, '1280')
shift_size_entry.pack(side=tkinter.TOP)

vol_threshold_entry = tkinter.Entry(root)
vol_threshold_entry.insert(tkinter.END, '-30')
vol_threshold_entry.pack(side=tkinter.TOP)

open_file_button = tkinter.Button(root, text='Open file')
open_file_button.pack(side=tkinter.TOP)
open_file_button.bind('<Button-1>', open_file)

frame_waveform = tkinter.Frame(root)
frame_waveform.pack(side=tkinter.TOP)

fig_waveform = plt.figure(figsize=(16, 3))
canvas_waveform = FigureCanvasTkAgg(fig_waveform, master=frame_waveform)
canvas_waveform.get_tk_widget().pack(side=tkinter.TOP)

frame_specgram = tkinter.Frame(root)
frame_specgram.pack(side=tkinter.TOP)

fig_specgram = plt.figure(figsize=(16, 5))
canvas_specgram = FigureCanvasTkAgg(fig_specgram, master=frame_specgram)
canvas_specgram.get_tk_widget().pack(side=tkinter.TOP)

root.mainloop()
