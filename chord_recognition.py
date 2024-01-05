import librosa
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from utils import Note, SpeechStatus, spec2chroma, wave2specgram, detect_speech

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

SR = int(input('Enter sampling rate (default: 16000): ') or 16000)
size_frame = int(input('Enter frame size (default: 512): ') or 512)
size_shift = int(input('Enter shift size (default: 160): ') or 160)
vol_threshold = int(input('Enter volume threshold (dB) (default: -30): ') or -30)
zero_cross_threshold = int(input('Enter zero-cross threshold (default: 70): ') or 70)

hamming_window = np.hamming(size_frame)

def extract_chord(chroma):
  '''Return the index of the most likely chord.
  '''
  score_l = []
  for chord in CHORDS_L:
    score_l.append(
      1.0 * chroma[chord[1].value] +
      0.5 * chroma[chord[2].value] +
      0.8 * chroma[chord[3].value]
    )
  return np.argmax(score_l)

def input_callback(in_data, frame_count, time_info, status_flags):
  wave = np.frombuffer(in_data, dtype=np.int16)
  wave = wave / 32768.0  # normalize

  status = detect_speech(wave, vol_threshold, zero_cross_threshold)
  if status == SpeechStatus.QUIET:
    return (in_data, pyaudio.paContinue)
  chroma = spec2chroma(np.abs(np.fft.rfft(wave * hamming_window)), SR)
  note = Note(np.argmax(chroma))
  chord = extract_chord(chroma)
  print(f'Chord: {CHORDS_L[chord][0]:8} Note: {note.name:2}')

  return (in_data, pyaudio.paContinue)

if input('Realtime input? (Y/n)') != 'n':
  p = pyaudio.PyAudio()
  stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=SR,
    frames_per_buffer=size_frame,
    input=True,
    stream_callback=input_callback,
  )
  stream.start_stream()
  while stream.is_active():
    pass
  stream.stop_stream()
  stream.close()
  p.terminate()
else:
  filename = input('Filename of test data (without .wav): ')
  data = librosa.load(f"wave/{filename}.wav", sr=SR)[0]
  specgram = wave2specgram(data, size_frame, size_shift)
  log_specgram = np.log(np.abs(specgram))
  chromagram = []
  chordgram = []
  for spec in specgram:
    chroma = spec2chroma(np.abs(spec), SR)
    chromagram.append(chroma)
    chordgram.append(extract_chord(chroma))

  plt.rcParams['image.cmap'] = "inferno"
  fig = plt.figure()

  # plot spectrogram (log)
  plt.xlabel('Sample')
  plt.ylabel('Frequency [Hz]')
  plt.imshow(
    np.flipud(np.array(log_specgram).T),
    extent=[0, len(log_specgram), 0, SR/2],
    aspect='auto',
    interpolation='nearest',
  )
  plt.gca().set_yscale('log')
  plt.ylim([60, SR / 2])
  fig.savefig(f'fig/{filename}_specgram.png')
  plt.clf()

  # plot chromagram & chordgram
  ax1 = fig.add_subplot(111)
  ax1.set_xlabel('Sample')
  ax1.set_ylabel('Chroma')
  ax1.imshow(
    np.flipud(np.array(chromagram).T),
    extent=[0, len(chromagram), 0, 12],
    aspect='auto',
    interpolation='nearest',
  )
  ax2 = ax1.twinx()
  ax2.set_ylabel('Chord')
  ax2.plot(chordgram, color='red', label='chord')
  fig.savefig(f'fig/{filename}_chromagram.png')
  plt.clf()
