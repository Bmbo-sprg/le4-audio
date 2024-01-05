import numpy as np
import time
import pyaudio
from collections import deque

SR = 16000
SIZE_FRAME = 512

hamming_window = np.hamming(SIZE_FRAME)

def input_callback(in_data, frame_count, time_info, status_flags):
  data = np.frombuffer(in_data, dtype=np.int16)
  data = data / 32768.0
  rms = np.sqrt(np.mean(data**2))
  vol = 20 * np.log10(rms)

  autocorr = np.correlate(data, data, mode='full')
  autocorr = autocorr[len(autocorr)//2:]
  peakindices = [
    i for i in range(1, len(autocorr)-1) if autocorr[i-1] < autocorr[i] and autocorr[i+1] < autocorr[i]
  ]
  if len(peakindices) == 0:
    f_0 = 0
  else:
    max_peak_index = max(peakindices, key=lambda i: autocorr[i])
    f_0 = SR / max_peak_index

  zero_cross = int(sum(np.abs(np.diff(np.sign(data))) // 2))
  if vol > -30:
    if zero_cross > 70:
      print(f'unvoiced, zero_cross={zero_cross}')
    else:
      print(f'voiced: f_0={f_0:.2f}Hz')
  else:
    print('not speaking')
  return (in_data, pyaudio.paContinue)

p = pyaudio.PyAudio()
stream = p.open(
  format=pyaudio.paInt16,
  channels=1,
  rate=SR,
  input=True,
  frames_per_buffer=SIZE_FRAME,
  stream_callback=input_callback,
)

stream.start_stream()
time.sleep(100)
