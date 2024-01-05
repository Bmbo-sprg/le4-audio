import numpy as np
import pyaudio
from utils import SpeechStatus, detect_speech, extract_f0

SR = 16000
SIZE_FRAME = 512

hamming_window = np.hamming(SIZE_FRAME)

def input_callback(in_data, frame_count, time_info, status_flags):
  data = np.frombuffer(in_data, dtype=np.int16) / 32768.0

  f0 = extract_f0(data, SR)
  speech_status = detect_speech(data, -30, 70)

  if speech_status == SpeechStatus.QUIET:
    print('quiet')
  elif speech_status == SpeechStatus.UNVOICED:
    print(f'unvoiced')
  else:
    print(f'voiced: f0={f0:.2f}Hz')

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
while stream.is_active():
  pass
stream.stop_stream()
stream.close()
p.terminate()
