from enum import Enum
import math
import numpy as np

class SpeechStatus(Enum):
  QUIET = 0
  UNVOICED = 1
  VOICED = 2

class Note(Enum):
  C = 0
  CS = 1
  D = 2
  DS = 3
  E = 4
  F = 5
  FS = 6
  G = 7
  GS = 8
  A = 9
  AS = 10
  B = 11

def nn2hz(nn):
  return 440 * 2 ** ((nn - 69) / 12)

def hz2nn(hz):
  return int(round(12 * math.log2(hz / 440) + 69))

def spec2chroma(spec, sr):
  chroma = np.zeros(12)
  for i, s in enumerate(spec):
    if i == 0:
      continue
    nn = hz2nn(i * sr / 2 / len(spec))
    chroma[nn % 12] += np.abs(s)
  return chroma

def wave2specgram(wave, size_frame, size_shift):
  specgram = []
  for i in np.arange(0, len(wave) - size_frame, size_shift):
    idx = int(i)
    frame = wave[idx : idx + size_frame] * np.hamming(size_frame)
    spec = np.fft.rfft(frame)
    specgram.append(spec)
  return specgram

def wave2cepsgram(wave, size_frame, size_shift, ceps_size=None):
  cepsgram = []
  for i in np.arange(0, len(wave) - size_frame, size_shift):
    idx = int(i)
    frame = wave[idx : idx + size_frame] * np.hamming(size_frame)
    spec = np.log(np.abs(np.fft.rfft(frame)))
    ceps = np.real(np.fft.rfft(spec))
    cepsgram.append(ceps if ceps_size == None else ceps[:ceps_size])
  return cepsgram

def frame2spec(frame):
  return np.fft.rfft(frame * np.hamming(len(frame)))

def frame2ceps(frame, ceps_size=None):
  spec = np.log(np.abs(np.fft.rfft(frame)))
  ceps = np.real(np.fft.rfft(spec))
  return ceps if ceps_size == None else ceps[:ceps_size]

def frame2vol(frame):
  return 20 * np.log10(np.sqrt(np.mean(frame**2)))

def detect_speech(frame, vol_threshold, zero_cross_threshold):
  vol = frame2vol(frame)
  zero_cross = int(sum(np.abs(np.diff(np.sign(frame))) // 2))

  if vol < vol_threshold:
    return SpeechStatus.QUIET
  if zero_cross > zero_cross_threshold:
    return SpeechStatus.UNVOICED
  return SpeechStatus.VOICED

def extract_f0(frame, sr):
  autocorr = np.correlate(frame, frame, mode='full')
  autocorr = autocorr[len(autocorr)//2:]
  peakindices = [
    i for i in range(1, len(autocorr)-1)
    if autocorr[i-1] < autocorr[i] and autocorr[i+1] < autocorr[i]
  ]
  if len(peakindices) == 0:
    return 0
  max_peak_index = max(peakindices, key=lambda i: autocorr[i])
  return sr / max_peak_index
