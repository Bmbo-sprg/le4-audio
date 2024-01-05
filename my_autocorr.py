import numpy as np
import time

def my_autocorr(data):
  fft = np.fft.fft(data)
  power = fft * np.conj(fft)
  autocorr = np.real(np.fft.ifft(power))
  return autocorr

# test code
for i in range(1, 16):
  SAMPLE = 10

  a = [np.random.rand(2**i) for _ in range(SAMPLE)]

  my_time = []
  np_time = []
  allclose = []

  for j in range(SAMPLE):
    start = time.perf_counter()
    autocorr = my_autocorr(a[j])
    end = time.perf_counter()
    my_time.append(end - start)

    start = time.perf_counter()
    np_autocorr = np.correlate(a[j], a[j], mode='full')
    np_autocorr = np_autocorr[len(np_autocorr)//2:]
    end = time.perf_counter()
    np_time.append(end - start)

    allclose.append(np.allclose(autocorr, np_autocorr, atol=1e-10))

  my_time_avg = sum(my_time) / SAMPLE
  np_time_avg = sum(np_time) / SAMPLE
  print(f'2^{i} elements my_autocorr: {my_time_avg} sec')
  print(f'2^{i} elements np.correlate: {np_time_avg} sec')
  print(f'diff: {my_time_avg - np_time_avg} sec')
  print(f'factor: {my_time_avg / np_time_avg}')
  print(''.join([('.' if i else 'F') for i in allclose]))
