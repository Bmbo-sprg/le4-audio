import numpy as np
import time

def my_fft(a):
  """
  Only support array with 2^k elements.
  """
  n = len(a)
  if n & (n - 1) != 0:
    raise ValueError('Only support array with 2^k elements.')

  def _my_fft(a, n, w):
    if n == 1:
      return a
    even = _my_fft(a[::2], n//2, w[::2])
    odd_w = _my_fft(a[1::2], n//2, w[::2]) * w[:n//2]
    return np.concatenate([even + odd_w, even - odd_w])

  w = np.exp(-2j * np.pi * np.arange(n) / n)
  return _my_fft(a, n, w)

# test code
for i in range(16):
  SAMPLE = 10

  a = [np.random.rand(2**i) for _ in range(SAMPLE)]

  my_time = []
  np_time = []
  allclose = []

  for j in range(SAMPLE):
    start = time.perf_counter()
    fft = my_fft(a[j])
    end = time.perf_counter()
    my_time.append(end - start)

    start = time.perf_counter()
    np_fft = np.fft.fft(a[j])
    end = time.perf_counter()
    np_time.append(end - start)

    allclose.append(np.allclose(fft, np_fft, atol=1e-10))

  my_time_avg = sum(my_time) / SAMPLE
  np_time_avg = sum(np_time) / SAMPLE
  print(f'2^{i} elements my_fft: {my_time_avg} sec')
  print(f'2^{i} elements np.fft: {np_time_avg} sec')
  print(f'diff: {my_time_avg - np_time_avg} sec')
  print(f'factor: {my_time_avg / np_time_avg}')
  print(''.join([('.' if i else 'F') for i in allclose]))
