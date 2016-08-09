#%%

import numpy
import matplotlib.pyplot as plt

y = 1.0
D = 0.1

N = 128

dt = 0.1
dx = 1

u = numpy.sin(numpy.array(range(0, N)) * 0.1) * 0.5 + numpy.random.rand(N) - 0.5#signal ###numpy.zeros(N)
u2 = numpy.fft.fft(u)

w = 2.0 * numpy.pi * numpy.fft.fftfreq(N, 1.0)
w2 = w * w

plt.plot(numpy.real(numpy.fft.ifft(u2)))
plt.show()

for t in range(0, 3001):
    u3 = numpy.real(numpy.fft.ifft(u2))
    fftlap = numpy.fft.fft(u3 * u3 * u3)
    u2 = u2 / (1 + D * w2 * dt * (-1 + y * w2))
    u2 = u2 - D * w2 * dt * fftlap

    if (t) % 100 == 0 or t == 0:
        plt.title("t = {0}".format(t))
        plt.plot(numpy.real(numpy.fft.ifft(u2)))
        plt.ylim((-1.0, 1.0))
        plt.show()

#plt.plot()
u4 = numpy.real(numpy.fft.ifft(u2))

#%%
import seaborn as sns
import scipy

dy = numpy.real(numpy.fft.ifft(u2))

sns.distplot(dy[:-1] - dy[1:], bins = 'auto', fit = scipy.stats.cauchy, kde = False)
plt.show()
#%%
H = scipy.linalg.expm(numpy.random.randn(100, 100))
print H
scipy.linalg.eig(H)[0]