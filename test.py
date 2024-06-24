import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np


# Parameters
x_min = -10.0
x_max = 10.0
N = 2**8

# Generate k and w
k = np.arange(N)
w = (0.5 - N / 2 + k) * (2 * np.pi / (x_max - x_min))

# Characteristic function
cffun = lambda w: np.exp(-0.5 * w**2)

# Characteristic exponent
cfexp = lambda w: -0.5 * w**2

# alpha=1
# nu=2
# cffun = lambda w: ( 1 -  (1j * w) / alpha) ** (-nu)

# cf = cffun(w[int(N/2):])
# cf = np.concatenate([np.conj(cf[::-1]), cf])

cf = cfexp(w[int(N/2):])
cf = np.concatenate([np.conj(cf[::-1]), cf])

# Compute dx, C, and D
dx = (x_max - x_min) / N

C = (-1+0j) ** ((1 - 1 / N) * (x_min / dx + k)) / (x_max - x_min)

D = (-1+0j) ** (-2 * (x_min / (x_max - x_min)) * k)

# Compute the PDF
pdf = np.real(C * np.fft.fft(D * cf))

# Compute the CDF
# cdf = np.cumsum(pdf * dx)

# Generate x values
x = x_min + k * dx

# Plot the PDF
plt.figure()
plt.plot(x, pdf, label='inverse CF')
plt.plot(x, norm.logpdf(x), label='scipy')
plt.title('PDF')
plt.grid()
plt.legend()
plt.show()

#Plot the CDF
# plt.figure()
# plt.plot(x, cdf)
# plt.title('CDF')
# plt.grid()
# plt.show()