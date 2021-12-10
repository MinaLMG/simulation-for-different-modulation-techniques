from scipy.signal import blackman
import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq
# https://www.youtube.com/watch?v=ufO_BScIHDQ


def m(t):
    x = 1.5*np.cos(2000*t*2*np.pi)
    return x

def integration_m(t):
    x = 1.5*np.sin(2000*t*2*np.pi)/(2*np.pi*2000)
    return x

def m_plus_val(x):
    return 5*(3+x)


def neg_m_plus_val(x):
    return -5*(3+x)


def AM_DSB_LC(x):
    val = 5*np.cos(100000*2*np.pi*x)*(3+m(x))
    return val

def AM_DSB_SC(x):
    val = 5*np.cos(100000*2*np.pi*x)*(m(x))
    return val

def FM(x,kf):
    val = 5*np.cos(100000*2*np.pi*x+kf*integration_m(x))
    return val

def PM(x,kp):
    val = 5*np.cos(100000*2*np.pi*x+kp*m(x))
    return val

def AM_SSB_SC(x):
    val = 5*1.5/2*np.cos(2*np.pi*(100000+2000)*x)
    return val

N = pow(10,3)
T = 1/1000/N
x_values = np.linspace(0,1/1000, N)
y_values = m(x_values)
fig1, ax1 = plt.subplots()
plt.plot(x_values, y_values, label="m(t)")
plt.title("functions in time domain")
plt.xlabel("time (s)")
plt.ylabel("amplitude (v)")
plt.legend()
plt.tight_layout()

######### large carrier #############

y_values_AM_DSB_LC = AM_DSB_LC(x_values)
fig2, ax2 = plt.subplots()
plt.plot(x_values, y_values_AM_DSB_LC, label="(m(t)+3)cos(wct)")
plt.title("functions in time domain")
plt.xlabel("time (s)")
plt.ylabel("amplitude (v)")
plt.legend()
plt.tight_layout()
plt.plot(x_values,y_values_AM_DSB_LC,label="m(t)AM_DSB_LC")

######### suppressed carrier #############

y_values_AM_DSB_SC = AM_DSB_SC(x_values)
fig3, ax3 = plt.subplots()
plt.plot(x_values, y_values_AM_DSB_SC, label="m(t)cos(wct)")
plt.title("functions in time domain")
plt.xlabel("time (s)")
plt.ylabel("amplitude (v)")
plt.legend()
plt.tight_layout()
plt.plot(x_values,y_values_AM_DSB_SC,label="m(t)AM_DSB_SC")

######### frequency modulation #############

y_values_FM = FM(x_values,1000)
fig4, ax4 = plt.subplots()
plt.plot(x_values, y_values_FM, label="ac(cos(wct+kf/wc*sin(wct))")
plt.title("functions in time domain")
plt.xlabel("time (s)")
plt.ylabel("amplitude (v)")
plt.legend()
plt.tight_layout()
plt.plot(x_values,y_values_FM,label="m(t)FM")

######### phase modulation #############

y_values_PM = PM(x_values,10)
fig5, ax5 = plt.subplots()
plt.plot(x_values, y_values_PM, label="ac(cos(wct+kp*sin(wct))")
plt.title("functions in time domain")
plt.xlabel("time (s)")
plt.ylabel("amplitude (v)")
plt.legend()
plt.tight_layout()
plt.plot(x_values,y_values_PM,label="m(t)PM")

######### single side band modulation #############

y_values_AM_SSB_SC = AM_SSB_SC(x_values)
fig6, ax6 = plt.subplots()
plt.plot(x_values, y_values_AM_SSB_SC, label="ac*am/2*cos((wc+or-wm)t)")
plt.title("functions in time domain")
plt.xlabel("time (s)")
plt.ylabel("amplitude (v)")
plt.legend()
plt.tight_layout()
plt.plot(x_values,y_values_AM_SSB_SC,label="m(t)SSB_SC")



########### frequency part #############

########### original signal ############
figf1, axf1 = plt.subplots()
y_values_f = fft(y_values)
w = blackman(N)
ywf = fft(y_values*w)
xf = fftfreq(N, 1/1000/N)[:N//2]
plt.semilogy(xf[1:N//2], 2.0/N * np.abs(y_values_f[1:N//2]), '-b')
# plt.semilogy(xf[1:N//2], 2.0/N * np.abs(ywf[1:N//2]), '-r')
plt.legend(['FFT original signal'])
plt.show()

######### large carrier #############

figf2, axf2 = plt.subplots()
y_values_f = fft(y_values_AM_DSB_LC)
w = blackman(N)
ywf = fft(y_values_AM_DSB_LC*w)
xf = fftfreq(N, 1/1000/N)[:N//2]
plt.semilogy(xf[1:N//2], 2.0/N * np.abs(y_values_f[1:N//2]), '-b')
plt.legend(['FFT large carrier'])
plt.show()

######### suppressed carrier #############

figf3, axf3 = plt.subplots()
y_values_f = fft(y_values_AM_DSB_SC)
w = blackman(N)
ywf = fft(y_values_AM_DSB_SC*w)
xf = fftfreq(N, 1/1000/N)[:N//2]
plt.semilogy(xf[1:N//2], 2.0/N * np.abs(y_values_f[1:N//2]), '-b')
plt.legend(['FFT suppressed carrier'])
plt.show()

######### frequency modulation #############

figf4, axf4 = plt.subplots()
y_values_f = fft(y_values_FM)
w = blackman(N)
ywf = fft(y_values_FM*w)
xf = fftfreq(N, 1/1000/N)[:N//2]
plt.semilogy(xf[1:N//2], 2.0/N * np.abs(y_values_f[1:N//2]), '-b')
plt.legend(['FFT frequency modulation'])
plt.show()

######### phase modulation #############

figf5, axf5 = plt.subplots()
y_values_f = fft(y_values_PM)
w = blackman(N)
ywf = fft(y_values_PM*w)
xf = fftfreq(N, 1/1000/N)[:N//2]
plt.semilogy(xf[1:N//2], 2.0/N * np.abs(y_values_f[1:N//2]), '-b')
plt.legend(['FFT phase modulation'])
plt.show()

######### single side band modulation #############

figf6, axf6 = plt.subplots()
y_values_f = fft(y_values_AM_SSB_SC)
w = blackman(N)
ywf = fft(y_values_AM_SSB_SC*w)
xf = fftfreq(N, 1/1000/N)[:N//2]
plt.semilogy(xf[1:N//2], 2.0/N * np.abs(y_values_f[1:N//2]), '-b')
plt.legend(['FFT single side band modulation'])
plt.show()
