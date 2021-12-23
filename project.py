import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.function_base import angle
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
    val = np.cos(100000*2*np.pi*x)*(5+m(x))
    return val


def AM_DSB_SC(x):
    val = 5*np.cos(100000*2*np.pi*x)*(m(x))
    return val


def FM(x, kf):
    val = 5*np.cos(100000*2*np.pi*x+kf*integration_m(x))
    return val


def PM(x, kp):
    val = 5*np.cos(100000*2*np.pi*x+kp*m(x))
    return val


def AM_SSB_SC(x):
    val = 5*1.5/2*np.cos(2*np.pi*(100000+2000)*x)
    return val


T = 1/1000
N = pow(10, 4)
x_values = np.linspace(0, T, N)
y_values = m(x_values)
fig1, ax1 = plt.subplots()
plt.plot(x_values, y_values, label="m(t)")
plt.title("the original signal in time domain")
plt.xlabel("time (s)")
plt.ylabel("amplitude (v)")
plt.legend()
plt.tight_layout()
plt.savefig('original_time.png')
######### large carrier #############

y_values_AM_DSB_LC = AM_DSB_LC(x_values)
fig2, ax2 = plt.subplots()
plt.plot(x_values, y_values_AM_DSB_LC, label="(m(t)+5)cos(wct)")
plt.title("large carrier in time domain")
plt.xlabel("time (s)")
plt.ylabel("amplitude (v)")
plt.legend()
plt.tight_layout()
plt.savefig('DSBLC_time.png')

######### suppressed carrier #############

y_values_AM_DSB_SC = AM_DSB_SC(x_values)
fig3, ax3 = plt.subplots()
plt.plot(x_values, y_values_AM_DSB_SC, label="m(t)cos(wct)")
plt.title("double side band suppressed carrier in time domain")
plt.xlabel("time (s)")
plt.ylabel("amplitude (v)")
plt.legend()
plt.tight_layout()
plt.savefig('DSBSC_time.png')

######### frequency modulation #############

y_values_FM = FM(x_values, 1000)
fig4, ax4 = plt.subplots()
plt.plot(x_values, y_values_FM, label="ac(cos(wct+kf/wc*sin(wct))")
plt.title("frequency modulation in time domain")
plt.xlabel("time (s)")
plt.ylabel("amplitude (v)")
plt.legend()
plt.tight_layout()
plt.savefig('FM_time.png')

######### phase modulation #############

y_values_PM = PM(x_values, 10)
fig5, ax5 = plt.subplots()
plt.plot(x_values, y_values_PM, label="ac(cos(wct+kp*sin(wct))")
plt.title("phase modulation in time domain")
plt.xlabel("time (s)")
plt.ylabel("amplitude (v)")
plt.legend()
plt.tight_layout()
plt.savefig('PM_time.png')

######### single side band modulation #############

y_values_AM_SSB_SC = AM_SSB_SC(x_values)
fig6, ax6 = plt.subplots()
plt.plot(x_values, y_values_AM_SSB_SC, label="ac*am/2*cos((wc+or-wm)t)")
plt.title("single side band in time domain")
plt.xlabel("time (s)")
plt.ylabel("amplitude (v)")
plt.legend()
plt.tight_layout()
plt.savefig('SSB_time.png')


# ########### frequency part #############

# ########### original signal ############

figf7, axf7 = plt.subplots()
sig_fft = fft(y_values)
amplitude1= np.abs(sig_fft)[:100]/10000
power=amplitude1**2
anglee=np.angle(sig_fft)
sampling_frequency1=fftfreq(N,T)[:100]
amplitude2=amplitude1[::-1]
amplitude =np.append(amplitude2,amplitude1)
sampling_frequency2=-1*sampling_frequency1[::-1]
sampling_frequency=np.append(sampling_frequency2,sampling_frequency1)
plt.plot(sampling_frequency,amplitude )
plt.title("the original signal in frequency domain")
plt.xlabel("frequency in KHz")
plt.legend('FFT original signal')
plt.tight_layout()
plt.savefig('original_frequency.png')

# ######### large carrier #############

figf8, axf8 = plt.subplots()

sig_fft = fft(y_values_AM_DSB_LC)
amplitude1= np.abs(sig_fft)[:200]/10000
power=amplitude1**2
anglee=np.angle(sig_fft)
sampling_frequency1=fftfreq(N,T)[:200]
amplitude2=amplitude1[::-1]
amplitude =np.append(amplitude2,amplitude1)
sampling_frequency2=-1*sampling_frequency1[::-1]
sampling_frequency=np.append(sampling_frequency2,sampling_frequency1)
plt.plot(sampling_frequency,amplitude )
plt.title("double side band large carrier in frequency domain")
plt.legend('FFT large carrier')
plt.tight_layout()
plt.savefig('DSBLC_frequency.png')

# ######### suppressed carrier #############

figf9, axf9 = plt.subplots()
sig_fft = fft(y_values_AM_DSB_SC)
amplitude1= np.abs(sig_fft)[:200]/10000
power=amplitude1**2
anglee=np.angle(sig_fft)
sampling_frequency1=fftfreq(N,T)[:200]
amplitude2=amplitude1[::-1]
amplitude =np.append(amplitude2,amplitude1)
sampling_frequency2=-1*sampling_frequency1[::-1]
sampling_frequency=np.append(sampling_frequency2,sampling_frequency1)
plt.plot(sampling_frequency,amplitude )
plt.title("double side band suppressed carrier in frequency domain")
plt.legend('FFT suppressed carrier')
plt.tight_layout()
plt.savefig('DSBSC_frequency.png')

# ######### frequency modulation #############
 
figf10, axf10 = plt.subplots()
sig_fft = fft(y_values_FM)
amplitude1= np.abs(sig_fft)[:200]/10000
power=amplitude1**2
anglee=np.angle(sig_fft)
sampling_frequency1=fftfreq(N,T)[:200]
amplitude2=amplitude1[::-1]
amplitude =np.append(amplitude2,amplitude1)
sampling_frequency2=-1*sampling_frequency1[::-1]
sampling_frequency=np.append(sampling_frequency2,sampling_frequency1)
plt.plot(sampling_frequency,amplitude )
plt.title("frequeency modulation in frequency domain")
plt.legend('FFT frequency modulation')
plt.tight_layout()
plt.savefig('FM_frequency.png')

# ######### phase modulation #############

figf11, axf11 = plt.subplots()
sig_fft = fft(y_values_PM)
amplitude1= np.abs(sig_fft)[:200]/10000
power=amplitude1**2
anglee=np.angle(sig_fft)
sampling_frequency1=fftfreq(N,T)[:200]
amplitude2=amplitude1[::-1]
amplitude =np.append(amplitude2,amplitude1)
sampling_frequency2=-1*sampling_frequency1[::-1]
sampling_frequency=np.append(sampling_frequency2,sampling_frequency1)
plt.plot(sampling_frequency,amplitude )
plt.title("phase modulation in frequency domain")
plt.legend('FFT phase modulation')
plt.tight_layout()
plt.savefig('PM_frequency.png')

# ######### single side band modulation #############

figf12, axf12 = plt.subplots()
sig_fft = fft(y_values_AM_SSB_SC)
amplitude1= np.abs(sig_fft)[:200]/10000
power=amplitude1**2
anglee=np.angle(sig_fft)
sampling_frequency1=fftfreq(N,T)[:200]
amplitude2=amplitude1[::-1]
amplitude =np.append(amplitude2,amplitude1)
sampling_frequency2=-1*sampling_frequency1[::-1]
sampling_frequency=np.append(sampling_frequency2,sampling_frequency1)
plt.plot(sampling_frequency,amplitude )
plt.title("single side band in frequency domain")
plt.legend('FFT single side band modulation')
plt.tight_layout()
plt.savefig('SSB_frequency.png')
plt.show()