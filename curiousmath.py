import numpy as np
import matplotlib.pyplot as plt


# camelCase is on recieving/modifable end, these_vars are sim vars, ALL_CAPS are constants

# Assume pulse occurs at t=0; Stand in for either detecting a frequency shift or inital pulse

# NOTE Lower frequencies have a longer detection time thus reduce resolution; Higher frequencies are better

SIMULAITON_LENGTH = 5 # s
SPEED_OF_SOUND = 1 # 343 #m/s
SAMPLE_RATE_ADC = 9600 # NOTE Ardunio using analogRead() but can use hax to icnrease it ot 38.5kHz or buy/user better ADCs
MIC_SAMPLES = int(SIMULAITON_LENGTH*SAMPLE_RATE_ADC)
simulation_times = np.linspace(0, SIMULAITON_LENGTH, num=MIC_SAMPLES)

# Generate Pulse Signal
PULSE_LENGTH = 0.5 # s
PULSE_SAMPLES = int(SAMPLE_RATE_ADC*PULSE_LENGTH)
pulse_times = np.linspace(0, PULSE_LENGTH, num=PULSE_SAMPLES)

# TODO PULSE FREQUENCY MUST GRADUALLY CHANGE ELSE SPECTRAL SPLATTER OCCURS
pulse_signal = np.zeros(PULSE_SAMPLES)
pulse_signal[0:PULSE_SAMPLES//2] = np.sin(2*np.pi*400*pulse_times[0:PULSE_SAMPLES//2])
pulse_signal[PULSE_SAMPLES//2:] = np.sin(2*np.pi*800*pulse_times[PULSE_SAMPLES//2:])

soundPos = np.array([0,3,0]).reshape(3,1) # X, Y, Z
micList = []

class Mic:
    def __init__(self, pos):
        self.pos = np.array(pos).reshape(3,1) # X, Y, Z

        micList.append(self)

    def sampleList(self):
        results = np.zeros(shape=(MIC_SAMPLES,))

        # May need to make this more complicated later to include multipathing, etc, etc based on room or use more adv. sim
        # Also note that ceil makes each read of the pulse signal start perfectly with the pulse instead of random phase offsets due to when the first sample is taken realitive to immediatly after the wave gets there
        sampleStart = int(np.ceil(SAMPLE_RATE_ADC * np.sqrt(np.linalg.norm(self.pos - soundPos)) / SPEED_OF_SOUND))
        length = min(MIC_SAMPLES-sampleStart, PULSE_SAMPLES)
        results[sampleStart:sampleStart+length] = pulse_signal[0:length]

        # Noise
        results = results + np.random.normal(0, 0.01, MIC_SAMPLES)

        return results

Mic([1,0,0])
Mic([-1,0,0])
Mic([0,1,0])
Mic([0,-1,0])
Mic([0,0,1])
Mic([0,0,-1])
NUM_MICS = len(micList)

samples = [m.sampleList() for m in micList]

# fMax = fS/2; deltaF (Resolution) = fs/N = 1/T
MAXIMUMMEASUREABLEFREQUENCY = SAMPLE_RATE_ADC / 2 # Nyquilst criteria
LOWESTMEASURABLEFREQUENCY = 200 # Sets frame window
SNRTHRESHOLD = 25 # dB

SamplesPerWindow = int(np.ceil(5 * SAMPLE_RATE_ADC / LOWESTMEASURABLEFREQUENCY)) # 5 cycles of lowest frequency set max number of samples needed for detection
SampleWindowPeriod = SamplesPerWindow/SAMPLE_RATE_ADC # NOTE: This also determines the minimum transmission pulse length


sampleWindow = np.zeros((NUM_MICS, SamplesPerWindow))

freqs = np.fft.rfftfreq(SamplesPerWindow, d=(1/SAMPLE_RATE_ADC)) # Get bins; Samples, timestep

frequenciesVsTime = np.zeros((MIC_SAMPLES, NUM_MICS, len(freqs))) # Spectrum of each mic per timestep
detectedFrequenciesVsTime = np.zeros((MIC_SAMPLES, NUM_MICS, len(freqs))) # Peak Frequency (#NOTE IRL may need to look specifcally at broadcast frequencies (discrete bandpass) to ensure interference)

# Detection Loop (What runs on the hardware)
for i, t in enumerate(simulation_times):
    # One time step is whenever the controller has a new sample from each mic; in a realsitic case the ADC would need to be faster and then the number would be reduced to reperesent overhead via the calculations or other optimizations could be used
    sampleWindow[:, :-1] = sampleWindow[:, 1:]
    for j, sampleArr in enumerate(samples):
        sampleWindow[j, -1] = sampleArr[i]

    freqMag = np.fft.rfft(sampleWindow, axis=1)
    frequenciesVsTime[i] = data = 20*np.log10(np.maximum(np.abs(freqMag), 1e-12) / SamplesPerWindow)
    noise = np.median(data, axis=1).reshape((data.shape[0],1))

    SNR = data - noise
    detectedFrequenciesVsTime[i] = np.where(SNR > SNRTHRESHOLD, SNR, 0)



colors = ["red", "green", "yellow", "purple", "blue", "orange"]

figPulse = plt.figure("Pulse")
sub = figPulse.add_subplot(2, 1, 1)
sub.plot(pulse_times, pulse_signal)
sub.set_xlabel("Pulse Time")
sub.set_ylabel("Amplitude")
sub.set_title("Pulse Signal")

sub = figPulse.add_subplot(2, 1, 2)
freqsPulse = np.fft.fftfreq(PULSE_SAMPLES, d=(1/SAMPLE_RATE_ADC)) # Get bins; Samples, timestep
mags = np.fft.fft(pulse_signal)
sub.plot(freqsPulse, np.abs(mags))
sub.set_xlabel("Frequency")
sub.set_ylabel("Amplitude")
sub.set_title("FFT of Pulse")

figPulse.tight_layout()

figRecSignals = plt.figure("Recieved")
sub = figRecSignals.add_subplot(3, 1, 1)
for i, s in enumerate(samples):
    sub.plot(simulation_times, s, color=colors[i])
sub.set_title("Signal Ampltidude vs Time of Microphones")
sub.set_xlabel("Simulation Time")
sub.set_ylabel("Amplitude")

sub = figRecSignals.add_subplot(3, 1, 2)
for i in range(NUM_MICS):
    data = detectedFrequenciesVsTime[:, i]
    data = freqs[np.argmax(data, axis=1)]


    sub.plot(simulation_times, data, color=colors[i])

sub.set_xlabel("Simulation Time")
sub.set_ylabel("Frequency")
sub.set_title("Peak SNR vs Time of Microphones")

sub = figRecSignals.add_subplot(3, 1, 3)
for i in range(NUM_MICS):
    sub.plot(simulation_times, frequenciesVsTime[:, i], color=colors[i])

sub.set_xlabel("Simulation Time")
sub.set_ylabel("Amplitude (dB)")
sub.set_title("Amplitude of Mics vs Time")

figRecSignals.tight_layout()


figFTMic1 = plt.figure("FT of Mic 1 vs Time")
sub = figFTMic1.add_subplot(1, 1, 1, projection='3d')

mask = (freqs >= 0) & (freqs <= 1000)
T, F = np.meshgrid(simulation_times ,freqs[mask], indexing='ij')

sub.plot_surface(T, F, frequenciesVsTime[:, 1, mask])

figFTMic1.tight_layout()

plt.show()