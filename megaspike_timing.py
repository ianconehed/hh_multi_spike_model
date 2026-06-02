"""
Paired-pulse protocol: P(megaspike) vs. axon-dendrite input timing.

Single axonal and dendritic inputs are paired at a range of delays.
For each delay we run several noisy trials and measure the soma spike
amplitude; a "megaspike" is any spike whose amplitude exceeds 30 mV.
The script plots P(megaspike) as a function of the axon->dendrite delay.

Self-contained: only the model in hh_annie_model_june_1_final.py is reused
(copied here). Nothing else from the repo is imported.
"""

from brian2 import *
from scipy.signal import find_peaks

#%% Brian2 settings
defaultclock.dt = 1*ms
rng = np.random.default_rng()
dt_ms = float(defaultclock.dt / ms)

#%% Morphology
cyl_len = 200
morpho = Soma(diameter = 3*um)
morpho.trunk = Cylinder(length=70*um, diameter=.25*um, n=70, type='trunk')
morpho.trunk.axon = Cylinder(length=70*um, diameter=.15*um, n=70, type='axon')
morpho.trunk.axon.distal = Cylinder(length=130*um, diameter=.15*um, n=130, type='axon')
morpho.trunk.axon.collateral = Cylinder(length=200*um, diameter=.15*um, n=cyl_len, type='collateral')
morpho.trunk.dendrite = Cylinder(length = 200*um, diameter=.15*um, n=cyl_len, type='dendrite')
morpho = morpho.generate_coordinates()

soma_idx = 0
trunk_start_idx = 1
axon_start_idx = 71
axon_distal_start_idx = 141
collateral_start_idx = 271
dendrite_start_idx = 471

#%% Parameters
VLk = -40*mV
VNa = 50*mV
VK = -77*mV
VCa = 120*mV
VCAN = 0*mV

Rm = 10*kohm*cm**2
Ra = 25*ohm*cm
Cm0 = 10*uF/cm**2

gLk0 = 1.4*msiemens/cm**2
gNa0 = 80*msiemens/cm**2
gK0 = 50*msiemens/cm**2
gCa0 = .1*msiemens/cm**2
gSK0 = 8*msiemens/cm**2
gCAN0 = 12*msiemens/cm**2

Vhalfm = -23*mV
a_m = .085/mV
taum = 40*ms

Vhalfh = -30*mV
a_h = .05/mV
tauh = 60*ms

Vhalfn = -23*mV
a_n = .06/mV
taun = 150*ms

Vhalfp = -20*mV
a_p = 0.1/mV
taup = 100*ms

tau_Ca = 1000*ms
alpha_Ca = 0.005*mmolar/(amp/meter**2*ms)

CahalfSK = .02*mmolar
a_SK = 400/mmolar
tau_SK = 500*ms

CahalfCAN = .02*mmolar
a_CAN = 400/mmolar
tau_r = 2000*ms

sigma_noise = 1*pA
tau_noise = 300*ms

#%% Equations
eqs = '''
ILk = gLk * (v - VLk) : amp/meter**2
INa = gNa * m**3 * h * (v - VNa) : amp/meter**2
IK = gK * n**4 * (v - VK) : amp/meter**2
ICa = gCa * p**2 * (v - VCa) : amp/meter**2
ISK = gSK * s * (v - VK) : amp/meter**2
ICAN = gCAN * r * (v - VCAN) : amp/meter**2

Im = -ILk -INa - IK - ICa - ISK - ICAN : amp/meter**2

I_dend = dend_mask * dend_stim(t) : amp (point current)
I_axon = axon_mask * axon_stim(t) : amp (point current)
I_noise = noise_mask * I_ou : amp (point current)

dend_mask : 1
axon_mask : 1
noise_mask : 1

m_inf = 1/(1 + exp(-2*a_m*(v-Vhalfm))) : 1
h_inf = 1/(1 + exp(2*a_h*(v-Vhalfh))) : 1
n_inf = 1/(1 + exp(-2*a_n*(v-Vhalfn))) : 1
p_inf = 1/(1 + exp(-2*a_p*(v-Vhalfp))) : 1
s_inf = 1/(1 + exp(-2*a_SK*(cai-CahalfSK))) : 1
r_inf = 1/(1 + exp(-2*a_CAN*(cai-CahalfCAN))) : 1

dn/dt = (n_inf - n)/taun : 1
dh/dt = (h_inf - h)/tauh : 1
dm/dt = (m_inf - m)/taum : 1
dp/dt = (p_inf - p)/taup : 1
ds/dt = (s_inf - s)/tau_SK : 1
dr/dt = (r_inf - r)/tau_r : 1

dcai/dt = -alpha_Ca * ICa - cai/tau_Ca : mmolar
dI_ou/dt = -I_ou/tau_noise + sigma_noise*sqrt(2/tau_noise)*xi : amp

gLk : siemens/meter**2
gNa : siemens/meter**2
gK : siemens/meter**2
gCa : siemens/meter**2
gSK : siemens/meter**2
gCAN : siemens/meter**2
'''

#%% Geometry of the spike-initiation / input zones (same as base model)
axon_proximal_len = 70*um

branch_siz_start = 0*um
branch_siz_end = 20*um

dend_siz_start = 85*um
dend_siz_end   = 175*um
dend_siz_center = dend_siz_start + abs(dend_siz_start-dend_siz_end)/2

axon_siz_start = 90*um
axon_siz_end   = 130*um
axon_siz_center = axon_siz_start + abs(axon_siz_start-axon_siz_end)/2

dend_input_start = dend_siz_center - 2.5*um
dend_input_end   = dend_siz_center + 2.5*um
axon_input_start = axon_siz_center - 2.5*um
axon_input_end   = axon_siz_center + 2.5*um


def build_neuron():
    """Construct the SpatialNeuron with the base-model conductance layout."""
    neuron = SpatialNeuron(morphology=morpho, model=eqs, method="heun",
                           Cm=Cm0, Ri=Ra)

    # Conductance distribution
    neuron.gLk = gLk0
    neuron.gNa = .05*gNa0
    neuron.gK = .05*gK0
    neuron.gCa = gCa0
    neuron.gSK = 0*gSK0
    neuron.gCAN = 0

    neuron.trunk.axon.collateral.gNa = 0.00*gNa0
    neuron.trunk.axon.collateral.gK = 0.00*gK0

    neuron.trunk.axon.distal[0*um:axon_proximal_len].gSK = .25*gSK0
    neuron.trunk.dendrite[0*um:dend_siz_start].gSK = .25*gSK0
    neuron.trunk.axon.collateral.gSK = 0*gSK0

    neuron.trunk.axon.distal[axon_siz_start - axon_proximal_len:axon_siz_end - axon_proximal_len].gNa = .75*gNa0
    neuron.trunk.axon.distal[axon_siz_start - axon_proximal_len:axon_siz_end - axon_proximal_len].gK = .75*gK0
    neuron.trunk.axon.distal[axon_siz_start - axon_proximal_len:axon_siz_end - axon_proximal_len].gSK = 0*gSK0
    neuron.trunk.dendrite[dend_siz_start:dend_siz_end].gNa = .5*gNa0
    neuron.trunk.dendrite[dend_siz_start:dend_siz_end].gK = .5*gK0
    neuron.trunk.dendrite[dend_siz_start:dend_siz_end].gSK = 0*gSK0

    neuron.trunk.axon[branch_siz_start:branch_siz_end].gNa = 1.5*gNa0
    neuron.trunk.axon[branch_siz_start:branch_siz_end].gK = 1.5*gK0
    neuron.trunk.axon[branch_siz_start:branch_siz_end].gSK = 1.5*gSK0
    neuron.trunk.axon[branch_siz_start:branch_siz_end].gCAN = 1*gCAN0

    # Initial conditions
    neuron.v = -40*mV
    neuron.m = neuron.m_inf[:]
    neuron.n = neuron.n_inf[:]
    neuron.h = neuron.h_inf[:]
    neuron.p = neuron.p_inf[:]
    neuron.r = neuron.r_inf[:]
    neuron.s = neuron.s_inf[:]
    neuron.cai = 0*mmolar

    # Input / noise masks (single dendritic site, single axonal site)
    neuron.dend_mask = 0
    neuron.trunk.dendrite[dend_input_start:dend_input_end].dend_mask = 1
    neuron.axon_mask = 0
    neuron.trunk.axon.distal[axon_input_start - axon_proximal_len:axon_input_end - axon_proximal_len].axon_mask = 1
    neuron.noise_mask = 0
    neuron[0].noise_mask = 1  # soma compartment only

    return neuron


def make_pulse_array(onset_ms, total_steps, pulse_dur_ms, amp_pA):
    """Return a step-current array with a single pulse starting at onset_ms."""
    arr = np.zeros(total_steps)
    i0 = int(onset_ms / dt_ms)
    i1 = int((onset_ms + pulse_dur_ms) / dt_ms)
    arr[i0:i1] = amp_pA
    return arr


#%% Protocol parameters
warmup_ms     = 500     # settle before the paired pulse
response_ms   = 1000    # observation window after the pulses
pulse_dur     = 50      # ms, single-pulse duration (per input)
stim_amp      = 5       # pA, per-input amplitude
megaspike_mV  = 25      # spike-amplitude threshold for a "megaspike"
n_trials      = 15      # noisy repeats per delay

# Delay = t_dend - t_axon (positive => dendrite lags axon)
delays_ms = np.arange(-300, 301, 10)

total_ms    = warmup_ms + response_ms
total_steps = int(total_ms / dt_ms)
sim_time    = total_ms * ms

p_megaspike   = np.zeros(len(delays_ms))
sem_megaspike = np.zeros(len(delays_ms))
mean_amp      = np.zeros(len(delays_ms))

#%% Run the protocol
for di, delay in enumerate(delays_ms):
    start_scope()

    t_axon = warmup_ms
    t_dend = warmup_ms + delay
    t_first = min(t_axon, t_dend)   # earlier of the two inputs (anchors the windows)

    axon_arr = make_pulse_array(t_axon, total_steps, pulse_dur, stim_amp)
    dend_arr = make_pulse_array(t_dend, total_steps, pulse_dur, stim_amp)
    axon_stim = TimedArray(axon_arr * pA, dt=defaultclock.dt)
    dend_stim = TimedArray(dend_arr * pA, dt=defaultclock.dt)

    neuron = build_neuron()
    mon = StateMonitor(neuron, 'v', record=[0])  # soma only

    store()

    hits = 0
    amps = []
    hit_flags = []   # per-trial binary megaspike outcome (for the std band)
    for trial in range(n_trials):
        restore()
        seed(1000 * di + trial)   # independent noise per trial
        run(sim_time)

        v_soma = mon.v[0] / mV
        t_ms = mon.t / ms
        # Baseline from a guaranteed-quiet window just before the first input,
        # so neither pulse (axon or dendrite) can contaminate it at any delay.
        baseline = np.mean(v_soma[(t_ms > t_first - 100) & (t_ms < t_first)])

        # Response window starts at the first input, so a dendrite-led
        # response is captured for negative delays too.
        resp = v_soma[t_ms >= t_first]
        peaks, props = find_peaks(resp, height=baseline + 5, prominence=5)
        if len(peaks):
            amp = props['peak_heights'].max() - baseline
        else:
            amp = 0.0   # no spike => amplitude 0, not the (sub-baseline) tail max
        amps.append(amp)
        is_mega = amp > megaspike_mV
        hit_flags.append(is_mega)
        if is_mega:
            hits += 1

    p_megaspike[di] = hits / n_trials
    sem_megaspike[di] = np.std(hit_flags) / np.sqrt(n_trials)   # standard error of P
    mean_amp[di] = np.mean(amps)
    print(f"delay {delay:+4d} ms:  P(megaspike) = {p_megaspike[di]:.2f}   "
          f"mean amp = {mean_amp[di]:.1f} mV")

#%% Plot P(megaspike) vs delay
fig, ax = plt.subplots(figsize=(5, 4))
ax.fill_between(delays_ms,
                np.clip(p_megaspike - sem_megaspike, 0, 1),
                np.clip(p_megaspike + sem_megaspike, 0, 1),
                color='steelblue', alpha=0.2, lw=0, label='±1 SEM')
ax.plot(delays_ms, p_megaspike, '-', color='steelblue', lw=1.5, ms=7)
ax.axvline(0, color='gray', ls='--', lw=0.8)
ax.set_xlabel('axon→dendrite delay (ms)')
ax.set_ylabel('P(megaspike)   (amp > %d mV)' % megaspike_mV)
ax.set_ylim(-0.05, 1.05)
ax.set_title('Megaspike probability vs. input timing')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig('megaspike_vs_timing.svg', format='svg', bbox_inches='tight')

#%% Plot mean spike amplitude vs delay (companion view)
fig2, ax2 = plt.subplots(figsize=(5, 4))
ax2.plot(delays_ms, mean_amp, 's-', color='tomato', lw=1.5, ms=7)
ax2.axhline(megaspike_mV, color='gray', ls='--', lw=0.8, label='megaspike threshold')
ax2.axvline(0, color='gray', ls=':', lw=0.8)
ax2.set_xlabel('axon→dendrite delay (ms)')
ax2.set_ylabel('mean spike amplitude (mV)')
ax2.set_title('Spike amplitude vs. input timing')
ax2.legend(fontsize=8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
fig2.tight_layout()

plt.show()
