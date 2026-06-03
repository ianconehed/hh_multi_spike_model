"""
Recovery protocol: megaspike susceptibility vs. time since a prior megaspike.

This removes the confound in megaspike_priming.py, where the axon->dendrite
delay was tangled up with the (slowly changing) SK/CAN state. Here there is NO
pairing: a single conditioning megaspike (two coincident axon+dendrite inputs)
is fired once, and then a SINGLE dendritic test input is delivered at a variable
gap afterwards. We sweep that gap and measure P(megaspike) of the test response.

So the x-axis is literally "time since the conditioning megaspike", and the
curve is a direct readout of how the residual CAN - SK drive gates a lone input.
A control sweep (identical single test input, but with NO conditioning megaspike)
gives the flat baseline: a single dendritic input on its own rarely megaspikes.

Reuses only the model from hh_annie_model_june_1_final.py (copied here).
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
axon_start_idx = 71
axon_distal_start_idx = 141
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

    neuron.trunk.axon.distal[axon_siz_start - axon_proximal_len:axon_siz_end - axon_proximal_len].gNa = .6*gNa0
    neuron.trunk.axon.distal[axon_siz_start - axon_proximal_len:axon_siz_end - axon_proximal_len].gK = .6*gK0
    neuron.trunk.axon.distal[axon_siz_start - axon_proximal_len:axon_siz_end - axon_proximal_len].gSK = 0*gSK0
    neuron.trunk.dendrite[dend_siz_start:dend_siz_end].gNa = .5*gNa0
    neuron.trunk.dendrite[dend_siz_start:dend_siz_end].gK = .5*gK0
    neuron.trunk.dendrite[dend_siz_start:dend_siz_end].gSK = 0*gSK0

    neuron.trunk.axon[branch_siz_start:branch_siz_end].gNa = 1.5*gNa0
    neuron.trunk.axon[branch_siz_start:branch_siz_end].gK = 1.5*gK0
    neuron.trunk.axon[branch_siz_start:branch_siz_end].gSK = 1.5*gSK0
    neuron.trunk.axon[branch_siz_start:branch_siz_end].gCAN = 1.0*gCAN0

    neuron.v = -40*mV
    neuron.m = neuron.m_inf[:]
    neuron.n = neuron.n_inf[:]
    neuron.h = neuron.h_inf[:]
    neuron.p = neuron.p_inf[:]
    neuron.r = neuron.r_inf[:]
    neuron.s = neuron.s_inf[:]
    neuron.cai = 0*mmolar

    neuron.dend_mask = 0
    neuron.trunk.dendrite[dend_input_start:dend_input_end].dend_mask = 1
    neuron.axon_mask = 0
    neuron.trunk.axon.distal[axon_input_start - axon_proximal_len:axon_input_end - axon_proximal_len].axon_mask = 1
    neuron.noise_mask = 0
    neuron[0].noise_mask = 1  # soma compartment only

    return neuron


def add_pulse(arr, onset_ms, pulse_dur_ms, amp_pA):
    """Add a single step pulse (in place) starting at onset_ms."""
    i0 = int(onset_ms / dt_ms)
    i1 = int((onset_ms + pulse_dur_ms) / dt_ms)
    arr[i0:i1] = amp_pA
    return arr


#%% Protocol parameters
warmup_ms     = 500     # settle before the conditioning megaspike
response_ms   = 1000    # observation window after the test input
pulse_dur     = 50      # ms, single-pulse duration
stim_amp      = 5       # pA, per-input amplitude
megaspike_mV  = 25      # spike-amplitude threshold for a "megaspike"
n_trials      = 8       # noisy repeats per gap

t_cond = warmup_ms      # the single conditioning megaspike (coincident axon+dend)

# x-axis: time of the single dendritic test input AFTER the conditioning megaspike
gaps_ms = np.arange(200, 10001, 200)


def run_recovery(primed):
    """Sweep the gap; return P, SEM, mean amp, and per-trial amps.

    primed=True  : a conditioning megaspike precedes the single test input.
    primed=False : the single test input alone (no conditioner) -> flat baseline.
    """
    p_mega     = np.zeros(len(gaps_ms))
    sem_mega   = np.zeros(len(gaps_ms))
    mean_amp   = np.zeros(len(gaps_ms))
    amp_trials = np.zeros((len(gaps_ms), n_trials))

    for gi, gap in enumerate(gaps_ms):
        start_scope()

        t_test = t_cond + gap
        total_ms = t_test + response_ms
        total_steps = int(total_ms / dt_ms)

        axon_arr = np.zeros(total_steps)
        dend_arr = np.zeros(total_steps)
        # Single dendritic test input.
        add_pulse(dend_arr, t_test, pulse_dur, stim_amp)
        # Conditioning megaspike: two coincident inputs at t_cond.
        if primed:
            add_pulse(axon_arr, t_cond, pulse_dur, stim_amp)
            add_pulse(dend_arr, t_cond, pulse_dur, stim_amp)

        axon_stim = TimedArray(axon_arr * pA, dt=defaultclock.dt)
        dend_stim = TimedArray(dend_arr * pA, dt=defaultclock.dt)

        neuron = build_neuron()
        mon = StateMonitor(neuron, 'v', record=[0])
        store()

        hits = 0
        amps = []
        hit_flags = []
        for trial in range(n_trials):
            restore()
            seed(1000 * gi + trial + (500000 if primed else 0))
            run(total_ms * ms)

            v_soma = mon.v[0] / mV
            t_ms = mon.t / ms
            # Resting baseline from before the conditioner (common reference).
            baseline = np.mean(v_soma[(t_ms > t_cond - 100) & (t_ms < t_cond)])

            # Response window: only the test input (the conditioner is long past).
            resp = v_soma[t_ms >= t_test]
            peaks, props = find_peaks(resp, height=baseline + 5, prominence=5)
            amp = props['peak_heights'].max() - baseline if len(peaks) else 0.0
            amps.append(amp)
            is_mega = amp > megaspike_mV
            hit_flags.append(is_mega)
            if is_mega:
                hits += 1

        p_mega[gi]     = hits / n_trials
        sem_mega[gi]   = np.std(hit_flags) / np.sqrt(n_trials)
        mean_amp[gi]   = np.mean(amps)
        amp_trials[gi] = amps
        tag = 'primed ' if primed else 'control'
        print(f"[{tag}] gap {gap:+5d} ms:  P(megaspike) = {p_mega[gi]:.2f}   "
              f"mean amp = {mean_amp[gi]:.1f} mV")

    return p_mega, sem_mega, mean_amp, amp_trials


#%% Run both conditions
print("=== primed: single test input after a conditioning megaspike ===")
p_prime, sem_prime, amp_prime, at_prime = run_recovery(primed=True)
print("=== control: single test input, no conditioner ===")
p_ctrl, sem_ctrl, amp_ctrl, at_ctrl = run_recovery(primed=False)

#%% Plot P(megaspike) vs time since megaspike
fig, ax = plt.subplots(figsize=(6, 4))
for p, sem, color, label in [
        (p_prime, sem_prime, 'tomato',    'primed (single input after megaspike)'),
        (p_ctrl,  sem_ctrl,  'steelblue', 'control (single input, no megaspike)')]:
    ax.fill_between(gaps_ms, np.clip(p - sem, 0, 1), np.clip(p + sem, 0, 1),
                    color=color, alpha=0.2, lw=0)
    ax.plot(gaps_ms, p, '-', color=color, lw=1.5, label=label)
ax.set_xlabel('time since conditioning megaspike (ms)')
ax.set_ylabel('P(megaspike)   (amp > %d mV)' % megaspike_mV)
ax.set_ylim(-0.05, 1.05)
ax.set_title('Megaspike susceptibility of a single input\nvs. time since a prior megaspike')
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig('megaspike_recovery.svg', format='svg', bbox_inches='tight')

#%% Per-trial amplitude scatter
fig2, ax2 = plt.subplots(figsize=(8, 4))
step = float(np.min(np.diff(gaps_ms))) if len(gaps_ms) > 1 else 200.0
half = step * 0.18
for at, color, sign, label in [
        (at_prime, 'tomato',    +1, 'primed'),
        (at_ctrl,  'steelblue', -1, 'control')]:
    for gi, gap in enumerate(gaps_ms):
        y = at[gi]
        x = gap + sign * half + rng.uniform(-half * 0.7, half * 0.7, size=len(y))
        ax2.plot(x, y, 'o', mfc='none', mec=color, mew=1.0, ms=5, alpha=0.6)
    ax2.plot(gaps_ms + sign * half, at.mean(axis=1), '-', color=color, lw=1.5, label=label)
ax2.axhline(megaspike_mV, color='gray', ls='--', lw=0.8, label='megaspike threshold')
ax2.set_xlabel('time since conditioning megaspike (ms)')
ax2.set_ylabel('per-trial spike amplitude (mV)')
ax2.set_title('Per-trial test-input amplitude vs. time since a prior megaspike')
ax2.legend(fontsize=8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
fig2.tight_layout()
fig2.savefig('megaspike_recovery_scatter.svg', format='svg', bbox_inches='tight')

plt.show()
