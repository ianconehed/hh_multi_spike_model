"""
Characterize the post-megaspike net current Im(t).

Run a single conditioning megaspike (two coincident axon+dendrite inputs) and
record the total membrane current Im(t) at the branch SIZ, where both gCAN and
gSK are expressed. The point of interest is the slow, post-spike component: SK
(hyperpolarizing) decays faster than CAN (depolarizing, tau_r = 2 s), so the net
depolarizing drive (~ -ICAN - ISK) can peak seconds AFTER the spike itself.

We find the time of that post-spike Im peak, which is the delay at which a
follow-up pairing should be maximally facilitated.

Reuses only the model from hh_annie_model_june_1_final.py (copied here).
"""

from brian2 import *
from scipy.signal import find_peaks

#%% Brian2 settings
defaultclock.dt = 1*ms
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
branch_siz_center = branch_siz_start + abs(branch_siz_start-branch_siz_end)/2
branch_siz_center_idx = int(axon_start_idx + branch_siz_center/um)

dend_siz_start = 85*um
dend_siz_end   = 175*um
dend_siz_center = dend_siz_start + abs(dend_siz_start-dend_siz_end)/2
dend_siz_center_idx = int(dendrite_start_idx + dend_siz_center/um)

axon_siz_start = 90*um
axon_siz_end   = 130*um
axon_siz_center = axon_siz_start + abs(axon_siz_start-axon_siz_end)/2
axon_siz_center_idx = int(axon_distal_start_idx + (axon_siz_center - axon_proximal_len)/um)

dend_input_start = dend_siz_center - 2.5*um
dend_input_end   = dend_siz_center + 2.5*um
axon_input_start = axon_siz_center - 2.5*um
axon_input_end   = axon_siz_center + 2.5*um

#%% Build neuron
neuron = SpatialNeuron(morphology=morpho, model=eqs, method="heun", Cm=Cm0, Ri=Ra)

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
neuron.trunk.axon[branch_siz_start:branch_siz_end].gCAN = 1.5*gCAN0

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
# Noise off for a clean, deterministic characterization.
neuron.noise_mask = 0

#%% Single conditioning megaspike: coincident axon + dendrite pulse
warmup_ms  = 500     # settle
t_spike    = warmup_ms
pulse_dur  = 50      # ms
stim_amp   = 5       # pA per input
record_ms  = 7000    # observe well past the expected net-current peak
total_ms   = warmup_ms + record_ms
total_steps = int(total_ms / dt_ms)

axon_arr = np.zeros(total_steps)
dend_arr = np.zeros(total_steps)
i0 = int(t_spike / dt_ms)
i1 = int((t_spike + pulse_dur) / dt_ms)
axon_arr[i0:i1] = stim_amp
dend_arr[i0:i1] = stim_amp
axon_stim = TimedArray(axon_arr * pA, dt=defaultclock.dt)
dend_stim = TimedArray(dend_arr * pA, dt=defaultclock.dt)

#%% Record Im and its slow components at the branch SIZ
M = StateMonitor(neuron, ['v', 'Im', 'ICAN', 'ISK', 'cai'],
                 record=[branch_siz_center_idx])

run(total_ms * ms, report='text')

#%% Locate the post-spike Im peak (skip the spike transient itself)
t_ms = M.t / ms
amp_unit = amp / meter**2
Im   = M.Im[0]   / amp_unit
ICAN = M.ICAN[0] / amp_unit
ISK  = M.ISK[0]  / amp_unit

# The CAN - SK net depolarizing drive (both as inward-positive contributions to Im).
can_minus_sk = -ICAN - ISK

skip_ms = 200  # ms after the pulse onset, to clear the fast spike currents
post = t_ms > (t_spike + skip_ms)
Im_post = Im[post]
t_post  = t_ms[post]

i_peak = int(np.argmax(Im_post))
t_peak_abs = t_post[i_peak]
t_peak_rel = t_peak_abs - t_spike
print(f"Branch SIZ Im peaks at t = {t_peak_abs:.0f} ms "
      f"({t_peak_rel:.0f} ms after the megaspike); Im = {Im_post[i_peak]:.3f} uA/cm^2")

# Same for the CAN-SK net, for cross-check.
cms_post = can_minus_sk[post]
j_peak = int(np.argmax(cms_post))
print(f"Branch SIZ (-ICAN - ISK) peaks at t = {t_post[j_peak]:.0f} ms "
      f"({t_post[j_peak] - t_spike:.0f} ms after the megaspike)")

#%% Plot
t_s = t_ms / 1000

fig, (axv, axi) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

axv.plot(t_s, M.v[0] / mV, color='black', lw=0.8)
axv.set_ylabel('V_branch (mV)')
axv.set_title('Single conditioning megaspike at branch SIZ')
axv.spines['top'].set_visible(False)
axv.spines['right'].set_visible(False)

axi.axhline(0, color='gray', lw=0.6)
axi.plot(t_s, Im,   color='black',      lw=1.2, label='Im (net)')
axi.plot(t_s, -ICAN, color='darkorchid', lw=1.0, alpha=0.8, label='-ICAN (depol.)')
axi.plot(t_s, -ISK,  color='steelblue',  lw=1.0, alpha=0.8, label='-ISK (hyperpol.)')
axi.plot(t_s, can_minus_sk, color='tomato', lw=1.2, ls='--', label='-ICAN - ISK')
axi.axvline(t_peak_abs / 1000, color='red', ls=':', lw=1.0,
            label=f'Im peak  (+{t_peak_rel:.0f} ms)')
# Zoom the y-axis to the slow tail; the spike transient is far off-scale.
finite_tail = np.abs(Im[t_ms > t_spike + skip_ms])
axi.set_ylim(-np.max(finite_tail) * 1.5, np.max(finite_tail) * 1.5)
axi.set_xlabel('time (s)')
axi.set_ylabel('current (µA/cm²)')
axi.legend(fontsize=8, loc='upper right')
axi.spines['top'].set_visible(False)
axi.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig('conditioning_current.svg', format='svg', bbox_inches='tight')
plt.show()
