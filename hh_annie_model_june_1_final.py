from brian2 import *
from brian2tools import *
from scipy.signal import find_peaks, lfilter

#%% Brian2 settings
defaultclock.dt = 1*ms
rng = np.random.default_rng(1)
dt_ms = float(defaultclock.dt / ms)

#%% Helpers
def poisson_times(rate, window_ms, pulse_dur_ms):
    """Generate Poisson-distributed event times within a window."""
    if rate <= 0:
        return np.array([])
    isi = rng.exponential(1000.0 / rate, size=int(3 * rate * window_ms / 1000))
    times = np.cumsum(isi)
    times = times[times < window_ms - pulse_dur_ms]  # leave room for last pulse
    return times

#%% Morphology
cyl_len = 200
morpho = Soma(diameter = 3*um)
morpho.trunk = Cylinder(length=70*um, diameter=.25*um, n=70, type='trunk')
# Axon split at 70 um from branch point: .distal continues the main axon, .collateral is the new branch
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

Rm = 10*kohm*cm**2 #specific, radial resistance
Ra = 25*ohm*cm #specific, longitudinal resistance
Cm0 = 10*uF/cm**2

# gLk0 = 1/Rm #.1msiemens/cm**2
gLk0 = 1.4*msiemens/cm**2 #Rm = .714*kohm*cm**2
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

#%% Build neuron
neuron = SpatialNeuron(morphology=morpho, model=eqs, method="heun",
                       Cm=Cm0, Ri=Ra)

branch_siz_start = 0*um
branch_siz_end = 20*um
branch_siz_size = abs(branch_siz_start-branch_siz_end)
branch_siz_center = branch_siz_start + branch_siz_size/2
branch_siz_center_idx = axon_start_idx + branch_siz_center/um 

dend_siz_start = 85*um
dend_siz_end   = 175*um
dend_siz_size = abs(dend_siz_start-dend_siz_end)
dend_siz_center = dend_siz_start + dend_siz_size/2
dend_siz_center_idx = dendrite_start_idx + dend_siz_center/um

axon_siz_start = 90*um
axon_siz_end   = 130*um
axon_proximal_len = 70*um
axon_siz_size = abs(axon_siz_start-axon_siz_end)
axon_siz_center = axon_siz_start + axon_siz_size/2
# SIZ falls in axon.distal (which starts at index 141, 70 um from branch point)
axon_siz_center_idx = axon_distal_start_idx + (axon_siz_center - axon_proximal_len)/um


dend_input_start = dend_siz_center - 2.5*um
dend_input_end   = dend_siz_center + 2.5*um

axon_input_start = axon_siz_center - 2.5*um
axon_input_end   = axon_siz_center + 2.5*um

#%% Conductance distribution
neuron.gLk = gLk0
neuron.gNa = .05*gNa0
neuron.gK = .05*gK0
neuron.gCa = gCa0
neuron.gSK = 0*gSK0
neuron.gCAN = 0

neuron.trunk.axon.gNa = 0.04*gNa0
neuron.trunk.axon.distal.gNa = 0.04*gNa0
neuron.trunk.axon.collateral.gNa = 0.00*gNa0
neuron.trunk.dendrite.gNa = 0.04*gNa0

neuron.trunk.axon.gK = 0.04*gK0
neuron.trunk.axon.distal.gK = 0.04*gK0
neuron.trunk.axon.collateral.gK = 0.00*gK0
neuron.trunk.dendrite.gK = 0.04*gK0

neuron.trunk.axon.distal[0*um:axon_proximal_len].gSK = 1*gSK0
neuron.trunk.dendrite[0*um:dend_siz_start].gSK = 1*gSK0

neuron.trunk.axon.distal[axon_siz_start - axon_proximal_len:axon_siz_end - axon_proximal_len].gNa = 1*gNa0
neuron.trunk.axon.distal[axon_siz_start - axon_proximal_len:axon_siz_end - axon_proximal_len].gK = 1*gK0
neuron.trunk.axon.distal[axon_siz_start - axon_proximal_len:axon_siz_end - axon_proximal_len].gSK = 0*gSK0
neuron.trunk.dendrite[dend_siz_start:dend_siz_end].gNa = .5*gNa0
neuron.trunk.dendrite[dend_siz_start:dend_siz_end].gK = .5*gK0
neuron.trunk.dendrite[dend_siz_start:dend_siz_end].gSK = 0*gSK0

neuron.trunk.axon[branch_siz_start:branch_siz_end].gNa = 1.5*gNa0
neuron.trunk.axon[branch_siz_start:branch_siz_end].gK = 1*gK0
neuron.trunk.axon[branch_siz_start:branch_siz_end].gSK = 1*gSK0
neuron.trunk.axon[branch_siz_start:branch_siz_end].gCAN = 1.5*gCAN0

#%% Intial conditions and monitors
neuron.v = -42*mV
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

M = StateMonitor(neuron, ['v', 'cai', 'ISK', 'ICAN'], record=[0, int(branch_siz_center_idx), int(axon_siz_center_idx), int(dend_siz_center_idx)])
M2 = StateMonitor(neuron, ['v'], record=True)

#%% Stimulus parameters
stim_rate_dend = .5  # Hz
stim_rate_axon = .5 # Hz
stim_pulse_dur = 50   # ms per pulse
stim_amp = 5         # pA
stim_window = 1400000   # ms total window for stim
axon_on = 1
dend_on = 1

# %% Build stimuli
dend_times = poisson_times(stim_rate_dend, stim_window, stim_pulse_dur) if dend_on else np.array([])
axon_times = poisson_times(stim_rate_axon, stim_window, stim_pulse_dur) if axon_on else np.array([])

pre_steps  = int(500 / dt_ms)
stim_steps = int(stim_window / dt_ms)
post_steps = int(500 / dt_ms)
total_steps = pre_steps + stim_steps + post_steps

dend_arr = np.zeros(total_steps)
axon_arr = np.zeros(total_steps)

for t_ms in dend_times:
    i0 = pre_steps + int(t_ms / dt_ms)
    i1 = pre_steps + int((t_ms + stim_pulse_dur) / dt_ms)
    dend_arr[i0:i1] = stim_amp

for t_ms in axon_times:
    i0 = pre_steps + int(t_ms / dt_ms)
    i1 = pre_steps + int((t_ms + stim_pulse_dur) / dt_ms)
    axon_arr[i0:i1] = stim_amp

dend_stim = TimedArray(dend_arr * pA, dt=defaultclock.dt)
axon_stim = TimedArray(axon_arr * pA, dt=defaultclock.dt)

#%% Run
seed(1)
run((500 + stim_window + 500) * ms, report='text')

#%% Detect spikes
t_plot = M.t / ms
v_soma = M.v[0] / mV 
peaks, props = find_peaks(v_soma, height=-35, distance=100, prominence=5)
peak_times = t_plot[peaks]
peak_amps = props['peak_heights']

#%% Plot peaks
plt.figure()
plot(t_plot, v_soma)
plot(peak_times, peak_amps, 'rx', markersize=10)
ylim(-65, 0)
ylabel('V_{soma} (mV)')
xlabel('time (ms)')

# for col, label in [(0, 'soma'), (1, 'branch'), (2, 'axon'), (3, 'dend')]:
#     plt.figure()
#     plot(t_plot, M.v.T[:, col] / mV)
#     ylim(-65, 0)
#     ylabel(f'V_{{{label}}} (mV)')
#     xlabel('time (ms)')

#%% Plot soma trace
t_s = t_plot / 1000  
fig_trace, ax_trace = plt.subplots(figsize=(14, 2.5))
ax_trace.plot(t_s, v_soma, color='black', linewidth=0.5)
ax_trace.axis('off')
x0 = t_s.min() + 1
y0 = v_soma.min() + 1
ax_trace.plot([x0, x0 + 1], [y0, y0], 'k-', linewidth=1.5, clip_on=False)
ax_trace.plot([x0, x0], [y0, y0 + 10], 'k-', linewidth=1.5, clip_on=False)
ax_trace.text(x0 + 0.5, y0 - 0.5, '1 s', ha='center', va='top', fontsize=9)
ax_trace.text(x0 - 0.3, y0 + 5, '10 mV', ha='right', va='center', fontsize=9, rotation=90)
fig_trace.tight_layout()
fig_trace.savefig('soma_trace.svg', format='svg', bbox_inches='tight')

#%% Plot spike amplitudes
fig, ax = plt.subplots(figsize=(2.5, 5))
jitter = rng.uniform(-0.15, 0.15, size=len(peak_amps))
for x_jit, y in zip(jitter, peak_amps + 42):
    ax.plot(x_jit, y, 'o', mfc='none', mec='steelblue', mew=1.2, ms=8, alpha=0.85)
ax.set_ylabel('Spike amp (mV)')
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(0, 40)
ax.set_xticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
fig.savefig('spike_amps.svg', format='svg', bbox_inches='tight')

#%% Plot Calcium
fig_cai, ax_cai = plt.subplots(figsize=(14, 2.5))
ax_isk = ax_cai.twinx()
t_s = t_plot / 1000
ax_cai.plot(t_s, M.cai[1] / mmolar * 1000, color='tomato', linewidth=0.8, label='[Ca²⁺]')
ax_isk.plot(t_s, M.ISK[1] / (amp/meter**2) * 100, color='steelblue', linewidth=0.8, alpha=0.8, label='ISK')
ax_isk.plot(t_s, M.ICAN[1] / (amp/meter**2) * 100, color='darkorchid', linewidth=0.8, alpha=0.8, label='ICAN')
ax_isk.legend(loc='upper right', fontsize=8)
ax_cai.set_xlabel('time (s)')
ax_cai.set_ylabel('[Ca²⁺] (µM)', color='tomato')
ax_isk.set_ylabel('current (µA/cm²)', color='gray')
ax_cai.tick_params(axis='y', labelcolor='tomato')
ax_cai.set_title('Branch SIZ — [Ca²⁺], ISK, and ICAN')
fig_cai.tight_layout()

#%% Megaspike probability vs axon-dendrite input timing
# For each axon input: find the closest dendritic input (signed offset dt = t_dend - t_axon),
# then classify the spike that axon input triggered as megaspike vs minispike.
MEGASPIKE_THRESH = 25.0      # mV spike amplitude dividing mega- from mini-spikes
V_REST_OFFSET    = 42.0      # amplitude = peak voltage + 42 mV (rest = -42 mV); change to 0 for raw peak_amps
ATTRIB_WINDOW    = 150.0     # ms after axon onset in which a spike is attributed to that input

# Convert stimulus event times to simulation time (stim arrays begin after the pre-stim pad)
stim_offset_ms   = pre_steps * dt_ms          # = 500 ms
axon_event_times = axon_times + stim_offset_ms
dend_event_times = dend_times + stim_offset_ms

peak_amp_rel = peak_amps + V_REST_OFFSET      # amplitude of every detected soma peak

offsets  = []   # signed dt to closest dendritic input, per axon input
amps     = []   # amplitude of the spike triggered by that axon input (nan if none)
category = []   # 1 = megaspike, 0 = minispike, nan = no spike triggered

for t_ax in axon_event_times:
    if len(dend_event_times) > 0:
        j  = np.argmin(np.abs(dend_event_times - t_ax))
        dt = np.abs(dend_event_times[j] - t_ax)   # unsigned offset to closest dendritic input
    else:
        dt = np.nan
    in_win = (peak_times >= t_ax) & (peak_times <= t_ax + ATTRIB_WINDOW)
    offsets.append(dt)
    if np.any(in_win):
        amp = peak_amp_rel[in_win].max()
        amps.append(amp)
        category.append(1.0 if amp > MEGASPIKE_THRESH else 0.0)
    else:
        amps.append(np.nan)
        category.append(np.nan)

offsets  = np.array(offsets)
amps     = np.array(amps)
category = np.array(category)

n_mega = int(np.nansum(category == 1))
n_mini = int(np.nansum(category == 0))
n_none = int(np.sum(np.isnan(category)))
print(f'Axon inputs: {len(offsets)} | megaspikes: {n_mega} | minispikes: {n_mini} | no spike: {n_none}')

#%% Scatter: timing offset vs triggered spike amplitude
fig_sc, ax_sc = plt.subplots(figsize=(6, 4))
mega = category == 1
mini = category == 0
ax_sc.scatter(offsets[mini], amps[mini], facecolors='none', edgecolors='steelblue',
              label='minispike', alpha=0.8)
ax_sc.scatter(offsets[mega], amps[mega], facecolors='none', edgecolors='tomato',
              label='megaspike', alpha=0.8)
ax_sc.axhline(MEGASPIKE_THRESH, color='gray', ls='--', lw=1)
ax_sc.set_xlabel('|axon − dendrite| offset |Δt| (ms)')
ax_sc.set_ylabel('spike amplitude (mV)')
ax_sc.legend(frameon=False, fontsize=8)
fig_sc.tight_layout()

#%% Binned probability of megaspike vs timing offset
triggered = ~np.isnan(category)
ofs = offsets[triggered]
cat = category[triggered]

bin_edges = np.arange(0, 201, 25)
bin_cent  = 0.5 * (bin_edges[:-1] + bin_edges[1:])
idx       = np.digitize(ofs, bin_edges) - 1
b_mega = np.zeros(len(bin_cent))
b_mini = np.zeros(len(bin_cent))
for b in range(len(bin_cent)):
    sel = idx == b
    b_mega[b] = np.sum(cat[sel] == 1)
    b_mini[b] = np.sum(cat[sel] == 0)
b_tot = b_mega + b_mini
prob  = np.where(b_tot > 0, b_mega / b_tot, np.nan)

fig_pr, ax_pr = plt.subplots(figsize=(7, 4))
ax_cnt = ax_pr.twinx()
ax_cnt.bar(bin_cent, b_tot, width=20, color='lightgray', alpha=0.6, zorder=0,
           label='# triggered spikes')
ax_pr.plot(bin_cent, prob, '-o', color='tomato', zorder=2, label='P(megaspike)')
ax_pr.set_xlabel('|axon − dendrite| offset |Δt| (ms)')
ax_pr.set_ylabel('P(megaspike | spike)', color='tomato')
ax_pr.set_ylim(-0.02, 1.02)
ax_pr.tick_params(axis='y', labelcolor='tomato')
ax_cnt.set_ylabel('# triggered spikes', color='gray')
ax_cnt.tick_params(axis='y', labelcolor='gray')
ax_pr.set_zorder(ax_cnt.get_zorder() + 1)
ax_pr.patch.set_visible(False)
fig_pr.tight_layout()
fig_pr.savefig('megaspike_prob_vs_timing.svg', format='svg', bbox_inches='tight')

#%% Inspect outlier megaspikes (megaspike despite large |Δt| to nearest dend input)
from matplotlib.patches import Patch

OUTLIER_OFFSET = 500.0    # ms; megaspikes whose nearest dend input is beyond this are "isolated"
ZOOM           = 1500.0   # ms shown on each side of the triggering axon input

outlier_idx = np.where((category == 1) & (offsets > OUTLIER_OFFSET))[0]
print(f'\n{len(outlier_idx)} isolated megaspikes (|Δt| > {OUTLIER_OFFSET:.0f} ms):')
for k in outlier_idx:
    print(f'  axon input @ {axon_event_times[k]/1000:.2f} s | nearest dend |Δt| = '
          f'{offsets[k]:.0f} ms | amp = {amps[k]:.1f} mV')

# Mark all outlier axon inputs on the full soma trace
fig_ov, ax_ov = plt.subplots(figsize=(14, 2.5))
ax_ov.plot(t_plot / 1000, v_soma, color='black', lw=0.4)
for k in outlier_idx:
    ax_ov.axvline(axon_event_times[k] / 1000, color='red', lw=0.8, alpha=0.7)
ax_ov.set_xlabel('time (s)')
ax_ov.set_ylabel('V_soma (mV)')
ax_ov.set_title(f'soma trace — red = isolated megaspike axon inputs (n={len(outlier_idx)})')
fig_ov.tight_layout()

# Zoomed traces around each outlier
n_show = min(len(outlier_idx), 6)
if n_show > 0:
    fig_z, axz = plt.subplots(n_show, 1, figsize=(10, 2.2 * n_show), squeeze=False)
    for ax, k in zip(axz[:, 0], outlier_idx[:n_show]):
        t0 = axon_event_times[k]
        win = (t_plot >= t0 - ZOOM) & (t_plot <= t0 + ZOOM)
        ax.plot(t_plot[win], v_soma[win], color='black', lw=0.6)
        for ta in axon_event_times:
            if t0 - ZOOM <= ta <= t0 + ZOOM:
                ax.axvspan(ta, ta + stim_pulse_dur, color='steelblue', alpha=0.25)
        for td in dend_event_times:
            if t0 - ZOOM <= td <= t0 + ZOOM:
                ax.axvspan(td, td + stim_pulse_dur, color='seagreen', alpha=0.25)
        ax.axvline(t0, color='red', lw=1.0)
        pm = (peak_times >= t0 - ZOOM) & (peak_times <= t0 + ZOOM)
        ax.plot(peak_times[pm], peak_amps[pm], 'rx', ms=8)
        ax.set_title(f'axon @ {t0/1000:.2f} s | nearest dend |Δt| = {offsets[k]:.0f} ms | '
                     f'amp = {amps[k]:.1f} mV', fontsize=8)
        ax.set_ylabel('V_soma (mV)')
    axz[-1, 0].set_xlabel('time (ms)')
    axz[0, 0].legend(handles=[Patch(color='steelblue', alpha=0.25, label='axon pulse'),
                              Patch(color='seagreen', alpha=0.25, label='dend pulse'),
                              Patch(color='red', label='triggering axon input')],
                     fontsize=7, loc='upper right')
    fig_z.tight_layout()

plt.show()