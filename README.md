# HH Multi-Compartment Spike Model

A biophysical, multi-compartment Hodgkin–Huxley neuron model built with Brian2 (https://brian2.readthedocs.io). The model simulates a soma with a trunk that branches into an axon (with a distal segment and a collateral) and a dendrite, each with spatially distributed ion channels. It reproduces somatic spiking driven by Poisson stimulation of the dendrite and axon, and tracks calcium dynamics together with SK and CAN currents at the branch spike-initiation
zone (SIZ).

## Requirements

See environment.yml 

## Setup

Using conda:

```bash
conda env create -f environment.yml
conda activate hh-annie-model
```

## Running

```bash
python hh_annie_model_june_1_final.py
```

The full simulation covers ~141 s of model time (500 ms pre + 140 s stimulus +
500 ms post at a .1 ms timestep) and should take < 5 minutes. 
It opens the figures interactively and writes `soma_trace.svg`
and `spike_amps.svg` to the working directory.

## Files

- `hh_annie_model_june_1_final.py` — the full model and analysis/plotting script
- `environment.yml` — conda environment specification
