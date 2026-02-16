# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Superconducting circuit simulator using the **method of nodes**. Models transmon qubits and similar circuits: builds graph topology, constructs capacitance/inductance matrices, performs charge-basis quantization, Crank-Nicolson time evolution under SFQ pulse trains, and evaluates gate fidelity.

## Running

```bash
# Main simulation (transmon Rabi + fidelity) — takes ~1-2 min due to Crank-Nicolson loops
python non_linear.py

# Linear circuit demo
python linear.py

# Parameter sweep (detuning, amplitude, sigma)
python param_sweep.py

# Tests (manual assertions, not pytest)
python tests.py
```

Uses a `.venv` with Python 3.13. Dependencies: numpy, scipy, matplotlib.

## Architecture

**Simulation pipeline** (executed in `non_linear.py`):
1. Define circuit as a `graph_rep` dict (nodes, capacitors, inductors, josephson_junctions, external_flux)
2. `Circuit(graph_rep)` builds graph, partitions nodes, constructs C and L^-1 matrices
3. `_quantize(n_cut)` builds charge-basis Hamiltonian (single-node only)
4. `_diagonalize()` gets energy eigenvalues/eigenstates
5. `_change_basis()` transforms charge operator to energy eigenbasis
6. `crank_nicolson(...)` time-evolves under SFQ Gaussian pulse train
7. `_build_unitary(...)` constructs gate unitary from basis state evolution
8. `_calculate_fidelity(d)` computes average gate fidelity vs target (default: Pauli-X)

**Key modules:**
- `Elements.py` — Graph primitives: `Node`, `Branch`, `Capacitor`, `Inductor`, `JosephsonJunction`, `Graph`
- `Circuit.py` — `Circuit` class with all physics (graph construction, matrices, quantization, time evolution, fidelity). Exports `PHI_0`.
- `plotting.py` — All visualization functions. `plot_all()` generates 2x3 dashboard. `charge_to_phase_basis()` does vectorized charge-to-phase Fourier transform.
- `config.py` — Optimal SFQ pulse parameters (detuning, amplitude_scale, sigma)
- `sim.py` — Standalone procedural script (original prototype, not using Circuit class)

**Design notes:**
- Circuit state is mutated in-place through the pipeline (`circuit.energies`, `circuit.states`, `circuit.U`, etc.)
- Quantization (`_quantize`) only supports single-node (N=1) circuits; raises `NotImplementedError` otherwise
- The `graph_rep` dict format is the user-facing API for defining circuits — see `non_linear.py` or `linear.py` for examples
- `_drive_params(detuning)` is the shared helper for computing f01, f_drive, T_drive from eigenvalues
