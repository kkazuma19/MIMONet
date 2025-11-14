# MIMONet
__MIMONet__ is a __multi-input, multi-output neural operator__ framework designed for __real-time virtual sensing and diagnostics__ in environments where direct instrumentation is infeasible, such as nuclear and thermal-fluid systems. The model learns mappings from heterogeneous boundary and sensor inputs to full-field physical quantities (e.g., pressure, temperature, velocity, turbulence kinetic energy) with spatial resolution and uncertainty quantification.  


## Key Features

- **Operator Learning for Field Reconstruction:** Learns nonlinear mappings $G: U \to S$ between input functions $U$ (e.g., boundary or forcing conditions) and spatially distributed output fields $S$ (pressure, velocity, temperature, etc.).
- **Heterogeneous Input Fusion:** Encodes both scalar and function-valued inputs into a shared latent space using multiplicative fusion for efficient multimodal representation.
- **Coupled Multiphysics Outputs:** Simultaneously decodes multiple physically correlated fields, preserving inter-field coherence across channels.
- **Uncertainty Quantification:** Integrates Monte Carlo dropout with conformal calibration to provide statistically reliable and interpretable confidence intervals.
- **Noise Robustness:** Demonstrates resilience to sensor noise, drift, and perturbations without retraining.
- **General Applicability:** Validated on problems, including:
  - Lid-Driven Cavity Flow (**LDC**)
  - Pressurized Water Reactor Subchannel (**Subchannel**)
  - Heat Exchanger Flow (**HeatExchanger**)


## Source Code (`src/`)

The `src/` directory contains the full implementation of MIMONet, including model architectures, training routines, and utility functions.  


Each component is modularized for reproducibility and clarity:
- **`mimonet.py` / `mimonet_drop.py`:** Define the neural operator architecture and its stochastic extension.  
- **`training.py`:** Entry point for model training and validation.  
- **`utils.py`:** Includes data handling, plotting, and analysis helpers.  
- **`fcn.py`:** Implements a standard fully connected network baseline for comparison.  


## Citation

If you use this work (code, datasets), please cite:

> **Kobayashi, K., Ahmed, F., & Alam, S. B.**  
> *Virtual sensing to enable real-time monitoring of inaccessible locations & unmeasurable parameters.*  
> arXiv preprint arXiv:2412.00107, 2024.  