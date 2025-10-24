# Physics-Informed ANN & Markov Chain MPPT for Solar Photovoltaic Systems

[![MATLAB](https://img.shields.io/badge/MATLAB-R2020b%2B-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![Simulink](https://img.shields.io/badge/Simulink-Enabled-orange.svg)](https://www.mathworks.com/products/simulink.html)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Toolbox-brightgreen.svg)](https://www.mathworks.com/products/deep-learning.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An advanced Machine Learning and Power Electronics project implementing **Physics-Informed Artificial Neural Networks (PINN)** and **Markov Chain Stochastic Weather Modeling** for Maximum Power Point Tracking (MPPT) in solar photovoltaic (PV) systems.

---

## 📌 Executive Summary
Traditional Maximum Power Point Tracking (MPPT) algorithms such as **Perturb & Observe (P&O)** and **Incremental Conductance (InC)** suffer from steady-state power oscillations around the MPP and tracking sluggishness under rapid solar irradiance fluctuations.

This project delivers a complete framework that:
1. Directly maps environmental conditions (G, T) to optimal operating points (Vmpp, Impp).
2. Embeds semiconductor physics into the network feature space.
3. Prevents data leakage and overfitting through group-based cross-validation splits and Bayesian Regularization.
4. Stress-tests MPPT controllers with realistic stochastic weather generated via a discrete 2D Markov Chain simulator.

---

## 🔬 Core Innovations

### 1. Physics-Informed Feature Engineering
Rather than treating the solar cell as an arbitrary black-box regression, 6 physically motivated features are constructed:
- ln(G): Represents logarithmic open-circuit voltage scaling.
- sqrt(G): Captures nonlinear boundary and recombination dynamics.
- (T - T_ref)²: Captures non-linear high-temperature efficiency penalties.
- G_norm * (T - T_ref): Models thermal coupling.
