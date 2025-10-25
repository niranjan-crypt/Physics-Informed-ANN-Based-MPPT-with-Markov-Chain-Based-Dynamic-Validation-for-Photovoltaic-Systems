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
1. **Directly maps environmental conditions $(G, T)$ to optimal operating points $(V_{mpp}, I_{mpp})$** using an Artificial Neural Network, enabling single-step, zero-oscillation tracking.
2. **Embeds semiconductor physics** (logarithmic open-circuit voltage dependence, thermal voltage derating, and $G\text{--}T$ coupling) into the network feature space.
3. **Prevents data leakage and overfitting** through group-based cross-validation splits, Bayesian Regularization (`trainbr`), and statistical hypothesis testing (two-sample $t$-test and Cohen's $d$).
4. **Stress-tests MPPT controllers with realistic stochastic weather** generated via a discrete 2D Markov Chain simulator (capturing diurnal cycles and sudden cloud transients).
5. **Benchmarks against classical P&O** using a 5-parameter single-diode PV model solved with Newton-Raphson iterations.
6. **Integrates with Simulink** for DC-DC converter control, Hybrid Energy Storage Systems (HESS), and multi-algorithm comparisons (P&O, InC, ANN, FLC, PSO).

---

## 🔬 Core Innovations

### 1. Physics-Informed Feature Engineering
Rather than treating the solar cell as an arbitrary black-box regression, 6 physically motivated features are constructed:
$$\mathbf{X} = \left[ G, \, T, \, \ln(G), \, \sqrt{G}, \, (T - T_{\text{ref}})^2, \, \frac{G}{G_{\text{ref}}}(T - T_{\text{ref}}) \right]$$
* **$\ln(G)$**: Represents the fundamental diode equation where open-circuit voltage scales logarithmically with photogenerated current ($V_{oc} \propto \frac{nkT}{q} \ln\frac{I_{ph}}{I_0}$).
* **$\sqrt{G}$**: Captures nonlinear boundary and recombination dynamics.
* **$(T - T_{\text{ref}})^2$**: Captures non-linear high-temperature efficiency penalties.
* **$\frac{G}{G_{\text{ref}}}(T - T_{\text{ref}})$**: Models physical coupling between solar irradiation and cell temperature rise.

### 2. Dual Output & Physical Data Augmentation
* **Simultaneous $(V_{mpp}, I_{mpp})$ Prediction**: Eliminates secondary lookups and computes maximum power directly: $P_{mpp} = V_{mpp} \times I_{mpp}$.
* **Physics-Constrained Augmentation**: Synthesizes realistic variants using physical temperature coefficients ($\frac{dV}{dT} \approx -0.3\%/\text{°C}$) and current-irradiance proportionality ($I \propto G$).
* **Group-Based Partitioning**: All augmented variations of an original $(G, T)$ cluster stay within the same partition, strictly preventing data leakage across train, validation, and test sets.

### 3. Statistical Overfitting Verification
The training pipeline (`OVERFIT_RED.m`) automatically evaluates:
* **Two-Sample $t$-test (`ttest2`)**: Verifies the null hypothesis that train and test error distributions are indistinguishable ($p \ge 0.05$).
* **Cohen's $d$ Effect Size**: Measures error divergence between partitions (target $d < 0.2$ for trivial difference).

### 4. Markov Chain Dynamic PV Weather Simulator
* Discretizes the continuous $(T, G)$ operating space into 300 discrete states ($15 \text{ Temperature bins} \times 20 \text{ Irradiance bins}$).
* Computes an empirical row-stochastic transition probability matrix $P_{ij} = P(S_{t+1}=j \mid S_t=i)$ from 45,000+ field measurements.
* Simulates multi-day synthetic time series that exhibit realistic temporal autocorrelation and cloud-cover volatility for stress-testing controllers.

---

## 📂 Project Structure

| File / Directory | Description |
| :--- | :--- |
| **`OVERFIT_RED.m`** | **Flagship training script**: Physics-informed feature engineering, group-split, Bayesian regularization (`trainbr`), statistical overfitting tests ($t$-test, Cohen's $d$). |
| **`MARKOV01.m`** | **Markov Chain Weather Simulator**: State discretization, transition probability computation, and synthetic weather generation. |
| **`YETTTT_FIN.m`** | **Closed-loop dynamic pipeline**: Feeds Markov weather sequences into trained Physics-ANN, predicts dynamic $(V_{mpp}, I_{mpp}, P_{mpp})$, and renders 9-panel diagnostics. |
| **`PandO.m`** | **P&O Benchmark**: Simulates classical Perturb & Observe MPPT against a 5-parameter Newton-Raphson PV cell model under identical Markov sequences. |
| **`ASSIGNMENT.m`** | **Interactive CLI verification**: Compares ANN-predicted $V_{mpp}$ against real measured $I\text{--}V$ and $P\text{--}V$ curves with error metrics. |
| **`CLEAN_DATASETnyt.m`** | Preprocessing script filtering daytime operating conditions ($G > 10\text{ W/m}^2$, $V > 1\text{ V}$). |
| **`clean_dataset.csv`** | Cleaned field measurement dataset (45,000+ samples: Temperature, Irradiance, Voltage, Current, Power). |
| **`daytime_dataset.csv`** | Filtered daytime operational data. |
| **`ANN_MPPT_PhysicsInformed.mat`**| Pre-trained Physics-Informed Neural Network model weights and normalization scalers. |
| **`Markov_PV_Simulator_Output.mat`**| Generated synthetic weather sequences used for reproducible benchmarking. |
| **`PV_SYSTEM_WITH_VARIOUS_MPPT (PO-INC-ANN-FLC-PSO)/`** | Comprehensive Simulink benchmark comparing P&O, Incremental Conductance, ANN, Fuzzy Logic, and PSO. |
| **`LOADED_ANN1.slx`** | Simulink model integrating the trained neural network block into a DC-DC converter circuit. |
| **`StaticVSG_HESS.slx`** | Simulink model for Virtual Synchronous Generator (VSG) and Hybrid Energy Storage System (HESS) grid interface. |

---

## 🚀 Quick Start Guide

### Prerequisites
* **MATLAB R2020b or later**
* **Deep Learning Toolbox**
* **Statistics and Machine Learning Toolbox**
* **Simulink & Simscape Electrical** (for `.slx` models)

### Step 1: Train the Physics-Informed ANN
Run [`OVERFIT_RED.m`](OVERFIT_RED.m) in MATLAB:
```matlab
run('OVERFIT_RED.m')
```
* Loads `clean_dataset.csv`, groups $(G, T)$ operating points, constructs physical features, applies Bayesian Regularization, and verifies statistical generalization.
* Saves the model to `ANN_MPPT_PhysicsInformed.mat`.

### Step 2: Generate Markov Dynamic Weather Scenarios
Run [`MARKOV01.m`](MARKOV01.m) in MATLAB:
```matlab
run('MARKOV01.m')
```
* Computes the $300 \times 300$ transition probability matrix and simulates a 7-day realistic weather sequence.
* Saves sequence to `Markov_PV_Simulator_Output.mat`.

### Step 3: Run Dynamic Testing & Diagnostics
Run [`YETTTT_FIN.m`](YETTTT_FIN.m) in MATLAB:
```matlab
run('YETTTT_FIN.m')
```
* Propagates the Markov sequence through the Physics-Informed ANN.
* Produces a 9-panel diagnostic dashboard depicting $T\text{--}G$ phase space, predicted $(V_{mpp}, I_{mpp}, P_{mpp})$, and environmental correlation curves.

### Step 4: Run Classical P&O Benchmark Comparison
Run [`PandO.m`](PandO.m) in MATLAB:
```matlab
run('PandO.m')
```
* Executes the Perturb & Observe algorithm over the same weather sequence using a Newton-Raphson 5-parameter PV model.
* Reports MAE, RMSE, energy yield ($\text{kWh}$), and tracking efficiency percentage.

### Step 5: Interactive Curve Validation
Run [`ASSIGNMENT.m`](ASSIGNMENT.m) in MATLAB:
```matlab
run('ASSIGNMENT.m')
```
* Allows you to enter any custom Irradiance and Temperature values in the console to inspect the ANN prediction superimposed on measured $I\text{--}V$ and $P\text{--}V$ curves.

---

## 📊 Benchmark & Comparative Performance

| Metric | Classical P&O Algorithm | Physics-Informed ANN (This Work) |
| :--- | :---: | :---: |
| **Tracking Method** | Iterative perturbation ($\Delta V = 0.5\text{ V}$) | Direct algebraic prediction $\mathbf{x} \to (V_{mpp}, I_{mpp})$ |
| **Convergence Speed** | Several iterations ($20\text{--}50\text{ steps}$) | Instantaneous ($1\text{ step}$) |
| **Steady-State Oscillation** | Present ($\pm \Delta V$ hunting around peak) | **Zero oscillation** |
| **Dynamic Cloud Transient Loss**| Prone to tracking in opposite direction | Adapts instantaneously to new $(G, T)$ |
| **Overall Tracking Efficiency** | $\approx 96.5\%\text{--}98.2\%$ | **$> 99.1\%$** |
| **Generalization (Cohen's $d$)**| N/A | **$< 0.2$ (Trivial error difference)** |

---

## 🛠️ Simulink Integration

To deploy the trained ANN inside Simulink:
1. Load the trained network:
   ```matlab
   load('ANN_MPPT_PhysicsInformed.mat', 'net');
   gensim(net); % Generates a native Simulink neural network block
   ```
2. Open [`LOADED_ANN1.slx`](LOADED_ANN1.slx) or [`PV_SYSTEM_WITH_VARIOUS_MPPT (PO-INC-ANN-FLC-PSO)/PV_SYSTEM_WITH_VARIOUS_MPPT.slx`](PV_SYSTEM_WITH_VARIOUS_MPPT%20(PO-INC-ANN-FLC-PSO)/PV_SYSTEM_WITH_VARIOUS_MPPT.slx).
3. Connect sensor inputs $(G, T)$ to the neural network block and route the predicted $V_{mpp}$ reference to the DC-DC boost converter PWM controller.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
