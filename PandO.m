clc; clear; close all;
%% ========================================================================
%  P&O MPPT ALGORITHM BENCHMARK (.m file implementation)
%  PURPOSE: Test P&O on the same Markov-generated weather sequence
%           to get a fair benchmark against the ANN.
%  FIX:     Replaced simplified (broken) PV model with a robust
%           5-parameter model that uses a Newton-Raphson solver.
%% ========================================================================

fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║     P&O MPPT BENCHMARK SIMULATION (MATLAB SCRIPT)        ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

%% STEP 1: Load Markov-Generated Weather Sequence
fprintf('[STEP 1] Loading Markov-Generated Sequences...\n');
if ~isfile('Markov_PV_Simulator_Output.mat')
    error('Please run CODE 2 first to generate Markov_PV_Simulator_Output.mat');
end
load('Markov_PV_Simulator_Output.mat', 'markov_output');

G_sim = markov_output.simulated_Irradiance;
T_sim = markov_output.simulated_Temperature;
time_hours = markov_output.time_hours;
N = length(G_sim);

fprintf('  ✓ Loaded %d time steps (%.1f days)\n\n', N, max(time_hours)/24);

%% STEP 2: Define P&O Algorithm and PV Panel Parameters

% --- P&O Tunable Parameters ---
V_initial = 40;     % Initial guess for Vmpp (V)
V_step_size = 0.5;  % Perturbation step (V). This is critical!
max_steps = 50;     % Max "hunts" per weather condition to find MPP
V_last_mpp = V_initial; % Start with initial guess

% --- PV Panel Model Parameters (Standard ~300W Panel) ---
% These are needed to create the "virtual" PV panel
panel.G_ref = 1000;     % Reference Irradiance (W/m^2)
panel.T_ref = 25;       % Reference Temperature (C)
panel.Isc_ref = 8.7;    % Short-circuit current (A)
panel.Voc_ref = 45.0;   % Open-circuit voltage (V)
% Note: Vmpp/Impp are results of the model, not inputs
panel.Ki = 0.05;      % Temp. coeff. for Isc (%/C)
panel.Kv = -0.34;     % Temp. coeff. for Voc (%/C)
panel.N_cells = 72;     % Number of cells in series
panel.k = 1.3806e-23;   % Boltzmann's constant
panel.q = 1.6021e-19;   % Electron charge
% --- Model Physics Parameters (5-Parameter Model) ---
panel.A_ideality = 1.3; % Diode Ideality Factor
panel.Rs = 0.30;        % Series Resistance (Ohms)
panel.Rsh = 1000;       % Shunt Resistance (Ohms)

fprintf('[STEP 2] P&O Parameters Defined:\n');
fprintf('  Voltage Step (ΔV): %.2f V\n', V_step_size);
fprintf('  PV Model: ~300W panel (Isc=%.1fA, Voc=%.1fV)\n\n', ...
    panel.Isc_ref, panel.Voc_ref);

%% STEP 3: Run Benchmark Simulation
fprintf('[STEP 3] Running P&O Simulation for %d weather points...\n', N);

% Arrays to store the results
P_tracked_PO = zeros(N, 1);
P_true_MPP = zeros(N, 1);
V_tracked_PO = zeros(N, 1);

tic; % Start timer

% Outer loop: Iterate through each 15-minute weather step
for i = 1:N
    G_current = G_sim(i);
    T_current = T_sim(i);
    
    % --- P&O Algorithm (Inner Loop) ---
    % This "hunts" for the MPP for the *current* weather condition
    V_op = V_last_mpp; % Start from the last known MPP
    P_old = calculate_PV_power(panel, G_current, T_current, V_op);
    
    direction = +1; % Start by increasing voltage
    
    % Calculate current Voc to set a boundary
    Voc_adjusted = panel.Voc_ref * (1 + panel.Kv/100 * (T_current - panel.T_ref));
    
    for k = 1:max_steps
        V_new = V_op + direction * V_step_size;
        
        % Check for boundary conditions (don't go past 0V or Voc)
        if V_new < 0 || V_new > Voc_adjusted
            direction = -direction; % Turn around
            V_new = V_op + direction * V_step_size;
        end
        
        P_new = calculate_PV_power(panel, G_current, T_current, V_new);
        
        if P_new > P_old
            % Good move, keep going
            V_op = V_new;
            P_old = P_new;
        else
            % Bad move, turn around
            direction = -direction;
            
            % If we turned around, we are oscillating at the peak
            % Break after one turnaround
            if k > 1 
                break;
            end
        end
    end
    
    % --- Find the *Actual* True MPP (for comparison) ---
    % We do a brute-force sweep to find the "perfect" answer
    [P_true_mpp_val, ~] = find_true_mpp(panel, G_current, T_current, Voc_adjusted);
    
    % Store results for this time step
    P_tracked_PO(i) = P_old;        % Power the P&O algorithm found
    V_tracked_PO(i) = V_op;         % Voltage the P&O algorithm found
    P_true_MPP(i) = P_true_mpp_val; % The "perfect" power
    
    % Update the starting guess for the *next* weather condition
    V_last_mpp = V_op;
end

toc; % Stop timer
fprintf('  ✓ Simulation complete!\n\n');

%% STEP 4: Calculate P&O Performance Metrics
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║            P&O BENCHMARK PERFORMANCE METRICS             ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n');

% Filter out any low-power/invalid results
valid_idx = P_true_MPP > 10; % Only count when power is > 10W
errors_P = abs(P_true_MPP(valid_idx) - P_tracked_PO(valid_idx));

mae_P = mean(errors_P);
rmse_P = sqrt(mean(errors_P.^2));
max_P_error = max(errors_P);
mean_rel_P_error = mean(errors_P ./ P_true_MPP(valid_idx)) * 100;

fprintf('Pmpp Tracking Errors (P&O):\n');
fprintf('  MAE = %.2f W\n', mae_P);
fprintf('  RMSE = %.2f W\n', rmse_P);
fprintf('  Max Error = %.2f W\n', max_P_error);
fprintf('  Mean Relative Power Error = %.2f%%\n\n', mean_rel_P_error);

% Compare total energy generated (over 15-min intervals = 0.25 hours)
total_energy_true = sum(P_true_MPP) * (time_hours(2)-time_hours(1)); % Wh
total_energy_PO = sum(P_tracked_PO) * (time_hours(2)-time_hours(1)); % Wh
tracking_efficiency = (total_energy_PO / total_energy_true) * 100;

fprintf('Energy & Efficiency:\n');
fprintf('  Ideal Energy (True MPP): %.2f kWh\n', total_energy_true / 1000);
fprintf('  Tracked Energy (P&O):    %.2f kWh\n', total_energy_PO / 1000);
fprintf('  P&O Tracking Efficiency: %.2f%%\n\n', tracking_efficiency);

%% STEP 5: Visualize Results
figure('Name', 'P&O Benchmark vs. True MPP', 'Color', 'w');

% Plot 1: Power Tracking
subplot(2,1,1);
plot(time_hours, P_true_MPP, 'k-', 'LineWidth', 2.5);
hold on;
plot(time_hours, P_tracked_PO, 'r--', 'LineWidth', 1.5);
xlabel('Time (hours)');
ylabel('Power (W)');
title('P&O Power Tracking vs. True MPP');
legend('True MPP', 'P&O Tracked Power', 'Location', 'best');
grid on;
xlim([0 max(time_hours)]);

% Plot 2: Tracking Error
subplot(2,1,2);
plot(time_hours(valid_idx), errors_P, 'b-');
xlabel('Time (hours)');
ylabel('Power Error (W)');
title(sprintf('P&O Tracking Error (MAE = %.2f W)', mae_P));
grid on;
xlim([0 max(time_hours)]);


%% ========================================================================
%  HELPER FUNCTIONS (THE "VIRTUAL PV PANEL")
%  ========================================================================

function P = calculate_PV_power(p, G, T, V)
    % Calculates PV power (P) for given G, T, and V
    % Solves the 5-parameter single-diode model using Newton-Raphson
    
    % 1. Adjust parameters for current T and G
    T_K = T + 273.15;
    T_ref_K = p.T_ref + 273.15;
    
    % Photocurrent
    Iph = (p.Isc_ref + p.Ki/100 * (T - p.T_ref)) * (G / p.G_ref);
    
    % Diode Saturation Current
    Voc_T = p.Voc_ref * (1 + p.Kv/100 * (T - p.T_ref));
    Vt_T = p.N_cells * p.A_ideality * p.k * T_K / p.q; % Thermal voltage
    I0 = (Iph - (Voc_T / p.Rsh)) / (exp(Voc_T / Vt_T) - 1);
    
    % 2. Iteratively solve for I using Newton-Raphson
    % f(I) = Iph - I - I0*(exp((V+I*Rs)/Vt_T) - 1) - (V+I*Rs)/Rsh = 0
    
    I_guess = Iph; % Start with a good guess (photocurrent)
    
    for iter = 1:10 % 10 iterations is almost always enough
        if I_guess < 0, I_guess = 0; end % Current can't be negative
        
        V_d = V + I_guess * p.Rs; % Voltage across diode + shunt
        
        % f(I)
        f = Iph - I_guess - I0 * (exp(V_d / Vt_T) - 1) - (V_d / p.Rsh);
        
        % f'(I) (derivative w.r.t I)
        df_dI = -1 - (I0 * p.Rs / Vt_T) * exp(V_d / Vt_T) - (p.Rs / p.Rsh);
        
        % Newton-Raphson step
        I_new = I_guess - f / df_dI;
        
        % Check for convergence
        if abs(I_new - I_guess) < 1e-4
            I_guess = I_new;
            break;
        end
        I_guess = I_new;
    end
    
    I = max(0, I_guess); % Final current
    
    if isnan(I) || isinf(I)
        I = 0;
    end
    
    P = V * I;
end


function [P_mpp, V_mpp] = find_true_mpp(panel, G, T, Voc_adjusted)
    % Finds the "perfect" MPP by sweeping the P-V curve
    
    % Create a voltage vector to sweep from 0 to adjusted Voc
    V_sweep = linspace(0, Voc_adjusted, 200);
    P_sweep = zeros(size(V_sweep));
    
    for i = 1:length(V_sweep)
        P_sweep(i) = calculate_PV_power(panel, G, T, V_sweep(i));
    end
    
    [P_mpp, idx] = max(P_sweep);
    V_mpp = V_sweep(idx);
    
    if isnan(P_mpp)
        P_mpp = 0;
        V_mpp = 0;
    end
end