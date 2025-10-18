clc; clear; close all;

%% ========================================================================
%  MARKOV CHAIN-BASED PV CONDITION SIMULATOR
%  Purpose: Generate realistic Temperature and Irradiance sequences
%           for testing MPPT algorithms and Physics-Informed ANNs
%  Author: Solar PV Modeling Workflow
%  Dataset: 45,000+ samples at 15-minute intervals
%% ========================================================================

fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║   MARKOV CHAIN PV CONDITION SIMULATOR FOR MPPT TESTING     ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

%% STEP 1: Load and Analyze Dataset
fprintf('[STEP 1] Loading PV Dataset...\n');
data = readtable('clean_dataset.csv');

% Display dataset statistics
fprintf('Dataset Summary:\n');
fprintf('  Total Samples: %d\n', height(data));
fprintf('  Temperature Range: %.1f°C to %.1f°C\n', min(data.Temperature), max(data.Temperature));
fprintf('  Irradiance Range: %.1f W/m² to %.1f W/m²\n', min(data.Irradiance), max(data.Irradiance));
fprintf('  Voltage Range: %.2f V to %.2f V\n', min(data.Voltage), max(data.Voltage));
fprintf('  Time Resolution: 15-minute intervals\n\n');

%% STEP 2: Filter Valid Operating Conditions
% Remove low irradiance (nighttime) and invalid data points
fprintf('[STEP 2] Filtering Valid Operating Conditions...\n');
valid_idx = data.Irradiance > 100 & data.Voltage > 1 & data.Current > 0;
data_filtered = data(valid_idx, :);

fprintf('  Valid Samples: %d (%.1f%% of total)\n', height(data_filtered), ...
    100*height(data_filtered)/height(data));
fprintf('  Filtered Temperature: %.1f°C to %.1f°C\n', ...
    min(data_filtered.Temperature), max(data_filtered.Temperature));
fprintf('  Filtered Irradiance: %.1f to %.1f W/m²\n\n', ...
    min(data_filtered.Irradiance), max(data_filtered.Irradiance));

%% STEP 3: Discretize Temperature and Irradiance into Bins
fprintf('[STEP 3] Discretizing Temperature and Irradiance...\n');

% Define number of bins (adjustable for resolution vs. sparsity trade-off)
num_T_bins = 15;  % Temperature bins (e.g., 15 bins covers -20°C to 60°C)
num_G_bins = 20;  % Irradiance bins (e.g., 20 bins covers 100-1200 W/m²)

% Create bin edges
T_min = min(data_filtered.Temperature);
T_max = max(data_filtered.Temperature);
G_min = min(data_filtered.Irradiance);
G_max = max(data_filtered.Irradiance);

T_edges = linspace(T_min, T_max, num_T_bins + 1);
G_edges = linspace(G_min, G_max, num_G_bins + 1);

% Discretize into bins
T_bins = discretize(data_filtered.Temperature, T_edges);
G_bins = discretize(data_filtered.Irradiance, G_edges);

% Calculate bin midpoints (for mapping back to real values later)
T_midpoints = (T_edges(1:end-1) + T_edges(2:end)) / 2;
G_midpoints = (G_edges(1:end-1) + G_edges(2:end)) / 2;

fprintf('  Temperature: %d bins from %.1f°C to %.1f°C (bin width: %.1f°C)\n', ...
    num_T_bins, T_min, T_max, (T_max-T_min)/num_T_bins);
fprintf('  Irradiance: %d bins from %.1f to %.1f W/m² (bin width: %.1f W/m²)\n', ...
    num_G_bins, G_min, G_max, (G_max-G_min)/num_G_bins);

%% STEP 4: Create Combined States (Temperature × Irradiance)
fprintf('\n[STEP 4] Creating Combined States...\n');

% Each state is a unique combination of (T_bin, G_bin)
% State index: state = (T_bin - 1) * num_G_bins + G_bin
% This creates a 1D state space from 2D (T, G) grid

num_states = num_T_bins * num_G_bins;
state_sequence = (T_bins - 1) * num_G_bins + G_bins;

fprintf('  Total Possible States: %d (T×G combinations)\n', num_states);
fprintf('  Observed States in Data: %d\n', length(unique(state_sequence)));
fprintf('  State Coverage: %.1f%%\n\n', 100*length(unique(state_sequence))/num_states);

%% STEP 5: Build Transition Probability Matrix
fprintf('[STEP 5] Computing Transition Probability Matrix...\n');

% Initialize transition count matrix
transition_counts = zeros(num_states, num_states);

% Count transitions: state(t) -> state(t+1)
for i = 1:(length(state_sequence) - 1)
    current_state = state_sequence(i);
    next_state = state_sequence(i + 1);
    
    % Handle NaN states (rare edge cases)
    if ~isnan(current_state) && ~isnan(next_state)
        transition_counts(current_state, next_state) = ...
            transition_counts(current_state, next_state) + 1;
    end
end

% Convert counts to probabilities (row-wise normalization)
transition_matrix = zeros(num_states, num_states);
for i = 1:num_states
    row_sum = sum(transition_counts(i, :));
    if row_sum > 0
        transition_matrix(i, :) = transition_counts(i, :) / row_sum;
    else
        % If state never observed, assume uniform transition (rare case)
        transition_matrix(i, :) = 1 / num_states;
    end
end

% Display transition matrix statistics
non_zero_transitions = sum(transition_matrix(:) > 0);
sparsity = 100 * (1 - non_zero_transitions / numel(transition_matrix));

fprintf('  Transition Matrix Size: %d × %d\n', num_states, num_states);
fprintf('  Non-zero Transitions: %d\n', non_zero_transitions);
fprintf('  Matrix Sparsity: %.1f%%\n', sparsity);
fprintf('  Matrix is Row-Stochastic: %s\n\n', ...
    mat2str(all(abs(sum(transition_matrix, 2) - 1) < 1e-10)));

%% STEP 6: Visualize Transition Matrix
fprintf('[STEP 6] Visualizing Transition Matrix...\n');

figure('Name', 'Markov Transition Matrix', 'Color', 'w', 'Position', [100 100 1000 400]);

subplot(1, 2, 1);
imagesc(transition_matrix);
colorbar; colormap('hot');
title('Full Transition Matrix');
xlabel('Next State'); ylabel('Current State');
set(gca, 'FontSize', 10);

subplot(1, 2, 2);
imagesc(log10(transition_matrix + 1e-10));  % Log scale for better visualization
colorbar; colormap('jet');
title('Transition Matrix (Log Scale)');
xlabel('Next State'); ylabel('Current State');
set(gca, 'FontSize', 10);

%% STEP 7: Simulate Markov Chain Sequence
fprintf('[STEP 7] Simulating PV Condition Sequences...\n');

% Simulation parameters
sim_duration_hours = 24 * 7;  % 1 week simulation
time_step_minutes = 15;       % 15-minute intervals
num_sim_steps = (sim_duration_hours * 60) / time_step_minutes;

fprintf('  Simulation Duration: %.1f hours (%.1f days)\n', ...
    sim_duration_hours, sim_duration_hours/24);
fprintf('  Time Steps: %d (15-min intervals)\n', num_sim_steps);

% Initialize simulation
rng(42);  % For reproducibility

% Choose initial state (use most common state from data)
state_counts = histcounts(state_sequence, 1:(num_states+1));
[~, initial_state] = max(state_counts);

simulated_states = zeros(num_sim_steps, 1);
simulated_states(1) = initial_state;

% Simulate state transitions using Markov chain
fprintf('  Simulating state transitions...\n');
for t = 2:num_sim_steps
    current_state = simulated_states(t - 1);
    
    % Get transition probabilities from current state
    trans_probs = transition_matrix(current_state, :);
    
    % Sample next state using cumulative probability
    cumulative_prob = cumsum(trans_probs);
    rand_val = rand();
    next_state = find(cumulative_prob >= rand_val, 1, 'first');
    
    simulated_states(t) = next_state;
end

fprintf('  ✓ Simulation complete!\n\n');

%% STEP 8: Map States Back to Temperature and Irradiance
fprintf('[STEP 8] Mapping Simulated States to Physical Values...\n');

% Convert state indices back to (T_bin, G_bin)
simulated_T_bins = floor((simulated_states - 1) / num_G_bins) + 1;
simulated_G_bins = mod(simulated_states - 1, num_G_bins) + 1;

% Map bins to midpoint values
simulated_Temperature = T_midpoints(simulated_T_bins)';
simulated_Irradiance = G_midpoints(simulated_G_bins)';

% Add small random noise for realism (within bin width)
T_bin_width = (T_max - T_min) / num_T_bins;
G_bin_width = (G_max - G_min) / num_G_bins;

simulated_Temperature = simulated_Temperature + ...
    (rand(num_sim_steps, 1) - 0.5) * T_bin_width * 0.5;
simulated_Irradiance = simulated_Irradiance + ...
    (rand(num_sim_steps, 1) - 0.5) * G_bin_width * 0.5;

% Create time vector
time_hours = (0:(num_sim_steps-1))' * (time_step_minutes / 60);

fprintf('  Simulated Temperature: %.1f°C to %.1f°C (mean: %.1f°C)\n', ...
    min(simulated_Temperature), max(simulated_Temperature), mean(simulated_Temperature));
fprintf('  Simulated Irradiance: %.1f to %.1f W/m² (mean: %.1f W/m²)\n\n', ...
    min(simulated_Irradiance), max(simulated_Irradiance), mean(simulated_Irradiance));

%% STEP 9: Statistical Comparison (Simulated vs. Real Data)
fprintf('[STEP 9] Statistical Validation...\n');

real_T_mean = mean(data_filtered.Temperature);
real_T_std = std(data_filtered.Temperature);
real_G_mean = mean(data_filtered.Irradiance);
real_G_std = std(data_filtered.Irradiance);

sim_T_mean = mean(simulated_Temperature);
sim_T_std = std(simulated_Temperature);
sim_G_mean = mean(simulated_Irradiance);
sim_G_std = std(simulated_Irradiance);

fprintf('  Temperature:\n');
fprintf('    Real - Mean: %.2f°C, Std: %.2f°C\n', real_T_mean, real_T_std);
fprintf('    Simulated - Mean: %.2f°C, Std: %.2f°C\n', sim_T_mean, sim_T_std);
fprintf('    Error - Mean: %.2f%%, Std: %.2f%%\n', ...
    100*abs(sim_T_mean-real_T_mean)/real_T_mean, ...
    100*abs(sim_T_std-real_T_std)/real_T_std);

fprintf('  Irradiance:\n');
fprintf('    Real - Mean: %.2f W/m², Std: %.2f W/m²\n', real_G_mean, real_G_std);
fprintf('    Simulated - Mean: %.2f W/m², Std: %.2f W/m²\n', sim_G_mean, sim_G_std);
fprintf('    Error - Mean: %.2f%%, Std: %.2f%%\n\n', ...
    100*abs(sim_G_mean-real_G_mean)/real_G_mean, ...
    100*abs(sim_G_std-real_G_std)/real_G_std);

%% STEP 10: Visualize Simulated Sequences
fprintf('[STEP 10] Generating Visualization Plots...\n');

figure('Name', 'Markov Simulated PV Conditions', 'Color', 'w', ...
    'Position', [100 100 1200 700]);

% Plot 1: Temperature Time Series
subplot(3, 2, 1);
plot(time_hours, simulated_Temperature, 'b-', 'LineWidth', 1.2);
xlabel('Time (hours)'); ylabel('Temperature (°C)');
title('Simulated Temperature Profile');
grid on; xlim([0 max(time_hours)]);
yline(mean(simulated_Temperature), 'r--', 'Mean', 'LineWidth', 1.5);

% Plot 2: Irradiance Time Series
subplot(3, 2, 2);
plot(time_hours, simulated_Irradiance, 'Color', [1 0.5 0], 'LineWidth', 1.2);
xlabel('Time (hours)'); ylabel('Irradiance (W/m²)');
title('Simulated Irradiance Profile');
grid on; xlim([0 max(time_hours)]);
yline(mean(simulated_Irradiance), 'r--', 'Mean', 'LineWidth', 1.5);

% Plot 3: Temperature Histogram Comparison
subplot(3, 2, 3);
histogram(data_filtered.Temperature, 30, 'Normalization', 'probability', ...
    'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'k', 'FaceAlpha', 0.6);
hold on;
histogram(simulated_Temperature, 30, 'Normalization', 'probability', ...
    'FaceColor', 'b', 'EdgeColor', 'none', 'FaceAlpha', 0.5);
xlabel('Temperature (°C)'); ylabel('Probability');
title('Temperature Distribution');
legend('Real Data', 'Simulated', 'Location', 'best');
grid on;

% Plot 4: Irradiance Histogram Comparison
subplot(3, 2, 4);
histogram(data_filtered.Irradiance, 30, 'Normalization', 'probability', ...
    'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'k', 'FaceAlpha', 0.6);
hold on;
histogram(simulated_Irradiance, 30, 'Normalization', 'probability', ...
    'FaceColor', [1 0.5 0], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
xlabel('Irradiance (W/m²)'); ylabel('Probability');
title('Irradiance Distribution');
legend('Real Data', 'Simulated', 'Location', 'best');
grid on;

% Plot 5: Temperature-Irradiance Joint Distribution (Real)
subplot(3, 2, 5);
scatter(data_filtered.Temperature, data_filtered.Irradiance, 10, ...
    [0.5 0.5 0.5], 'filled', 'MarkerFaceAlpha', 0.3);
xlabel('Temperature (°C)'); ylabel('Irradiance (W/m²)');
title('Real Data: T-G Joint Distribution');
grid on;

% Plot 6: Temperature-Irradiance Joint Distribution (Simulated)
subplot(3, 2, 6);
scatter(simulated_Temperature, simulated_Irradiance, 10, ...
    'b', 'filled', 'MarkerFaceAlpha', 0.5);
xlabel('Temperature (°C)'); ylabel('Irradiance (W/m²)');
title('Simulated: T-G Joint Distribution');
grid on;

%% STEP 11: Save Results for ANN Integration
fprintf('[STEP 11] Saving Results...\n');

% Create output structure
markov_output.transition_matrix = transition_matrix;
markov_output.T_midpoints = T_midpoints;
markov_output.G_midpoints = G_midpoints;
markov_output.num_T_bins = num_T_bins;
markov_output.num_G_bins = num_G_bins;
markov_output.T_edges = T_edges;
markov_output.G_edges = G_edges;
markov_output.simulated_Temperature = simulated_Temperature;
markov_output.simulated_Irradiance = simulated_Irradiance;
markov_output.time_hours = time_hours;
markov_output.num_states = num_states;

% Save to MAT file
save('Markov_PV_Simulator_Output.mat', 'markov_output');

% Export simulated sequence to CSV for easy use
sim_table = table(time_hours, simulated_Temperature, simulated_Irradiance, ...
    'VariableNames', {'Time_hours', 'Temperature_C', 'Irradiance_Wm2'});
writetable(sim_table, 'Markov_Simulated_Sequence.csv');

fprintf('  ✓ Saved: Markov_PV_Simulator_Output.mat\n');
fprintf('  ✓ Saved: Markov_Simulated_Sequence.csv\n\n');

%% STEP 12: Generate Multiple Scenarios for Robustness Testing
fprintf('[STEP 12] Generating Additional Test Scenarios...\n');

num_scenarios = 5;
scenario_duration_hours = 24;  % 1 day per scenario
scenario_steps = (scenario_duration_hours * 60) / time_step_minutes;

scenarios = struct();

for s = 1:num_scenarios
    % Simulate scenario
    scenario_states = zeros(scenario_steps, 1);
    scenario_states(1) = initial_state;
    
    for t = 2:scenario_steps
        current_state = scenario_states(t - 1);
        trans_probs = transition_matrix(current_state, :);
        cumulative_prob = cumsum(trans_probs);
        next_state = find(cumulative_prob >= rand(), 1, 'first');
        scenario_states(t) = next_state;
    end
    
    % Map to physical values
    scenario_T_bins = floor((scenario_states - 1) / num_G_bins) + 1;
    scenario_G_bins = mod(scenario_states - 1, num_G_bins) + 1;
    
    scenarios(s).Temperature = T_midpoints(scenario_T_bins)' + ...
        (rand(scenario_steps, 1) - 0.5) * T_bin_width * 0.5;
    scenarios(s).Irradiance = G_midpoints(scenario_G_bins)' + ...
        (rand(scenario_steps, 1) - 0.5) * G_bin_width * 0.5;
    scenarios(s).time_hours = (0:(scenario_steps-1))' * (time_step_minutes / 60);
    
    fprintf('  Scenario %d: T∈[%.1f, %.1f]°C, G∈[%.1f, %.1f] W/m²\n', ...
        s, min(scenarios(s).Temperature), max(scenarios(s).Temperature), ...
        min(scenarios(s).Irradiance), max(scenarios(s).Irradiance));
end

save('Markov_PV_Test_Scenarios.mat', 'scenarios');
fprintf('  ✓ Saved: Markov_PV_Test_Scenarios.mat\n\n');

%% STEP 13: Summary and Next Steps
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║                   SIMULATION COMPLETE                      ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

fprintf('SUMMARY:\n');
fprintf('  • Transition Matrix: %d×%d (%.1f%% sparse)\n', ...
    num_states, num_states, sparsity);
fprintf('  • Main Simulation: %d steps (%.1f days)\n', ...
    num_sim_steps, sim_duration_hours/24);
fprintf('  • Test Scenarios: %d scenarios (24 hours each)\n', num_scenarios);
fprintf('\nOUTPUT FILES:\n');
fprintf('  1. Markov_PV_Simulator_Output.mat - Full transition matrix & main sequence\n');
fprintf('  2. Markov_Simulated_Sequence.csv - Main sequence (CSV format)\n');
fprintf('  3. Markov_PV_Test_Scenarios.mat - 5 test scenarios for ANN validation\n');

fprintf('\nNEXT STEPS FOR ANN INTEGRATION:\n');
fprintf('  1. Load: load(''Markov_PV_Simulator_Output.mat'')\n');
fprintf('  2. Use: markov_output.simulated_Temperature & markov_output.simulated_Irradiance\n');
fprintf('  3. Feed to Physics-Informed ANN to predict Vmpp, Impp, Pmpp\n');
fprintf('  4. Compare ANN predictions with real MPP values\n');
fprintf('  5. Use test scenarios for robustness evaluation\n\n');

fprintf('✓ Markov Chain PV Simulator Ready for ANN Testing!\n');