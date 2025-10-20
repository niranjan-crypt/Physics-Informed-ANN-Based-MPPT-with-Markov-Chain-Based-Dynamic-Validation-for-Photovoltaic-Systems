clc; clear; close all;

%% ========================================================================
%  DIRECT INTEGRATION: Markov Simulator + Existing Physics-Informed ANN
%  Purpose: Test your trained ANN on Markov-generated realistic sequences
%  Advantage: Uses your existing trained model - no retraining needed!
%% ========================================================================

fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║     MARKOV SEQUENCES + YOUR TRAINED PHYSICS-ANN (DIRECT)  ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

%% STEP 1: Load Your Trained Physics-Informed ANN
fprintf('[STEP 1] Loading YOUR Trained ANN Model...\n');

if ~isfile('ANN_MPPT_PhysicsInformed.mat')
    error('Please train the Physics-Informed ANN first (run the training code)');
end

load('ANN_MPPT_PhysicsInformed.mat', 'net', 'ps_input', 'ps_target', 'G_ref', 'T_ref');

% Get network architecture info safely
num_inputs = net.numInputs;
num_outputs = net.numOutputs;
hidden_layers = net.numLayers - 1;

fprintf('  ✓ Model loaded: %d inputs → %d hidden layers → %d outputs\n', ...
    num_inputs, hidden_layers, num_outputs);
fprintf('  ✓ Reference: G_ref=%d W/m², T_ref=%d°C\n\n', G_ref, T_ref);

%% STEP 2: Load Markov Simulation Output
fprintf('[STEP 2] Loading Markov-Generated Sequences...\n');

if ~isfile('Markov_PV_Simulator_Output.mat')
    error('Please run the Markov simulator first to generate sequences');
end

load('Markov_PV_Simulator_Output.mat', 'markov_output');

T_sim = markov_output.simulated_Temperature;
G_sim = markov_output.simulated_Irradiance;
time_hours = markov_output.time_hours;

fprintf('  ✓ Loaded %d time steps (%.1f days)\n', length(T_sim), max(time_hours)/24);
fprintf('  Temperature: %.1f to %.1f°C (mean: %.1f°C)\n', ...
    min(T_sim), max(T_sim), mean(T_sim));
fprintf('  Irradiance: %.1f to %.1f W/m² (mean: %.1f W/m²)\n\n', ...
    min(G_sim), max(G_sim), mean(G_sim));

%% STEP 3: Generate Physics-Based Features (same as training)
fprintf('[STEP 3] Generating Physics-Based Features...\n');

G_norm = G_sim / G_ref;
T_delta = T_sim - T_ref;
ln_G = log(G_sim);
G_sqrt = sqrt(G_sim);
T_squared = T_delta.^2;
GT_interaction = G_norm .* T_delta;

% Construct input: [G; T; ln(G); sqrt(G); T²; G×T] - SAME as training
inputs_sim = [G_sim'; T_sim'; ln_G'; G_sqrt'; T_squared'; GT_interaction'];

fprintf('  ✓ Features: [G, T, ln(G), sqrt(G), T², G×T] → %d × %d\n\n', ...
    size(inputs_sim, 1), size(inputs_sim, 2));

%% STEP 4: Normalize and Predict with YOUR Trained ANN
fprintf('[STEP 4] Running ANN Predictions...\n');

inputs_norm = mapminmax('apply', inputs_sim, ps_input);
outputs_norm = net(inputs_norm);
outputs = mapminmax('reverse', outputs_norm, ps_target);

Vmpp_pred = outputs(1, :)';
Impp_pred = outputs(2, :)';
Pmpp_pred = Vmpp_pred .* Impp_pred;

fprintf('  ✓ Predictions complete!\n');
fprintf('  Vmpp: %.2f to %.2f V (mean: %.2f V)\n', ...
    min(Vmpp_pred), max(Vmpp_pred), mean(Vmpp_pred));
fprintf('  Impp: %.2f to %.2f A (mean: %.2f A)\n', ...
    min(Impp_pred), max(Impp_pred), mean(Impp_pred));
fprintf('  Pmpp: %.2f to %.2f W (mean: %.2f W)\n\n', ...
    min(Pmpp_pred), max(Pmpp_pred), mean(Pmpp_pred));

%% STEP 5: Visualize Complete Markov-ANN Workflow
fprintf('[STEP 5] Creating Comprehensive Visualization...\n');

figure('Name', 'Markov-ANN MPPT Prediction Workflow', 'Color', 'w', ...
    'Position', [50 50 1400 900]);

% Plot 1: Temperature Sequence
subplot(3, 3, 1);
plot(time_hours, T_sim, 'b-', 'LineWidth', 1.2);
xlabel('Time (hours)'); ylabel('Temperature (°C)');
title('Markov-Generated Temperature');
grid on; xlim([0 max(time_hours)]);

% Plot 2: Irradiance Sequence
subplot(3, 3, 2);
plot(time_hours, G_sim, 'Color', [1 0.5 0], 'LineWidth', 1.2);
xlabel('Time (hours)'); ylabel('Irradiance (W/m²)');
title('Markov-Generated Irradiance');
grid on; xlim([0 max(time_hours)]);

% Plot 3: T-G Phase Space
subplot(3, 3, 3);
scatter(T_sim, G_sim, 15, time_hours, 'filled', 'MarkerFaceAlpha', 0.6);
colorbar; colormap('jet');
xlabel('Temperature (°C)'); ylabel('Irradiance (W/m²)');
title('T-G Operating Points (colored by time)');
grid on;

% Plot 4: Predicted Vmpp
subplot(3, 3, 4);
plot(time_hours, Vmpp_pred, 'r-', 'LineWidth', 1.2);
xlabel('Time (hours)'); ylabel('Vmpp (V)');
title('ANN-Predicted Vmpp');
grid on; xlim([0 max(time_hours)]);

% Plot 5: Predicted Impp
subplot(3, 3, 5);
plot(time_hours, Impp_pred, 'g-', 'LineWidth', 1.2);
xlabel('Time (hours)'); ylabel('Impp (A)');
title('ANN-Predicted Impp');
grid on; xlim([0 max(time_hours)]);

% Plot 6: Predicted Pmpp
subplot(3, 3, 6);
plot(time_hours, Pmpp_pred, 'k-', 'LineWidth', 1.2);
xlabel('Time (hours)'); ylabel('Pmpp (W)');
title('ANN-Predicted Pmpp');
grid on; xlim([0 max(time_hours)]);

% Plot 7: Vmpp vs Irradiance
subplot(3, 3, 7);
scatter(G_sim, Vmpp_pred, 20, T_sim, 'filled', 'MarkerFaceAlpha', 0.6);
colorbar; xlabel('Irradiance (W/m²)'); ylabel('Vmpp (V)');
title('Vmpp vs G (colored by T)');
grid on;

% Plot 8: Impp vs Irradiance
subplot(3, 3, 8);
scatter(G_sim, Impp_pred, 20, T_sim, 'filled', 'MarkerFaceAlpha', 0.6);
colorbar; xlabel('Irradiance (W/m²)'); ylabel('Impp (A)');
title('Impp vs G (colored by T)');
grid on;

% Plot 9: Power vs Conditions
subplot(3, 3, 9);
scatter3(G_sim, T_sim, Pmpp_pred, 20, Pmpp_pred, 'filled');
colorbar; xlabel('G (W/m²)'); ylabel('T (°C)'); zlabel('Pmpp (W)');
title('3D: Pmpp = f(G, T)');
grid on; view(45, 30);

%% STEP 6: Validation Against Real Data
fprintf('[STEP 6] Validating Against Real Measured Data...\n');

data = readtable('clean_dataset.csv');
data = data(data.Irradiance > 100 & data.Voltage > 1, :);
data.Power = data.Voltage .* data.Current;

% Sample 200 random points from Markov sequence for validation
num_val = min(200, length(T_sim));
val_indices = randperm(length(T_sim), num_val);

validation_results = table();
validation_results.Time = time_hours(val_indices);
validation_results.G_sim = G_sim(val_indices);
validation_results.T_sim = T_sim(val_indices);
validation_results.Vmpp_pred = Vmpp_pred(val_indices);
validation_results.Impp_pred = Impp_pred(val_indices);
validation_results.Pmpp_pred = Pmpp_pred(val_indices);

% Initialize error arrays
errors_V = NaN(num_val, 1);
errors_I = NaN(num_val, 1);
errors_P = NaN(num_val, 1);
valid_count = 0;

tolerance_G = 25;
tolerance_T = 3;

fprintf('  Matching %d simulated points with real MPP...\n', num_val);

for i = 1:num_val
    % Find real data near simulated condition
    idx = abs(data.Irradiance - validation_results.G_sim(i)) <= tolerance_G & ...
          abs(data.Temperature - validation_results.T_sim(i)) <= tolerance_T;
    
    if sum(idx) >= 3
        V_meas = data.Voltage(idx);
        I_meas = data.Current(idx);
        [V_meas, sortIdx] = sort(V_meas);
        I_meas = I_meas(sortIdx);
        P_meas = V_meas .* I_meas;
        
        [Pmax, imax] = max(P_meas);
        Vmpp_real = V_meas(imax);
        Impp_real = I_meas(imax);
        
        valid_count = valid_count + 1;
        errors_V(valid_count) = abs(validation_results.Vmpp_pred(i) - Vmpp_real);
        errors_I(valid_count) = abs(validation_results.Impp_pred(i) - Impp_real);
        errors_P(valid_count) = abs(validation_results.Pmpp_pred(i) - Pmax);
    end
end

errors_V = errors_V(1:valid_count);
errors_I = errors_I(1:valid_count);
errors_P = errors_P(1:valid_count);

fprintf('  ✓ Successfully validated %d/%d points\n\n', valid_count, num_val);

%% STEP 7: Performance Statistics
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║              VALIDATION PERFORMANCE METRICS                ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n');

if valid_count > 0
    fprintf('Vmpp Errors:  MAE = %.3f V  |  RMSE = %.3f V  |  Max = %.3f V\n', ...
        mean(errors_V), sqrt(mean(errors_V.^2)), max(errors_V));
    fprintf('Impp Errors:  MAE = %.3f A  |  RMSE = %.3f A  |  Max = %.3f A\n', ...
        mean(errors_I), sqrt(mean(errors_I.^2)), max(errors_I));
    fprintf('Pmpp Errors:  MAE = %.2f W  |  RMSE = %.2f W  |  Max = %.2f W\n', ...
        mean(errors_P), sqrt(mean(errors_P.^2)), max(errors_P));
    fprintf('Mean Relative Power Error: %.2f%%\n\n', mean(errors_P./Pmpp_pred(val_indices(1:valid_count))) * 100);
    
    % Error distribution
    figure('Name', 'Validation Error Distribution', 'Color', 'w', 'Position', [100 100 1200 400]);
    
    subplot(1,3,1);
    histogram(errors_V, 20, 'FaceColor', 'r', 'EdgeColor', 'k');
    xlabel('|Error| (V)'); ylabel('Count');
    title(sprintf('Vmpp Error Distribution (MAE=%.3fV)', mean(errors_V)));
    grid on;
    
    subplot(1,3,2);
    histogram(errors_I, 20, 'FaceColor', 'g', 'EdgeColor', 'k');
    xlabel('|Error| (A)'); ylabel('Count');
    title(sprintf('Impp Error Distribution (MAE=%.3fA)', mean(errors_I)));
    grid on;
    
    subplot(1,3,3);
    histogram(errors_P, 20, 'FaceColor', 'b', 'EdgeColor', 'k');
    xlabel('|Error| (W)'); ylabel('Count');
    title(sprintf('Pmpp Error Distribution (MAE=%.2fW)', mean(errors_P)));
    grid on;
end

%% STEP 8: Save Predicted Sequences
fprintf('[STEP 8] Saving Prediction Results...\n');

% Save as MAT file
prediction_results.time_hours = time_hours;
prediction_results.Temperature = T_sim;
prediction_results.Irradiance = G_sim;
prediction_results.Vmpp = Vmpp_pred;
prediction_results.Impp = Impp_pred;
prediction_results.Pmpp = Pmpp_pred;

save('Markov_ANN_Predictions.mat', 'prediction_results');

% Save as CSV
output_table = table(time_hours, T_sim, G_sim, Vmpp_pred, Impp_pred, Pmpp_pred, ...
    'VariableNames', {'Time_hours', 'Temperature_C', 'Irradiance_Wm2', ...
                      'Vmpp_V', 'Impp_A', 'Pmpp_W'});
writetable(output_table, 'Markov_ANN_Predictions.csv');

fprintf('  ✓ Saved: Markov_ANN_Predictions.mat\n');
fprintf('  ✓ Saved: Markov_ANN_Predictions.csv\n\n');

%% STEP 9: Interactive I-V/P-V Curve Visualization
fprintf('[STEP 9] Interactive I-V/P-V Curve Visualization\n');
fprintf('────────────────────────────────────────────────────────────\n');

while true
    fprintf('\nEnter conditions to visualize I-V and P-V curves:\n');
    G_test = input('  Irradiance (W/m²) [0 to quit]: ');
    if G_test == 0, break; end
    T_test = input('  Temperature (°C): ');
    
    % Find closest Markov point
    [~, idx_markov] = min((G_sim - G_test).^2 + (T_sim - T_test).^2);
    fprintf('\n  Closest Markov point: G=%.1f W/m², T=%.1f°C\n', ...
        G_sim(idx_markov), T_sim(idx_markov));
    fprintf('  ANN Prediction: Vmpp=%.2fV, Impp=%.2fA, Pmpp=%.2fW\n', ...
        Vmpp_pred(idx_markov), Impp_pred(idx_markov), Pmpp_pred(idx_markov));
    
    % Find real measured data
    idx = abs(data.Irradiance - G_test) <= 20 & abs(data.Temperature - T_test) <= 2;
    if sum(idx) < 3
        idx = abs(data.Irradiance - G_test) <= 40 & abs(data.Temperature - T_test) <= 4;
    end
    
    if sum(idx) >= 3
        V_meas = data.Voltage(idx);
        I_meas = data.Current(idx);
        [V_meas, sortIdx] = sort(V_meas);
        I_meas = I_meas(sortIdx);
        P_meas = V_meas .* I_meas;
        
        [Pmax_meas, imax] = max(P_meas);
        Vmpp_meas = V_meas(imax);
        Impp_meas = I_meas(imax);
        
        fprintf('  Measured MPP: Vmpp=%.2fV, Impp=%.2fA, Pmpp=%.2fW\n', ...
            Vmpp_meas, Impp_meas, Pmax_meas);
        fprintf('  Errors: ΔV=%.2fV (%.1f%%), ΔP=%.2fW (%.1f%%)\n', ...
            Vmpp_pred(idx_markov)-Vmpp_meas, ...
            (Vmpp_pred(idx_markov)-Vmpp_meas)/Vmpp_meas*100, ...
            Pmpp_pred(idx_markov)-Pmax_meas, ...
            (Pmpp_pred(idx_markov)-Pmax_meas)/Pmax_meas*100);
        
        % Plot curves
        figure('Name', 'I-V & P-V Curves with Markov-ANN Prediction', ...
            'Color', 'w', 'Position', [100 100 1200 500]);
        
        subplot(1,2,1);
        plot(V_meas, I_meas, 'b-', 'LineWidth', 2); hold on;
        plot(Vmpp_meas, Impp_meas, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
        plot(Vmpp_pred(idx_markov), Impp_pred(idx_markov), 'gs', ...
            'MarkerSize', 12, 'MarkerFaceColor', 'g');
        xlabel('Voltage (V)'); ylabel('Current (A)');
        title(sprintf('I-V @ G≈%.0f W/m², T≈%.0f°C', G_test, T_test));
        legend('Measured', 'Real MPP', 'Markov-ANN MPP', 'Location', 'best');
        grid on;
        
        subplot(1,2,2);
        plot(V_meas, P_meas, 'k-', 'LineWidth', 2); hold on;
        plot(Vmpp_meas, Pmax_meas, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
        plot(Vmpp_pred(idx_markov), Pmpp_pred(idx_markov), 'gs', ...
            'MarkerSize', 12, 'MarkerFaceColor', 'g');
        xlabel('Voltage (V)'); ylabel('Power (W)');
        title('P-V Curve');
        legend('Measured', 'Real MPP', 'Markov-ANN MPP', 'Location', 'best');
        grid on;
    else
        fprintf('  ⚠ Not enough measured data near this condition\n');
    end
end

fprintf('\n╔════════════════════════════════════════════════════════════╗\n');
fprintf('║             MARKOV-ANN INTEGRATION COMPLETE!               ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');
fprintf('Next Steps:\n');
fprintf('  1. Analyze prediction accuracy across different operating regimes\n');
fprintf('  2. Test with multiple Markov scenarios (load Markov_PV_Test_Scenarios.mat)\n');
fprintf('  3. Compare ANN performance vs traditional MPPT algorithms\n');
fprintf('  4. Export results for publication/reporting\n\n');