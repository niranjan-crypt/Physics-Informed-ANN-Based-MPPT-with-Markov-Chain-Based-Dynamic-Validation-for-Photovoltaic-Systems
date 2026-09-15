clc; clear; close all;
%% HYBRID PHYSICS-INFORMED ANN FOR MPPT
% Combines neural network with physical PV equations for better accuracy
%
% USER MODIFICATIONS:
%   - Epochs: 500
%   - Network: [18, 15, 12]
%   - Overfitting Test: Statistical (t-test, Cohen's d)
%   - Data Split: Group-based (No data leakage)
%
%% STEP 1: Load and Filter Data
data = readtable('clean_dataset.csv');
data = data(data.Irradiance > 100 & data.Voltage > 1, :);
data.Power = data.Voltage .* data.Current;
fprintf('=== HYBRID PHYSICS-INFORMED APPROACH ===\n');
fprintf('Total samples: %d\n', height(data));
%% STEP 2: Extract MPP with Physics Features
G_bins = discretize(data.Irradiance, 50);
T_bins = discretize(data.Temperature, 20);
[groups, ~] = findgroups(G_bins, T_bins);
Vmpp = splitapply(@(v,p) v(p==max(p)), data.Voltage, data.Power, groups);
Impp = splitapply(@(i,p) i(p==max(p)), data.Current, data.Power, groups);
Gmpp = splitapply(@mean, data.Irradiance, groups);
Tmpp = splitapply(@mean, data.Temperature, groups);
valid = ~isnan(Vmpp);
Vmpp = Vmpp(valid); Impp = Impp(valid);
Gmpp = Gmpp(valid); Tmpp = Tmpp(valid);
% Get the number of *original* MPP data points (before augmentation)
N_original_groups = numel(Gmpp);
fprintf('Total unique G/T conditions (groups): %d\n', N_original_groups);
%% STEP 3: ADD PHYSICS-BASED FEATURES
% These features help the network learn physical relationships
fprintf('\nAdding physics-based features...\n');
% Feature engineering based on PV cell physics
G_ref = 1000;  % Reference irradiance
T_ref = 25;    % Reference temperature (°C)
% Normalized features
G_norm = Gmpp / G_ref;
T_delta = Tmpp - T_ref;
% Physics-informed features
ln_G = log(Gmpp);                    % Logarithmic irradiance (Isc relationship)
G_sqrt = sqrt(Gmpp);                 % Square root (nonlinear effects)
T_squared = (Tmpp - T_ref).^2;       % Temperature nonlinearity
GT_interaction = G_norm .* T_delta;  % G-T coupling
%% STEP 4: Data Augmentation with Physics Constraints
augmentation_factor = 2;
noise_G = 15;
noise_T = 1.5;
noise_V = 0.12;  % Reduced noise
noise_I = 0.06;
Gmpp_aug = []; Tmpp_aug = []; Vmpp_aug = []; Impp_aug = [];
for i = 1:numel(Gmpp)
    Gmpp_aug = [Gmpp_aug; Gmpp(i)];
    Tmpp_aug = [Tmpp_aug; Tmpp(i)];
    Vmpp_aug = [Vmpp_aug; Vmpp(i)];
    Impp_aug = [Impp_aug; Impp(i)];
    
    for j = 1:augmentation_factor
        G_new = Gmpp(i) + noise_G * randn();
        T_new = Tmpp(i) + noise_T * randn();
        
        % Physics-based augmentation:
        temp_coeff_V = -0.003;  % per °C
        irrad_coeff_I = Impp(i) / Gmpp(i);
        
        V_new = Vmpp(i) * (1 + temp_coeff_V * (T_new - Tmpp(i))) + noise_V * randn();
        I_new = irrad_coeff_I * G_new + noise_I * randn();
        
        % Clip to valid ranges
        G_new = max(100, min(1200, G_new));
        T_new = max(-20, min(60, T_new));
        V_new = max(30, min(50, V_new));
        I_new = max(0.1, min(15, I_new));
        
        Gmpp_aug = [Gmpp_aug; G_new];
        Tmpp_aug = [Tmpp_aug; T_new];
        Vmpp_aug = [Vmpp_aug; V_new];
        Impp_aug = [Impp_aug; I_new];
    end
end
fprintf('Augmented samples: %d\n', numel(Gmpp_aug));
%% STEP 5: Create Extended Feature Set
G_norm_aug = Gmpp_aug / G_ref;
T_delta_aug = Tmpp_aug - T_ref;
ln_G_aug = log(Gmpp_aug);
G_sqrt_aug = sqrt(Gmpp_aug);
T_squared_aug = T_delta_aug.^2;
GT_interaction_aug = G_norm_aug .* T_delta_aug;
% Extended input features: [G, T, ln(G), sqrt(G), T², G×T]
inputs = [Gmpp_aug'; Tmpp_aug'; ln_G_aug'; G_sqrt_aug'; ...
          T_squared_aug'; GT_interaction_aug'];  % 6×N
targets = [Vmpp_aug'; Impp_aug'];                % 2×N
fprintf('Input features: %d\n', size(inputs, 1));
%% ========================================================================
%  STEP 6: Train with Group-Based Split (Data Leakage Fixed)
%  ========================================================================
N_total_aug = size(inputs, 2);
rng(42);
% --- START: DATA LEAKAGE FIX (GROUP-BASED SPLIT) ---
% 1. Define the size of each "group" (1 original + N augmentations)
group_size = 1 + augmentation_factor;
if N_total_aug / group_size ~= N_original_groups
    error('Mismatch in augmentation logic. Check augmentation_factor.');
end
% 2. Get indices for the ORIGINAL groups (from STEP 2)
[trainGroups, valGroups, testGroups] = dividerand(N_original_groups, 0.7, 0.15, 0.15);
% 3. Expand the "group" indices to the full "augmented data" indices
trainInd = [];
for g = trainGroups
    start_idx = (g - 1) * group_size + 1;
    end_idx = g * group_size;
    trainInd = [trainInd, start_idx:end_idx];
end
valInd = [];
for g = valGroups
    start_idx = (g - 1) * group_size + 1;
    end_idx = g * group_size;
    valInd = [valInd, start_idx:end_idx];
end
testInd = [];
for g = testGroups
    start_idx = (g - 1) * group_size + 1;
    end_idx = g * group_size;
    testInd = [testInd, start_idx:end_idx];
end
fprintf('Group-based split: Train=%d, Val=%d, Test=%d (Total: %d)\n', ...
    numel(trainInd), numel(valInd), numel(testInd), N_total_aug);
% --- END: DATA LEAKAGE FIX ---
[inp_norm, ps_input] = mapminmax(inputs(:, trainInd));
[tgt_norm, ps_target] = mapminmax(targets(:, trainInd));
inputs_norm = mapminmax('apply', inputs, ps_input);
targets_norm = mapminmax('apply', targets, ps_target);
% --- USER-DEFINED NETWORK PARAMETERS ---
hiddenLayerSize = [18, 15, 12];
net = fitnet(hiddenLayerSize, 'trainbr');
net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = testInd;
net.trainParam.epochs = 500; % User-defined epoch count
net.trainParam.max_fail = 25;
net.trainParam.min_grad = 1e-8;
% --- END USER-DEFINED PARAMETERS ---
fprintf('\n=== TRAINING PHYSICS-INFORMED ANN ===\n');
fprintf('Network: [%s] | Features: 6 | Outputs: 2\n', num2str(hiddenLayerSize));
[net, tr] = train(net, inputs_norm, targets_norm);
%% === STEP 6.5: OVERFITTING ANALYSIS (STATISTICAL APPROACH) ===
fprintf('\n=== OVERFITTING ANALYSIS (STATISTICAL) ===\n');
% Get normalized outputs and targets for each set
train_out = net(inputs_norm(:, tr.trainInd));
val_out   = net(inputs_norm(:, tr.valInd));
test_out  = net(inputs_norm(:, tr.testInd));
train_tgt = targets_norm(:, tr.trainInd);
val_tgt   = targets_norm(:, tr.valInd);
test_tgt  = targets_norm(:, tr.testInd);
% Calculate normalized RMSE (for reference)
train_rmse_norm = sqrt(mean((train_tgt(:) - train_out(:)).^2));
val_rmse_norm   = sqrt(mean((val_tgt(:) - val_out(:)).^2));
test_rmse_norm  = sqrt(mean((test_tgt(:) - test_out(:)).^2));
fprintf('Normalized RMSE: Train=%.4f, Val=%.4f, Test=%.4f\n', ...
        train_rmse_norm, val_rmse_norm, test_rmse_norm);
fprintf('Best epoch: %d\n', tr.best_epoch);
% --- START: Robust Statistical Test for Overfitting ---
% Compare the distribution of *absolute errors* between train and test sets
train_errors_vec = abs(train_tgt(:) - train_out(:));
test_errors_vec  = abs(test_tgt(:) - test_out(:));
% 1. Statistical Significance (t-test)
% H0 (Null Hypothesis): Train errors and Test errors come from the same distribution.
[h, p_value] = ttest2(train_errors_vec, test_errors_vec);
fprintf('\nStatistical Significance Test (Train vs. Test Errors):\n');
fprintf('  p-value: %.4f\n', p_value);
if p_value >= 0.05
    fprintf('   RESULT: p >= 0.05. No statistically significant difference.\n');
    fprintf('   CONCLUSION: Model generalizes well. NO overfitting detected.\n');
else
    fprintf('  RESULT: p < 0.05. Statistically significant difference.\n');
    fprintf('  CONCLUSION: Overfitting is statistically significant.\n');
end
% 2. Effect Size (Cohen's d)
% Measures *how large* the difference is (if it exists)
mean_train_err = mean(train_errors_vec);
mean_test_err  = mean(test_errors_vec);
std_train_err  = std(train_errors_vec);
std_test_err   = std(test_errors_vec);
n_train = numel(train_errors_vec);
n_test  = numel(test_errors_vec);
pooled_std = sqrt( ((n_train-1)*std_train_err^2 + (n_test-1)*std_test_err^2) / (n_train + n_test - 2) );
cohens_d = abs(mean_train_err - mean_test_err) / pooled_std;
fprintf('Effect Size (Cohen''s d): %.3f\n', cohens_d);
if cohens_d < 0.2
    fprintf('  Interpretation: Trivial difference (Excellent)\n');
elseif cohens_d < 0.5
    fprintf('  Interpretation: Small difference (Good)\n');
elseif cohens_d < 0.8
    fprintf('  Interpretation: Medium difference (Warning)\n');
else
    fprintf('  Interpretation: Large difference (Significant Overfitting)\n');
end
% --- END: Robust Statistical Test ---
figure;
plotperform(tr);
title('Training Performance (Overfitting Analysis)');
grid on;
%% === STEP 7: EVALUATE PERFORMANCE (WITH ENGINEERING SPECS) ===
outputs_norm = net(inputs_norm);
outputs = mapminmax('reverse', outputs_norm, ps_target);
Vmpp_pred = outputs(1, :);
Impp_pred = outputs(2, :);
Pmpp_pred = Vmpp_pred .* Impp_pred;
Vmpp_true = targets(1, :);
Impp_true = targets(2, :);
Pmpp_true = Vmpp_true .* Impp_true;
% --- START: Engineering Specification Analysis ---
% Define performance specification (e.g., must be within 25W)
spec_tolerance_W = 25.0; 
% --- END: Engineering Specification Analysis ---
sets = {trainInd, valInd, testInd};
names = {'Train', 'Val', 'Test'};
fprintf('\n=== POWER PREDICTION PERFORMANCE ===\n');
fprintf('Engineering Spec: Target Power Error < %.1fW\n\n', spec_tolerance_W);
for i = 1:3
    P_true = Pmpp_true(sets{i});
    P_pred = Pmpp_pred(sets{i});
    
    % Calculate standard metrics
    rmse_p = sqrt(mean((P_true - P_pred).^2));
    mae_p = mean(abs(P_true - P_pred));
    mape = mean(abs((P_true - P_pred)./P_true)) * 100;
    r2_p = 1 - sum((P_true - P_pred).^2)/sum((P_true - mean(P_true)).^2);
    
    fprintf('%s Set Performance:\n', names{i});
    fprintf('  Metrics: RMSE=%.2fW | MAE=%.2fW | MAPE=%.2f%% | R²=%.5f\n', ...
        rmse_p, mae_p, mape, r2_p);
    
    % --- Engineering Analysis for this set ---
    P_errors_real = abs(P_true - P_pred);
    
    % 1. Worst-Case Analysis
    max_error_W = max(P_errors_real);
    fprintf('  Engineering: Max Error = %.2fW\n', max_error_W);
    % 2. Specification Compliance
    within_spec = sum(P_errors_real < spec_tolerance_W) / numel(P_errors_real);
    fprintf('  Engineering: Compliance = %.1f%% within spec (%.1fW)\n\n', ...
        within_spec*100, spec_tolerance_W);
end
save('ANN_MPPT_PhysicsInformed.mat', 'net', 'ps_input', 'ps_target', 'G_ref', 'T_ref');
fprintf('Physics-informed model saved.\n');
%% === STEP 7.5: ESSENTIAL VISUALIZATION (Test Set) ===
% This section is added as requested, based on the first script's plots.
% --- PLOTS FOR Vmpp ---
figure('Position', [100 100 1500 500]);
sgtitle('Vmpp Performance Plots (Test Set)', 'FontSize', 14, 'FontWeight', 'bold');
% Plot 1: Predicted vs Actual (Vmpp)
subplot(1,3,1);
scatter(Vmpp_true(testInd), Vmpp_pred(testInd), 30, 'b', 'filled', 'MarkerFaceAlpha', 0.7);
hold on;
plot([min(Vmpp_true) max(Vmpp_true)], [min(Vmpp_true) max(Vmpp_true)], 'r--', 'LineWidth', 2);
grid on;
xlabel('Actual Vmpp (V)');
ylabel('Predicted Vmpp (V)');
title('Predicted vs. Actual');
axis equal tight;
% Plot 2: Error Distribution (Vmpp)
subplot(1,3,2);
Vmpp_errors_test = Vmpp_pred(testInd) - Vmpp_true(testInd);
histogram(Vmpp_errors_test, 30);
grid on;
xlabel('Prediction Error (V)');
ylabel('Frequency');
title('Error Distribution');
xline(0, 'r--', 'LineWidth', 2);
% Plot 3: Residual Plot (Vmpp)
subplot(1,3,3);
scatter(Vmpp_pred(testInd), Vmpp_errors_test, 30, 'b', 'filled', 'MarkerFaceAlpha', 0.7);
hold on;
yline(0, 'r--', 'LineWidth', 2);
grid on;
xlabel('Predicted Vmpp (V)');
ylabel('Residual Error (V)');
title('Residuals vs. Predicted Values');
% --- PLOTS FOR Pmpp (Power) ---
figure('Position', [150 150 1500 500]);
sgtitle('Pmpp Performance Plots (Test Set)', 'FontSize', 14, 'FontWeight', 'bold');
% Plot 1: Predicted vs Actual (Pmpp)
subplot(1,3,1);
scatter(Pmpp_true(testInd), Pmpp_pred(testInd), 30, 'r', 'filled', 'MarkerFaceAlpha', 0.7);
hold on;
plot([min(Pmpp_true) max(Pmpp_true)], [min(Pmpp_true) max(Pmpp_true)], 'r--', 'LineWidth', 2);
grid on;
xlabel('Actual Pmpp (W)');
ylabel('Predicted Pmpp (W)');
title('Predicted vs. Actual');
axis equal tight;
% Plot 2: Error Distribution (Pmpp)
subplot(1,3,2);
Pmpp_errors_test = Pmpp_pred(testInd) - Pmpp_true(testInd);
histogram(Pmpp_errors_test, 30);
grid on;
xlabel('Prediction Error (W)');
ylabel('Frequency');
title('Error Distribution');
xline(0, 'r--', 'LineWidth', 2);
% Plot 3: Residual Plot (Pmpp)
subplot(1,3,3);
scatter(Pmpp_pred(testInd), Pmpp_errors_test, 30, 'r', 'filled', 'MarkerFaceAlpha', 0.7);
hold on;
yline(0, 'r--', 'LineWidth', 2);
grid on;
xlabel('Predicted Pmpp (W)');
ylabel('Residual Error (W)');
title('Residuals vs. Predicted Values');
%% === END OF ADDED PLOTS ===

%% === STEP 7.6: ADDED 3D ANN RESPONSE SURFACE PLOT ===
fprintf('\nGenerating 3D ANN Response Surface...\n');

% 1. Create a high-resolution grid of G and T
% Use the min/max of the *entire* augmented dataset for a full surface
G_range = linspace(min(inputs(1,:)), max(inputs(1,:)), 50);
T_range = linspace(min(inputs(2,:)), max(inputs(2,:)), 50);
[G_grid, T_grid] = meshgrid(G_range, T_range);

% 2. Create the 6-feature input matrix from the grid
N_grid = numel(G_grid);
G_norm_grid = G_grid(:) / G_ref;
T_delta_grid = T_grid(:) - T_ref;

ln_G_grid = log(G_grid(:));
G_sqrt_grid = sqrt(G_grid(:));
T_squared_grid = T_delta_grid.^2;
GT_interaction_grid = G_norm_grid .* T_delta_grid;

% Full 6-feature input grid
grid_inputs = [G_grid(:)'; T_grid(:)'; ln_G_grid'; G_sqrt_grid'; ...
               T_squared_grid'; GT_interaction_grid'];

% 3. Normalize, Predict, and Un-normalize
grid_inputs_norm = mapminmax('apply', grid_inputs, ps_input);
grid_outputs_norm = net(grid_inputs_norm);
grid_outputs = mapminmax('reverse', grid_outputs_norm, ps_target);

% 4. Calculate Predicted Power and reshape for plotting
Vmpp_grid = grid_outputs(1, :);
Impp_grid = grid_outputs(2, :);
Pmpp_grid = Vmpp_grid .* Impp_grid;

% Reshape the final Pmpp back into a 50x50 matrix for 'surf'
Pmpp_surface = reshape(Pmpp_grid, 50, 50);

% 5. Plot the 3D Surface
figure('Position', [200 200 900 700]);
surf(G_grid, T_grid, Pmpp_surface, 'FaceAlpha', 0.8, 'EdgeColor', 'none');
hold on;

% 6. Overlay the ACTUAL test data points
scatter3(inputs(1, testInd), inputs(2, testInd), Pmpp_true(testInd), ...
         50, 'r', 'filled', 'MarkerFaceAlpha', 0.7);

title('ANN Learned Response Surface (G, T) \rightarrow Pmpp', 'FontSize', 14);
xlabel('Irradiance (W/m²)');
ylabel('Temperature (°C)');
zlabel('Predicted Max Power (Pmpp)');
legend('ANN Learned Surface', 'Actual Test Data Points', 'Location', 'best');
colorbar;
grid on;
view(30, 25); % Set a good 3D viewing angle
fprintf('✓ 3D Surface Plot complete.\n');

%% === END OF 3D PLOT CODE ===


%% STEP 8: Interactive Testing
fprintf('\n=== INTERACTIVE TESTING ===\n');
while true
    G_input = input('Enter Irradiance (W/m²) [0 to quit]: ');
    if G_input == 0, break; end
    T_input = input('Enter Temperature (°C): ');
    
    % Create extended features
    G_norm_test = G_input / G_ref;
    T_delta_test = T_input - T_ref;
    ln_G_test = log(G_input);
    G_sqrt_test = sqrt(G_input);
    T_squared_test = T_delta_test^2;
    GT_interaction_test = G_norm_test * T_delta_test;
    
    test_input = [G_input; T_input; ln_G_test; G_sqrt_test; ...
                  T_squared_test; GT_interaction_test];
    test_input_norm = mapminmax('apply', test_input, ps_input);
    pred_norm = net(test_input_norm);
    pred_output = mapminmax('reverse', pred_norm, ps_target);
    
    Vmpp_pred_val = pred_output(1);
    Impp_pred_val = pred_output(2);
    Pmpp_pred_val = Vmpp_pred_val * Impp_pred_val;
    
    % Find measured MPP
    idx_close = abs(data.Irradiance - G_input) <= 20 & ...
                abs(data.Temperature - T_input) <= 2;
    if sum(idx_close) < 3
        idx_close = abs(data.Irradiance - G_input) <= 40 & ...
                    abs(data.Temperature - T_input) <= 4;
    end
    
    if sum(idx_close) >= 3
        V_meas = data.Voltage(idx_close);
        I_meas = data.Current(idx_close);
        [V_meas, sortIdx] = sort(V_meas);
        I_meas = I_meas(sortIdx);
        P_meas = V_meas .* I_meas;
        
        [Pmax_meas, iP] = max(P_meas);
        Vmpp_meas = V_meas(iP);
        Impp_meas = I_meas(iP);
        
        fprintf('\n=== RESULTS ===\n');
        fprintf('ANN: Vmpp=%.2fV | Impp=%.2fA | P=%.2fW\n', ...
            Vmpp_pred_val, Impp_pred_val, Pmpp_pred_val);
        fprintf('Measured: Vmpp=%.2fV | Impp=%.2fA | P=%.2fW\n', ...
            Vmpp_meas, Impp_meas, Pmax_meas);
        fprintf('Power Error: %.2fW (%.2f%%)\n\n', ...
            Pmpp_pred_val - Pmax_meas, (Pmpp_pred_val - Pmax_meas)/Pmax_meas*100);
    else
        fprintf('Predicted: Vmpp=%.2fV | Impp=%.2fA | P=%.2fW\n\n', ...
            Vmpp_pred_val, Impp_pred_val, Pmpp_pred_val);
    end
%% === ENHANCED I-V AND P-V CHARACTERISTICS PLOT WITH MPP MARKERS ===
fprintf('\n=== ENHANCED I-V AND P-V CHARACTERISTICS WITH MPP ===\n');
% Choose test condition
G_test = input('Enter Irradiance (W/m²) for curve plotting: ');
T_test = input('Enter Temperature (°C): ');
% Filter experimental points near this condition
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
    
    % Find ACTUAL measured MPP
    [Pmax_meas, idx_mpp] = max(P_meas);
    Vmpp_meas = V_meas(idx_mpp);
    Impp_meas = I_meas(idx_mpp);
    
    % Get ANN predicted MPP for same conditions
    G_norm_test = G_test / G_ref;
    T_delta_test = T_test - T_ref;
    ln_G_test = log(G_test);
    G_sqrt_test = sqrt(G_test);
    T_squared_test = T_delta_test^2;
    GT_interaction_test = G_norm_test * T_delta_test;
    
    test_input = [G_test; T_test; ln_G_test; G_sqrt_test; ...
                  T_squared_test; GT_interaction_test];
    test_input_norm = mapminmax('apply', test_input, ps_input);
    pred_norm = net(test_input_norm);
    pred_output = mapminmax('reverse', pred_norm, ps_target);
    
    Vmpp_ann = pred_output(1);
    Impp_ann = pred_output(2);
    Pmpp_ann = Vmpp_ann * Impp_ann;
    
    % Create enhanced figure
    figure('Position', [100, 100, 1200, 500]);
    
    % Subplot 1: I-V Curve
    subplot(1,2,1);
    plot(V_meas, I_meas, 'b-', 'LineWidth', 2); hold on;
    % Mark ACTUAL MPP
    plot(Vmpp_meas, Impp_meas, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r', ...
         'DisplayName', sprintf('Measured MPP (%.1fV, %.1fA)', Vmpp_meas, Impp_meas));
    % Mark ANN Predicted MPP
    plot(Vmpp_ann, Impp_ann, 'gs', 'MarkerSize', 10, 'MarkerFaceColor', 'g', ...
         'DisplayName', sprintf('ANN MPP (%.1fV, %.1fA)', Vmpp_ann, Impp_ann));
    
    grid on; xlabel('Voltage (V)'); ylabel('Current (A)');
    title(sprintf('I-V Curve @ %.0f W/m², %.0f°C', G_test, T_test));
    legend('show', 'Location', 'best');
    
    % Subplot 2: P-V Curve
    subplot(1,2,2);
    plot(V_meas, P_meas, 'r-', 'LineWidth', 2); hold on;
    % Mark ACTUAL MPP
    plot(Vmpp_meas, Pmax_meas, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r', ...
         'DisplayName', sprintf('Measured MPP (%.1fV, %.1fW)', Vmpp_meas, Pmax_meas));
    % Mark ANN Predicted MPP
    plot(Vmpp_ann, Pmpp_ann, 'gs', 'MarkerSize', 10, 'MarkerFaceColor', 'g', ...
         'DisplayName', sprintf('ANN MPP (%.1fV, %.1fW)', Vmpp_ann, Pmpp_ann));
    
    grid on; xlabel('Voltage (V)'); ylabel('Power (W)');
    title('P-V Curve with MPP Comparison');
    legend('show', 'Location', 'best');
    
    % Add text annotations with errors
    fprintf('\n=== MPP COMPARISON ===\n');
    fprintf('Measured MPP:  V=%.2fV, I=%.2fA, P=%.2fW\n', Vmpp_meas, Impp_meas, Pmax_meas);
    fprintf('ANN Pred MPP:  V=%.2fV, I=%.2fA, P=f.2W\n', Vmpp_ann, Impp_ann, Pmpp_ann);
    fprintf('Voltage Error: %.2fV (%.2f%%)\n', Vmpp_ann - Vmpp_meas, (Vmpp_ann - Vmpp_meas)/Vmpp_meas*100);
    fprintf('Power Error:   %.2fW (%.2f%%)\n', Pmpp_ann - Pmax_meas, (Pmpp_ann - Pmax_meas)/Pmax_meas*100);
    
else
    fprintf('Not enough measured data near this condition to plot I-V/P-V curve.\n');
end
end
fprintf('Session ended.\n');