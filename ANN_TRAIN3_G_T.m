
clc; clear; close all;

%% STEP 1: Load and analyze raw dataset
data = readtable('clean_dataset.csv');
fprintf('=== DATASET ANALYSIS ===\n');
fprintf('Total rows: %d\n', height(data));
fprintf('Voltage range: %.2f to %.2f V\n', min(data.Voltage), max(data.Voltage));
fprintf('Current range: %.2f to %.2f A\n', min(data.Current), max(data.Current));
fprintf('Irradiance range: %.1f to %.1f W/m²\n', min(data.Irradiance), max(data.Irradiance));
fprintf('Temperature range: %.1f to %.1f °C\n\n', min(data.Temperature), max(data.Temperature));

%% STEP 2: Visualize raw data distribution
figure('Position', [50 50 1400 400]);

subplot(1,3,1);
histogram(data.Voltage, 50, 'FaceColor', [0.2 0.6 0.8]);
xlabel('Voltage (V)'); ylabel('Frequency');
title('Raw Voltage Distribution');
grid on;

subplot(1,3,2);
histogram(data.Current, 50, 'FaceColor', [0.8 0.4 0.2]);
xlabel('Current (A)'); ylabel('Frequency');
title('Raw Current Distribution');
grid on;

subplot(1,3,3);
data.Power = data.Voltage .* data.Current;
scatter(data.Voltage, data.Current, 10, data.Power, 'filled');
xlabel('Voltage (V)'); ylabel('Current (A)');
title('I-V Curve (colored by Power)');
colorbar; grid on;

%% STEP 3: Filter dataset with relaxed voltage threshold
% CRITICAL FIX: Lower voltage threshold to include more operating points
data = data(data.Irradiance > 100 & data.Voltage > 1, :); % Changed from 5V to 1V
data.Power = data.Voltage .* data.Current;

fprintf('After filtering (G>100, V>1): %d rows\n', height(data));
fprintf('Filtered voltage range: %.2f to %.2f V\n\n', min(data.Voltage), max(data.Voltage));

%% STEP 4: Extract Vmpp more carefully
% Method 1: Direct MPP extraction (voltage at max power for each condition)
G_bins = discretize(data.Irradiance, 50);
T_bins = discretize(data.Temperature, 20);

% Find Vmpp at maximum power for each (G,T) bin
[groups, ~] = findgroups(G_bins, T_bins);

% Extract MPP parameters
vmpp_method1 = splitapply(@(v,p) v(p==max(p)), data.Voltage, data.Power, groups);
G_mpp1 = splitapply(@mean, data.Irradiance, groups);
T_mpp1 = splitapply(@mean, data.Temperature, groups);
P_max1 = splitapply(@max, data.Power, groups);

% Remove invalid entries
valid1 = ~isnan(vmpp_method1) & ~isnan(G_mpp1) & ~isnan(T_mpp1) & P_max1 > 1;
vmpp_method1 = vmpp_method1(valid1);
G_mpp1 = G_mpp1(valid1);
T_mpp1 = T_mpp1(valid1);

fprintf('Method 1 (Direct MPP): %d points extracted\n', length(vmpp_method1));
fprintf('Vmpp range: %.2f to %.2f V\n\n', min(vmpp_method1), max(vmpp_method1));

% Method 2: Use multiple voltage points if available (for better coverage)
% Sample points around MPP region (70%-95% of Voc)
v_threshold_low = 0.60;  % 60% of max voltage
v_threshold_high = 0.95; % 95% of max voltage
v_max = max(data.Voltage);

% Get data in MPP region
mpp_region = data.Voltage >= v_threshold_low*v_max & data.Voltage <= v_threshold_high*v_max;
data_mpp_region = data(mpp_region, :);

if height(data_mpp_region) > 200
    % Use samples from MPP region for better training
    vmpp_method2 = data_mpp_region.Voltage;
    G_mpp2 = data_mpp_region.Irradiance;
    T_mpp2 = data_mpp_region.Temperature;
    
    % Downsample if too many points
    if length(vmpp_method2) > 2000
        idx_sample = randsample(length(vmpp_method2), 2000);
        vmpp_method2 = vmpp_method2(idx_sample);
        G_mpp2 = G_mpp2(idx_sample);
        T_mpp2 = T_mpp2(idx_sample);
    end
    
    fprintf('Method 2 (MPP Region): %d points extracted\n', length(vmpp_method2));
    fprintf('Vmpp range: %.2f to %.2f V\n\n', min(vmpp_method2), max(vmpp_method2));
    
    % Combine both methods
    vmpp_combined = [vmpp_method1; vmpp_method2];
    G_combined = [G_mpp1; G_mpp2];
    T_combined = [T_mpp1; T_mpp2];
    
    fprintf('Combined dataset: %d points\n', length(vmpp_combined));
else
    % Use only method 1
    vmpp_combined = vmpp_method1;
    G_combined = G_mpp1;
    T_combined = T_mpp1;
    fprintf('Using Method 1 only (insufficient MPP region data)\n');
end

%% STEP 5: Prepare final dataset
inputs = [G_combined'; T_combined'];
targets = vmpp_combined';
N = size(inputs, 2);

fprintf('\n=== FINAL TRAINING DATASET ===\n');
fprintf('Total samples: %d\n', N);
fprintf('Vmpp range: %.2f to %.2f V\n', min(targets), max(targets));
fprintf('Irradiance range: %.1f to %.1f W/m²\n', min(inputs(1,:)), max(inputs(1,:)));
fprintf('Temperature range: %.1f to %.1f °C\n\n', min(inputs(2,:)), max(inputs(2,:)));

%% STEP 6: Shuffle and split dataset
rng(42);
idx = randperm(N);
inputs = inputs(:, idx);
targets = targets(:, idx);

n_train = round(0.7 * N);
n_val = round(0.15 * N);
n_test = N - n_train - n_val;

trainInd = 1:n_train;
valInd = (n_train+1):(n_train+n_val);
testInd = (n_train+n_val+1):N;

fprintf('Data split: %d train, %d val, %d test\n', n_train, n_val, n_test);

%% STEP 7: Normalize data
[inp_norm, ps_input] = mapminmax(inputs(:, trainInd), -1, 1);
[tgt_norm, ps_target] = mapminmax(targets(:, trainInd), -1, 1);

inputs_norm = mapminmax('apply', inputs, ps_input);
targets_norm = mapminmax('apply', targets, ps_target);

%% STEP 8: Create and configure ANN
hiddenLayerSize = [20, 15];  % Slightly larger for better coverage

net = fitnet(hiddenLayerSize, 'trainlm');

net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = testInd;

net.trainParam.epochs = 1500;
net.trainParam.goal = 1e-6;
net.trainParam.max_fail = 25;
net.trainParam.min_grad = 1e-8;
net.trainParam.showWindow = true;
net.performFcn = 'mse';

%% STEP 9: Train network
fprintf('\n=== TRAINING ANN: (G,T) → Vmpp ===\n');
[net, tr] = train(net, inputs_norm, targets_norm);

%% STEP 10: Evaluate performance
outputs_norm = net(inputs_norm);
outputs = mapminmax('reverse', outputs_norm, ps_target);

sets = {trainInd, valInd, testInd};
set_names = {'Training', 'Validation', 'Test'};

fprintf('\n=== PERFORMANCE METRICS ===\n');
for i = 1:3
    idx_set = sets{i};
    y_true = targets(idx_set);
    y_pred = outputs(idx_set);
    
    rmse = sqrt(mean((y_true - y_pred).^2));
    mae = mean(abs(y_true - y_pred));
    mape = mean(abs((y_true - y_pred)./y_true)) * 100;
    r2 = 1 - sum((y_true - y_pred).^2) / sum((y_true - mean(y_true)).^2);
    
    fprintf('%s: RMSE=%.3fV | MAE=%.3fV | MAPE=%.2f%% | R²=%.4f\n', ...
        set_names{i}, rmse, mae, mape, r2);
end

%% STEP 11: Comprehensive visualization
figure('Position', [100 100 1400 900]);

% Plot 1: Predicted vs Actual
subplot(2,3,1);
scatter(targets(testInd), outputs(testInd), 40, 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
plot([min(targets) max(targets)], [min(targets) max(targets)], 'r--', 'LineWidth', 2);
grid on;
xlabel('Actual Vmpp (V)', 'FontSize', 11);
ylabel('Predicted Vmpp (V)', 'FontSize', 11);
title('Test Set: Predicted vs Actual', 'FontSize', 12, 'FontWeight', 'bold');
axis equal tight;

% Plot 2: Error distribution
subplot(2,3,2);
errors = outputs(testInd) - targets(testInd);
histogram(errors, 30, 'FaceColor', [0.2 0.6 0.8]);
xlabel('Prediction Error (V)', 'FontSize', 11);
ylabel('Frequency', 'FontSize', 11);
title('Test Set: Error Distribution', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
xline(0, 'r--', 'LineWidth', 2);

% Plot 3: Vmpp vs Irradiance (colored by T)
subplot(2,3,3);
scatter(inputs(1,testInd), targets(testInd), 50, inputs(2,testInd), 'filled', 'MarkerEdgeColor', 'k');
hold on;
scatter(inputs(1,testInd), outputs(testInd), 50, inputs(2,testInd), 'o', 'LineWidth', 1.5);
colorbar;
xlabel('Irradiance (W/m²)', 'FontSize', 11);
ylabel('Vmpp (V)', 'FontSize', 11);
title('Vmpp vs Irradiance (colored by T)', 'FontSize', 12, 'FontWeight', 'bold');
legend('Actual', 'Predicted', 'Location', 'best');
grid on;

% Plot 4: Vmpp vs Temperature (colored by G)
subplot(2,3,4);
scatter(inputs(2,testInd), targets(testInd), 50, inputs(1,testInd), 'filled', 'MarkerEdgeColor', 'k');
hold on;
scatter(inputs(2,testInd), outputs(testInd), 50, inputs(1,testInd), 'o', 'LineWidth', 1.5);
colorbar;
xlabel('Temperature (°C)', 'FontSize', 11);
ylabel('Vmpp (V)', 'FontSize', 11);
title('Vmpp vs Temperature (colored by G)', 'FontSize', 12, 'FontWeight', 'bold');
legend('Actual', 'Predicted', 'Location', 'best');
grid on;

% Plot 5: Residual plot
subplot(2,3,5);
residuals = outputs(testInd) - targets(testInd);
scatter(targets(testInd), residuals, 40, 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
yline(0, 'r--', 'LineWidth', 2);
yline(mean(residuals) + 2*std(residuals), 'b--', 'LineWidth', 1);
yline(mean(residuals) - 2*std(residuals), 'b--', 'LineWidth', 1);
grid on;
xlabel('Actual Vmpp (V)', 'FontSize', 11);
ylabel('Residual (V)', 'FontSize', 11);
title('Residual Plot (Test Set)', 'FontSize', 12, 'FontWeight', 'bold');

% Plot 6: Error vs Irradiance
subplot(2,3,6);
scatter(inputs(1,testInd), abs(errors), 40, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('Irradiance (W/m²)', 'FontSize', 11);
ylabel('Absolute Error (V)', 'FontSize', 11);
title('Error vs Irradiance', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

%% STEP 12: Save model
save('ANN_MPPT_Corrected.mat', 'net', 'ps_input', 'ps_target');
fprintf('\n✓ Model saved: ANN_MPPT_Corrected.mat\n');

%% STEP 13: Interactive testing
fprintf('\n=== INTERACTIVE TESTING ===\n');
fprintf('Valid ranges:\n');
fprintf('  Irradiance: %.0f - %.0f W/m²\n', min(inputs(1,:)), max(inputs(1,:)));
fprintf('  Temperature: %.1f - %.1f °C\n', min(inputs(2,:)), max(inputs(2,:)));
fprintf('  Expected Vmpp: %.1f - %.1f V\n\n', min(targets), max(targets));

while true
    G = input('Enter Irradiance (W/m²) [0 to quit]: ');
    if G == 0, break; end
    
    T = input('Enter Temperature (°C): ');
    
    % Check if input is within training range
    if G < min(inputs(1,:)) || G > max(inputs(1,:))
        fprintf('⚠ Warning: Irradiance outside training range!\n');
    end
    if T < min(inputs(2,:)) || T > max(inputs(2,:))
        fprintf('⚠ Warning: Temperature outside training range!\n');
    end
    
    % Normalize and predict
    test_input = [G; T];
    test_input_norm = mapminmax('apply', test_input, ps_input);
    pred_norm = net(test_input_norm);
    pred_vmpp = mapminmax('reverse', pred_norm, ps_target);
    
    fprintf('→ Predicted Vmpp = %.2f V\n\n', pred_vmpp);
end

%% STEP 14: Export instructions
fprintf('\n=== SIMULINK EXPORT ===\n');
fprintf('To export to Simulink:\n');
fprintf('  >> gensim(net, ''ANN_MPPT_Block'')\n\n');

%% STEP 15: Data quality warning
if min(targets) > 30
    fprintf('⚠⚠⚠ DATA QUALITY WARNING ⚠⚠⚠\n');
    fprintf('Your dataset only contains voltages above %.1fV.\n', min(targets));
    fprintf('This suggests:\n');
    fprintf('  1. Dataset may only contain near-Voc measurements\n');
    fprintf('  2. Missing low-voltage, high-current operating points\n');
    fprintf('  3. The model will NOT work well for voltages < %.1fV\n\n', min(targets));
    fprintf('For complete MPPT control, you need I-V sweep data from:\n');
    fprintf('  - Short circuit (V≈0, I=Isc) to\n');
    fprintf('  - Open circuit (V=Voc, I≈0)\n');
    fprintf('Consider collecting full I-V curve data across all operating points.\n');
end

fprintf('\n=== TRAINING COMPLETE ===\n');