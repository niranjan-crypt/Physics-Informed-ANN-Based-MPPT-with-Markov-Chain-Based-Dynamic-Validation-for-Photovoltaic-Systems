clc; clear; close all;

%% STEP 1: Load and Analyze Raw Dataset
data = readtable('clean_dataset.csv');
fprintf('=== DATASET ANALYSIS ===\n');
fprintf('Total rows: %d\n', height(data));
fprintf('Voltage range: %.2f to %.2f V\n', min(data.Voltage), max(data.Voltage));
fprintf('Irradiance range: %.1f to %.1f W/m²\n', min(data.Irradiance), max(data.Irradiance));
fprintf('Temperature range: %.1f to %.1f °C\n\n', min(data.Temperature), max(data.Temperature));

%% STEP 2: Visualize Raw Data Distribution
figure('Position', [50 50 1400 400]);
subplot(1,3,1);
histogram(data.Voltage, 50);
title('Raw Voltage Distribution'); xlabel('Voltage (V)'); ylabel('Frequency'); grid on;
subplot(1,3,2);
histogram(data.Current, 50);
title('Raw Current Distribution'); xlabel('Current (A)'); ylabel('Frequency'); grid on;
subplot(1,3,3);
data.Power = data.Voltage .* data.Current;
scatter(data.Voltage, data.Current, 10, data.Power, 'filled');
title('I-V Curve (colored by Power)'); xlabel('Voltage (V)'); ylabel('Current (A)');
colorbar; grid on;

%% STEP 3: Filter Dataset for Relevant Operating Conditions
data = data(data.Irradiance > 100 & data.Voltage > 1, :);
data.Power = data.Voltage .* data.Current;
fprintf('After filtering (G>100, V>1): %d rows\n\n', height(data));

%% STEP 4: Extract a High-Quality MPP Training Dataset
% Method 1: Find the direct voltage at Maximum Power Point for each G/T condition.
G_bins = discretize(data.Irradiance, 50);
T_bins = discretize(data.Temperature, 20);
[groups, ~] = findgroups(G_bins, T_bins);
vmpp_method1 = splitapply(@(v,p) v(p==max(p)), data.Voltage, data.Power, groups);
G_mpp1 = splitapply(@mean, data.Irradiance, groups);
T_mpp1 = splitapply(@mean, data.Temperature, groups);
P_max1 = splitapply(@max, data.Power, groups);
valid1 = ~isnan(vmpp_method1) & ~isnan(G_mpp1) & ~isnan(T_mpp1) & P_max1 > 1;
vmpp_method1 = vmpp_method1(valid1);
G_mpp1 = G_mpp1(valid1);
T_mpp1 = T_mpp1(valid1);
fprintf('Method 1 (Direct MPP): %d points extracted\n', length(vmpp_method1));

% Method 2: Sample additional points from the likely MPP region for better data coverage.
v_max = max(data.Voltage);
mpp_region_indices = data.Voltage >= 0.60*v_max & data.Voltage <= 0.95*v_max;
data_mpp_region = data(mpp_region_indices, :);

if height(data_mpp_region) > 200
    vmpp_method2 = data_mpp_region.Voltage;
    G_mpp2 = data_mpp_region.Irradiance;
    T_mpp2 = data_mpp_region.Temperature;
    
    if length(vmpp_method2) > 2000 % Downsample if region is too dense
        idx_sample = randsample(length(vmpp_method2), 2000);
        vmpp_method2 = vmpp_method2(idx_sample);
        G_mpp2 = G_mpp2(idx_sample);
        T_mpp2 = T_mpp2(idx_sample);
    end
    fprintf('Method 2 (MPP Region): %d points extracted\n', length(vmpp_method2));
    
    % Combine both methods to create the final dataset
    vmpp_combined = [vmpp_method1; vmpp_method2];
    G_combined = [G_mpp1; G_mpp2];
    T_combined = [T_mpp1; T_mpp2];
    fprintf('Combined dataset: %d points\n', length(vmpp_combined));
else
    vmpp_combined = vmpp_method1;
    G_combined = G_mpp1;
    T_combined = T_mpp1;
    fprintf('Using Method 1 only (insufficient MPP region data)\n');
end

%% STEP 5: Prepare Final Dataset for Training
inputs = [G_combined'; T_combined'];
targets = vmpp_combined';
N = size(inputs, 2);
fprintf('\n=== FINAL TRAINING DATASET ===\n');
fprintf('Total samples: %d\n', N);
fprintf('Vmpp range: %.2f to %.2f V\n\n', min(targets), max(targets));

%% STEP 6: Shuffle and Split Data
rng(42);
idx = randperm(N);
inputs = inputs(:, idx);
targets = targets(:, idx);
[trainInd, valInd, testInd] = dividerand(N, 0.7, 0.15, 0.15);
fprintf('Data split: %d train, %d val, %d test\n', length(trainInd), length(valInd), length(testInd));

%% STEP 7: Normalize Data
[inp_norm, ps_input] = mapminmax(inputs(:, trainInd));
[tgt_norm, ps_target] = mapminmax(targets(:, trainInd));
inputs_norm = mapminmax('apply', inputs, ps_input);
targets_norm = mapminmax('apply', targets, ps_target);

%% STEP 8: Create and Configure the Neural Network
hiddenLayerSize = [20, 15];
net = fitnet(hiddenLayerSize, 'trainlm');
net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = testInd;
net.trainParam.epochs = 1500;
net.trainParam.goal = 1e-6;
net.trainParam.max_fail = 25;
net.trainParam.showWindow = true;

%% STEP 9: Train the Network
fprintf('\n=== TRAINING ANN: (G,T) → Vmpp ===\n');
[net, tr] = train(net, inputs_norm, targets_norm);

%% STEP 10: Evaluate Model Performance
outputs_norm = net(inputs_norm);
outputs = mapminmax('reverse', outputs_norm, ps_target);
sets = {trainInd, valInd, testInd};
set_names = {'Training', 'Validation', 'Test'};
fprintf('\n=== PERFORMANCE METRICS ===\n');
for i = 1:3
    y_true = targets(sets{i});
    y_pred = outputs(sets{i});
    rmse = sqrt(mean((y_true - y_pred).^2));
    mae = mean(abs(y_true - y_pred));
    r2 = 1 - sum((y_true - y_pred).^2) / sum((y_true - mean(y_true)).^2);
    fprintf('%s: RMSE=%.3fV | MAE=%.3fV | R²=%.4f\n', set_names{i}, rmse, mae, r2);
end

%% STEP 11: Essential Visualization (Simplified)
figure('Position', [100 100 1500 500]);
sgtitle('Essential Performance Plots (Test Set)', 'FontSize', 14, 'FontWeight', 'bold');

% Plot 1: Predicted vs Actual 
subplot(1,3,1);
scatter(targets(testInd), outputs(testInd), 30, 'filled', 'MarkerFaceAlpha', 0.7);
hold on;
plot([min(targets) max(targets)], [min(targets) max(targets)], 'r--', 'LineWidth', 2);
grid on;
xlabel('Actual Vmpp (V)');
ylabel('Predicted Vmpp (V)');
title('Predicted vs. Actual');
axis equal tight;

% Plot 2: Error Distribution
subplot(1,3,2);
errors = outputs(testInd) - targets(testInd);
histogram(errors, 30);
grid on;
xlabel('Prediction Error (V)');
ylabel('Frequency');
title('Error Distribution');
xline(0, 'r--', 'LineWidth', 2);

% Plot 3: Residual Plot
subplot(1,3,3);
scatter(outputs(testInd), errors, 30, 'filled', 'MarkerFaceAlpha', 0.7);
hold on;
yline(0, 'r--', 'LineWidth', 2);
grid on;
xlabel('Predicted Vmpp (V)');
ylabel('Residual Error (V)');
title('Residuals vs. Predicted Values');

%% STEP 12: Save the Trained Model
save('ANN_MPPT_Corrected.mat', 'net', 'ps_input', 'ps_target');
fprintf('\n✓ Model saved: ANN_MPPT_Corrected.mat\n');

%% STEP 13: Interactive Testing Loop
fprintf('\n=== INTERACTIVE TESTING ===\n');
while true
    G = input('Enter Irradiance (W/m²) [0 to quit]: ');
    if G == 0, break; end
    T = input('Enter Temperature (°C): ');
    
    test_input = [G; T];
    test_input_norm = mapminmax('apply', test_input, ps_input);
    pred_norm = net(test_input_norm);
    pred_vmpp = mapminmax('reverse', pred_norm, ps_target);
    
    fprintf('→ Predicted Vmpp = %.2f V\n\n', pred_vmpp);
end

fprintf('\n=== TRAINING COMPLETE ===\n');