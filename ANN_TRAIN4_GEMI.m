%% === ROBUST ANN MPPT MODEL: (G,T → Vmpp/Voltage) ===
% This corrected script addresses the core issues of input selection and
% model robustness for out-of-distribution predictions.
% VERSION 2: Now automatically detects if the target column is 'Vmpp' or 'Voltage'.

clc; 
clear; 
close all;

%% STEP 1: Load and filter dataset
data = readtable('clean_dataset.csv');
fprintf('Original dataset: %d rows\n', height(data));

% Keep strong daylight samples only (G > 100 avoids low-light noise)
data = data(data.Irradiance > 100 & data.Voltage > 0.1, :);
fprintf('Filtered dataset: %d rows (Irradiance > 100)\n', height(data));

%% STEP 2: Define Target Variable Name (ROBUST FIX)
% Check if a 'Vmpp' column exists, otherwise default to 'Voltage'.
if any(strcmp('Vmpp', data.Properties.VariableNames))
    targetVar = 'Vmpp';
else
    targetVar = 'Voltage';
end
fprintf('Using ''%s'' as the target variable for training.\n', targetVar);


%% STEP 3: Data Exploration (CRUCIAL NEW STEP)
% Before training, let's understand the data's limits and distribution.
figure;
% FIXED: Use the dynamically found target variable name
plotmatrix([data.Irradiance, data.Temperature, data.(targetVar)]);
title('Data Distribution and Correlations');
sgtitle('Data Exploration: Check for Gaps and Ranges');

% Store min/max ranges for later use in manual testing
min_G = min(data.Irradiance);
max_G = max(data.Irradiance);
min_T = min(data.Temperature);
max_T = max(data.Temperature);

fprintf('\n--- Data Ranges ---\n');
fprintf('Irradiance (G): [%.1f, %.1f] W/m²\n', min_G, max_G);
fprintf('Temperature (T): [%.1f, %.1f] °C\n', min_T, max_T);
fprintf('-------------------\n\n');


%% STEP 4: Define Inputs and Targets (CORRECTED & ROBUST)
inputs = [data.Irradiance.'; data.Temperature.']; % 2xN matrix
% FIXED: Use the dynamically found target variable name
targets = data.(targetVar).'; % 1xN matrix
N = size(inputs, 2);


%% STEP 5: Shuffle and Split Dataset
rng(42); % Set seed for reproducibility
idx = randperm(N);
inputs = inputs(:, idx);
targets = targets(:, idx);

% Split data into training (70%), validation (15%), and testing (15%)
[trainInd, valInd, testInd] = dividerand(N, 0.7, 0.15, 0.15);

% Create training-only subsets for normalization fitting
inputs_train = inputs(:, trainInd);
targets_train = targets(:, trainInd);


%% STEP 6: Normalize Data using mapminmax
[inp_train_norm, ps_input] = mapminmax(inputs_train);
[tgt_train_norm, ps_target] = mapminmax(targets_train);

% Apply the settings from the TRAINING data to the ENTIRE dataset
inp_norm = mapminmax('apply', inputs, ps_input);
tgt_norm = mapminmax('apply', targets, ps_target);


%% STEP 7: Create and Configure the ANN
hiddenLayerSize = [10 7]; 
net = fitnet(hiddenLayerSize, 'trainbr');

net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = testInd;

net.trainParam.epochs = 1000;
net.trainParam.goal = 1e-6;
net.trainParam.max_fail = 20;


%% STEP 8: Train the Network
fprintf('Training Robust ANN (G,T → %s)...\n', targetVar);
[net, tr] = train(net, inp_norm, tgt_norm);


%% STEP 9: Evaluate Model Performance
outputs_norm = net(inp_norm);
outputs = mapminmax('reverse', outputs_norm, ps_target);
targets_true = targets; 

rmse = sqrt(mean((outputs - targets_true).^2));
mae = mean(abs(outputs - targets_true));
r2 = 1 - sum((targets_true - outputs).^2) / sum((targets_true - mean(targets_true)).^2);
mape = mean(abs((outputs - targets_true) ./ targets_true)) * 100;

fprintf('\nPerformance Metrics (Robust G,T → %s):\n', targetVar);
fprintf('RMSE = %.3f V | MAE = %.3f V | R² = %.4f | MAPE = %.2f%%\n', ...
         rmse, mae, r2, mape);
fprintf('Train MSE = %.6f | Val MSE = %.6f | Test MSE = %.6f\n', ...
         tr.best_perf, tr.best_vperf, tr.best_tperf);


%% STEP 10: Save the Trained Model
save('ANN_MPPT_GT_Robust.mat', 'net', 'ps_input', 'ps_target', 'min_G', 'max_G', 'min_T', 'max_T', 'targetVar');
fprintf('\nModel saved as ANN_MPPT_GT_Robust.mat\n\n');


%% STEP 11: Visualize Prediction Quality
figure;
scatter(targets_true, outputs, 10, 'filled'); 
hold on;
plot([min(targets_true), max(targets_true)], [min(targets_true), max(targets_true)], 'r-', 'LineWidth', 1.5);
hold off;
grid on;
xlabel(['Actual ' targetVar ' (V)']); % Dynamic axis label
ylabel(['Predicted ' targetVar ' (V)']); % Dynamic axis label
title('ANN Prediction Performance (Actual vs. Predicted)');
legend('Predictions', 'Ideal y=x Line');


%% STEP 12: Interactive Manual Test Loop (IMPROVED)
fprintf('-------------------------------------------------------------\n');
fprintf('Entering interactive test mode. Press Ctrl+C to exit.\n');
fprintf('NOTE: Model is most accurate for inputs within training range:\n');
fprintf('G: [%.1f, %.1f], T: [%.1f, %.1f]\n', min_G, max_G, min_T, max_T);
fprintf('-------------------------------------------------------------\n');

while true
    G_test = input('Enter Irradiance (W/m²): ');
    T_test = input('Enter Temperature (°C): ');

    if G_test < min_G || G_test > max_G || T_test < min_T || T_test > max_T
        fprintf('*** WARNING: Input is outside the training data range. Prediction may be inaccurate. ***\n');
    end
    
    test_input = [G_test; T_test];
    test_input_norm = mapminmax('apply', test_input, ps_input);
    pred_norm = net(test_input_norm);
    pred_val = mapminmax('reverse', pred_norm, ps_target);
    
    fprintf('--> Predicted %s for G=%.1f W/m², T=%.1f °C → %.3f V\n\n', ...
             targetVar, G_test, T_test, pred_val);
end