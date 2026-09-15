%% === Step 2: Train ANN for MPPT ===
clc;

% Load the 28k daytime dataset
data = readtable('daytime_dataset.csv');
fprintf('Loaded dataset: %d rows\n', height(data));

% Define Inputs and Targets
inputs  = [data.Irradiance.'; data.Temperature.'; data.Voltage.'; data.Current.'];
targets = data.Voltage.';   % Using Voltage as proxy for Vmpp (approximation)

% Normalize inputs and targets
[inp_norm, ps_input] = mapminmax(inputs);
[tgt_norm, ps_target] = mapminmax(targets);

% Split data into Train/Val/Test (70/15/15)
N = size(inputs,2);
[trainInd,valInd,testInd] = dividerand(N,0.7,0.15,0.15);

% Define Network Architecture
hiddenLayerSize = [10 8];
net = fitnet(hiddenLayerSize,'trainlm');
net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.valInd   = valInd;
net.divideParam.testInd  = testInd;
net.trainParam.epochs    = 1000;
net.trainParam.goal      = 1e-6;

% Train the Network
fprintf('Training ANN on 28k daytime dataset...\n');
[net,tr] = train(net, inp_norm, tgt_norm);

% Predict on test set
outputs_norm = net(inp_norm);
outputs = mapminmax('reverse', outputs_norm, ps_target);

% Compute Error Metrics
targets_true = targets;
rmse = sqrt(mean((outputs - targets_true).^2));
mae  = mean(abs(outputs - targets_true));
r2   = 1 - sum((targets_true - outputs).^2)/sum((targets_true - mean(targets_true)).^2);
mape = mean(abs((outputs - targets_true)./targets_true))*100;
eff  = (mean(outputs./targets_true))*100;

fprintf('\nFinal RMSE = %.4f V\n', rmse);
fprintf('MAE = %.4f V | R² = %.4f | MAPE = %.2f%% | Tracking Efficiency = %.2f%%\n',...
         mae, r2, mape, eff);

% Save trained model
save('ANN_MPPT_Model_28k.mat','net','ps_input','ps_target');
fprintf('Model saved as ANN_MPPT_Model_28k.mat\n');

%% === Step 3: Test with manual input ===
prompt = 'Enter Irradiance (W/m²): '; G = input(prompt);
prompt = 'Enter Temperature (°C): ';  T = input(prompt);
prompt = 'Enter PV Voltage (V): ';    V = input(prompt);
prompt = 'Enter PV Current (A): ';    I = input(prompt);

test_input = [G; T; V; I];
test_input_norm = mapminmax('apply', test_input, ps_input);
pred_norm = net(test_input_norm);
pred_vmpp = mapminmax('reverse', pred_norm, ps_target);

fprintf('\nPredicted Vmpp for given conditions: %.3f V\n', pred_vmpp);
fprintf('-------------------------------------------------------------\n');
fprintf('Next Step: Connect ANN block in Simulink using >> gensim(net)\n');
fprintf('-------------------------------------------------------------\n');
