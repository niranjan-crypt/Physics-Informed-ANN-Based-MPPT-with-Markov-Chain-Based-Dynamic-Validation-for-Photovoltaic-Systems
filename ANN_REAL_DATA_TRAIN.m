%% ========================================================================
%  ANN-based MPPT Model (Dynamic: G, T, Vpv, Ipv -> Vmpp)
%  Using Real Dataset: clean_dataset.csv
%  -------------------------------------------------------------
%  Author  : Siddhu Team (MPPT Project)
%  Objective: Train ANN using real PV data instead of synthetic data
%  Dataset Columns: Temperature, Irradiance, Voltage, Current, Power
%  Inputs  : [Irradiance, Temperature, Voltage, Current]
%  Target  : Voltage (assumed Vmpp)
% ========================================================================

clc; clear; close all

%% 1. Load Real Dataset
% Ensure 'clean_dataset.csv' is in the current MATLAB folder.
% The CSV must contain headers: Temperature, Irradiance, Voltage, Current, Power
data = readtable('clean_dataset.csv');

% Display first few rows to verify
disp('First 5 rows of dataset:');
disp(data(1:5,:));

%% 2. Extract Required Columns
T  = data.Temperature;   % Temperature (°C)
G  = data.Irradiance;    % Solar Irradiance (W/m²)
Vpv = data.Voltage;      % PV Voltage (V)
Ipv = data.Current;      % PV Current (A)

% Target (Vmpp) → assume Voltage represents Vmpp from clean dataset
Vmpp = Vpv;  

%% 3. Prepare Input and Target Matrices
inputs  = [G.'; T.'; Vpv.'; Ipv.'];  % 4 × N matrix
targets = Vmpp.';                    % 1 × N vector

% Check data dimensions
fprintf('\nDataset size: %d samples\n', length(Vmpp));

%% 4. Normalize Data (same as before)
[inputs, ps_input] = mapminmax(inputs, 0, 1);
[targets, ps_output] = mapminmax(targets, 0, 1);

%% 5. Split Dataset (same ratios)
N = length(Vmpp);
trainRatio = 0.7; valRatio = 0.15; testRatio = 0.15;
[trainInd,valInd,testInd] = dividerand(N, trainRatio, valRatio, testRatio);

%% 6. Create & Configure ANN (same architecture)
hiddenLayerSize = [10 8];                 % 2 hidden layers
net = feedforwardnet(hiddenLayerSize, 'trainlm');  % Levenberg–Marquardt

net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.valInd   = valInd;
net.divideParam.testInd  = testInd;

net.trainParam.epochs = 500;
net.trainParam.goal   = 1e-5;
net.trainParam.showWindow = true;

%% 7. Train the Network
fprintf('\nTraining the ANN on real dataset...\n');
[net,tr] = train(net,inputs,targets);

%% 8. Test & Evaluate Model
outputs = net(inputs);

% Un-normalize
targets_un = mapminmax('reverse', targets, ps_output);
outputs_un = mapminmax('reverse', outputs, ps_output);

% RMSE Calculation
mse_error = mse(outputs_un - targets_un);
rmse = sqrt(mse_error);
fprintf('\nFinal RMSE on full dataset = %.4f V\n', rmse);

%% 9. Plot Results
figure;
plot(targets_un, 'b'); hold on;
plot(outputs_un, 'r--');
legend('Actual Vmpp','Predicted Vmpp');
xlabel('Sample Index');
ylabel('Voltage (V)');
title('ANN Prediction vs Actual Vmpp (Real Dataset)');
grid on;

%% 10. Save Trained Model
save('ANN_MPPT_RealData.mat','net','ps_input','ps_output');

fprintf('\nTrained network saved successfully as ANN_MPPT_RealData.mat\n');
fprintf('Use >> gensim(net) to generate a Simulink block.\n');
%% ========================================================================
