%% ========================================================================
%  ANN-based MPPT Model (Dynamic: G, T, Vpv, Ipv -> Vmpp)
%  Author : Siddhu Team (MPPT Project)
%  ------------------------------------------------------------------------
%  Goal: Train an ANN to learn Vmpp from dynamic PV data
%  Inputs : Irradiance (G), Temperature (T), PV Voltage (Vpv), PV Current (Ipv)
%  Output : Vmpp (Voltage at Maximum Power Point)
% ========================================================================

clc; clear; close all

%% 1. Load or Generate Dataset
% If you already have a dataset, load it here.
% Example columns: [G  T  Vpv  Ipv  Vmpp]
% For demo: we'll generate a synthetic but realistic dataset

N = 45000;  % number of samples
G = 200 + 800*rand(1,N);        % Irradiance: 200–1000 W/m²
T = 20 + 25*rand(1,N);          % Temperature: 20–45 °C
Vpv = 10 + 20*rand(1,N);        % PV voltage: 10–30 V
Ipv = (G/1000).*rand(1,N)*8;    % PV current proportional to irradiance

% Vmpp roughly 0.8 * Voc (linear approx) with temperature effect
Voc = 0.6*T/25 + 0.04*G/1000 + 30; 
Vmpp = 0.8 * Voc - 0.003*(T-25);  % Vmpp decreases slightly with T

% Add small dynamic noise (simulate clouds, fluctuation)
Vmpp = Vmpp + 0.5*randn(1,N);

%% 2. Prepare Input-Output Matrices
inputs  = [G; T; Vpv; Ipv];
targets = Vmpp;

% Normalize data for better ANN training
[inputs, ps_input] = mapminmax(inputs, 0, 1);
[targets, ps_output] = mapminmax(targets, 0, 1);

%% 3. Split Dataset
trainRatio = 0.7; valRatio = 0.15; testRatio = 0.15;
[trainInd,valInd,testInd] = dividerand(N, trainRatio, valRatio, testRatio);

%% 4. Create ANN
hiddenLayerSize = [10 8];   % two hidden layers
net = feedforwardnet(hiddenLayerSize, 'trainlm');  % Levenberg–Marquardt training

% Setup data division
net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.valInd   = valInd;
net.divideParam.testInd  = testInd;

% Training parameters
net.trainParam.epochs = 500;
net.trainParam.goal   = 1e-5;
net.trainParam.showWindow = true;

%% 5. Train ANN
[net,tr] = train(net,inputs,targets);

%% 6. Test ANN
outputs = net(inputs);
targets_un = mapminmax('reverse', targets, ps_output);
outputs_un = mapminmax('reverse', outputs, ps_output);

% Calculate error metrics
mse_error = mse(outputs_un - targets_un);
rmse = sqrt(mse_error);
fprintf('\nRMSE = %.4f V\n', rmse);

%% 7. Plot Results
figure;
plot(targets_un, 'b'); hold on;
plot(outputs_un, 'r--');
legend('Actual Vmpp','Predicted Vmpp');
xlabel('Sample'); ylabel('Voltage (V)');
title('ANN Prediction vs Actual Vmpp');
grid on;

%% 8. Save Trained Network
save('ANN_MPPT_Model.mat','net','ps_input','ps_output');

%% ========================================================================
%  Usage in Simulink:
%  You can import this trained network using:
%  >> gensim(net)
%  This creates a Simulink block automatically.
% ========================================================================
