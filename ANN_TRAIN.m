% ANN_TRAIN.m
% Step 1: Load Clean Dataset
data = readtable('clean_dataset.csv','VariableNamingRule','preserve');

% Step 2: Build Inputs (X) and Targets (T)
X = table2array(data(:, {'Temperature','Irradiance','Voltage','Current'}));
T = table2array(data(:, {'Power'}));

% Step 3: Transpose (so ANN understands correctly)
X = X';   % [4 x N]
T = T';   % [1 x N]

% Step 4: Define ANN
net = feedforwardnet(10);  % 10 hidden neurons (tunable)

% Step 5: Configure Data Division (70% train, 15% val, 15% test)
net.divideParam.trainRatio = 0.70;
net.divideParam.valRatio   = 0.15;
net.divideParam.testRatio  = 0.15;

% Step 6: Train ANN
[net,tr] = train(net, X, T);

% Step 7: Test ANN
Y = net(X);   % Predictions on all data
perf = perform(net, T, Y);  % Mean Squared Error on all

% Step 8: Extract Train/Val/Test performances
trainTargets = T .* tr.trainMask{1};
valTargets   = T .* tr.valMask{1};
testTargets  = T .* tr.testMask{1};

fprintf('MSE (Train): %f\n', perform(net, trainTargets, Y));
fprintf('MSE (Val):   %f\n', perform(net, valTargets, Y));
fprintf('MSE (Test):  %f\n', perform(net, testTargets, Y));

% Step 9: Visualize Results
figure;
plot(T, 'b'); hold on;
plot(Y, 'r');
legend('Target (Actual Power)','ANN Output (Predicted Power)');
title('ANN Training Results (70-15-15 Split)');
xlabel('Sample');
ylabel('Power (W)');
grid on;
