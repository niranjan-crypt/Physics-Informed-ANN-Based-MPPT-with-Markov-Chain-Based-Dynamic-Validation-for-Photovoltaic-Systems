clc; clear; close all;

%% STEP 1: Load Dataset
data = readtable('clean_dataset.csv');
fprintf('=== DATASET ANALYSIS ===\n');
fprintf('Rows: %d | V: %.2f–%.2f | I: %.2f–%.2f | G: %.1f–%.1f | T: %.1f–%.1f\n\n', ...
    height(data), min(data.Voltage), max(data.Voltage), ...
    min(data.Current), max(data.Current), ...
    min(data.Irradiance), max(data.Irradiance), ...
    min(data.Temperature), max(data.Temperature));

%% STEP 2: Filter Valid Data
data = data(data.Irradiance > 100 & data.Voltage > 1, :);
data.Power = data.Voltage .* data.Current;

%% STEP 3: Extract MPP Points
G_bins = discretize(data.Irradiance, 50);
T_bins = discretize(data.Temperature, 20);
[groups, ~] = findgroups(G_bins, T_bins);

Vmpp = splitapply(@(v,p) v(p==max(p)), data.Voltage, data.Power, groups);
Impp = splitapply(@(i,p) i(p==max(p)), data.Current, data.Power, groups);
Pmpp = splitapply(@max, data.Power, groups);
Gmpp = splitapply(@mean, data.Irradiance, groups);
Tmpp = splitapply(@mean, data.Temperature, groups);

valid = ~isnan(Vmpp);
Vmpp = Vmpp(valid); Impp = Impp(valid); Pmpp = Pmpp(valid);
Gmpp = Gmpp(valid); Tmpp = Tmpp(valid);

fprintf('Valid MPP samples: %d\n', numel(Vmpp));

%% STEP 4: Prepare ANN Data
inputs = [Gmpp'; Tmpp'];
targets = Vmpp';
N = size(inputs, 2);

rng(42);
[trainInd, valInd, testInd] = dividerand(N, 0.7, 0.15, 0.15);
[inp_norm, ps_input] = mapminmax(inputs(:, trainInd));
[tgt_norm, ps_target] = mapminmax(targets(:, trainInd));
inputs_norm = mapminmax('apply', inputs, ps_input);
targets_norm = mapminmax('apply', targets, ps_target);

%% STEP 5: ANN Training (unchanged)
hiddenLayerSize = [20, 15];
net = fitnet(hiddenLayerSize, 'trainlm');
net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = testInd;
net.trainParam.epochs = 1500;
net.trainParam.goal = 1e-6;
net.trainParam.max_fail = 25;

fprintf('\n=== TRAINING (G,T→Vmpp) ===\n');
[net, tr] = train(net, inputs_norm, targets_norm);

%% STEP 6: Performance
outputs_norm = net(inputs_norm);
outputs = mapminmax('reverse', outputs_norm, ps_target);
sets = {trainInd, valInd, testInd};
names = {'Train', 'Val', 'Test'};

fprintf('\n=== PERFORMANCE ===\n');
for i = 1:3
    y_true = targets(sets{i});
    y_pred = outputs(sets{i});
    rmse = sqrt(mean((y_true - y_pred).^2));
    mae = mean(abs(y_true - y_pred));
    r2 = 1 - sum((y_true - y_pred).^2)/sum((y_true - mean(y_true)).^2);
    rel_error = mean(abs(y_true - y_pred)./y_true) * 100;
    fprintf('%s: RMSE=%.3fV | MAE=%.3fV | R²=%.4f | Rel Err=%.1f%%\n', ...
        names{i}, rmse, mae, r2, rel_error);
end

fprintf('Training Vmpp targets: min=%.2f, mean=%.2f, max=%.2f\n', min(targets), mean(targets), max(targets));
histogram(targets, 30); title('Vmpp targets distribution'); xlabel('Vmpp (V)');


save('ANN_MPPT_Improved.mat', 'net', 'ps_input', 'ps_target');
fprintf('\nModel saved.\n');
%% OVERFITTING ANALYSIS
fprintf('\n=== OVERFITTING ANALYSIS ===\n');

% Check performance gaps
train_perf = sqrt(tr.best_perf);
val_perf = sqrt(tr.best_vperf);
test_perf = sqrt(tr.best_tperf);

fprintf('Final Errors - Train: %.3fV, Val: %.3fV, Test: %.3fV\n', ...
        train_perf, val_perf, test_perf);

overfitting_gap = (val_perf - train_perf) / train_perf * 100;
fprintf('Overfitting gap: %.1f%% (should be <15%%)\n', overfitting_gap);

% Check if validation error started increasing
if any(diff(tr.vperf) > 0)
    fprintf('⚠️  Validation error increased during training - potential overfitting\n');
else
    fprintf('✅ Validation error consistently decreased - good generalization\n');
end

% Rule of thumb: if val_error > 1.5× train_error → overfitting
if val_perf > 1.5 * train_perf
    fprintf('🚨 SIGNIFICANT OVERFITTING DETECTED\n');
elseif val_perf > 1.2 * train_perf  
    fprintf('⚠️  Mild overfitting detected\n');
else
    fprintf('✅ No significant overfitting detected\n');
end
%% INTERACTIVE TESTING (use real dataset curves for "true" MPP)
fprintf('\n=== INTERACTIVE TESTING (REAL DATA MPP COMPARE) ===\n');
tolerance_G = 20;   % W/m^2 tolerance to find nearby real points
tolerance_T = 2;    % degC tolerance

while true
    G_input = input('Enter Irradiance (W/m²) [0 to quit]: ');
    if G_input == 0, break; end
    T_input = input('Enter Temperature (°C): ');
    
    % ANN predict Vmpp
    test_input = [G_input; T_input];
    test_input_norm = mapminmax('apply', test_input, ps_input);
    pred_norm = net(test_input_norm);
    Vmpp_pred = mapminmax('reverse', pred_norm, ps_target);
    
    % Find measured rows close to the requested condition
    idx_close = abs(data.Irradiance - G_input) <= tolerance_G & ...
                abs(data.Temperature - T_input) <= tolerance_T;
    if sum(idx_close) < 3
        % If too few points found, expand tolerances gradually
        idx_close = abs(data.Irradiance - G_input) <= (2*tolerance_G) & ...
                    abs(data.Temperature - T_input) <= (2*tolerance_T);
    end
    
    if sum(idx_close) < 3
        fprintf('Not enough measured (V,I) points near G=%.1f, T=%.1f. Try different values.\n\n', G_input, T_input);
        continue;
    end
    
    % Extract measured V,I and compute power
    V_meas = data.Voltage(idx_close);
    I_meas = data.Current(idx_close);
    % Sort by voltage (important for plotting & interp)
    [V_meas, sortIdx] = sort(V_meas);
    I_meas = I_meas(sortIdx);
    P_meas = V_meas .* I_meas;
    
    % True measured MPP from dataset points (closest sample)
    [Pmax_meas, iP] = max(P_meas);
    Vmpp_meas = V_meas(iP);
    Impp_meas = I_meas(iP);
    
    % Interpolate measured curve to get Impp at Vmpp_pred (if within measured V range)
    Impp_at_pred = interp1(V_meas, I_meas, Vmpp_pred, 'linear', NaN);
    P_at_pred = Vmpp_pred * Impp_at_pred;
    if isnan(Impp_at_pred)
        % Predicted Vmpp outside measured voltage range -> extrapolate nearest value
        if Vmpp_pred < min(V_meas)
            Impp_at_pred = I_meas(1);
        else
            Impp_at_pred = I_meas(end);
        end
        P_at_pred = Vmpp_pred * Impp_at_pred;
    end
    
    % Print results
    fprintf('\n=== RESULTS ===\n');
    fprintf('Requested: G=%.1f W/m², T=%.1f°C\n', G_input, T_input);
    fprintf('ANN Predicted Vmpp = %.2f V | Impp_at_pred = %.2f A | P_at_pred = %.2f W\n', Vmpp_pred, Impp_at_pred, P_at_pred);
    fprintf('Measured Vmpp     = %.2f V | Impp_meas = %.2f A | Pmeas = %.2f W\n\n', Vmpp_meas, Impp_meas, Pmax_meas);
    
    % Plot measured I-V and P-V with markers
    figure('Name','Measured PV Curve & ANN Prediction','NumberTitle','off','Color','w','Position',[100 100 1100 420]);
    subplot(1,2,1);
    plot(V_meas, I_meas, '-b','LineWidth',1.6); hold on;
    plot(Vmpp_meas, Impp_meas, 'ro','MarkerFaceColor','r','MarkerSize',8);
    plot(Vmpp_pred, Impp_at_pred, 'gs','MarkerFaceColor','g','MarkerSize',9);
    xlabel('Voltage (V)'); ylabel('Current (A)');
    title(sprintf('I–V (measured) | G≈%.0f W/m², T≈%.1f°C', mean(data.Irradiance(idx_close)), mean(data.Temperature(idx_close))));
    legend('Measured I–V','Measured MPP','ANN Prediction','Location','best');
    grid on;
    
    subplot(1,2,2);
    plot(V_meas, P_meas, '-k','LineWidth',1.6); hold on;
    plot(Vmpp_meas, Pmax_meas, 'ro','MarkerFaceColor','r','MarkerSize',8);
    plot(Vmpp_pred, P_at_pred, 'gs','MarkerFaceColor','g','MarkerSize',9);
    xlabel('Voltage (V)'); ylabel('Power (W)');
    title('P–V (measured)');
    legend('Measured P–V','Measured MPP','ANN Prediction','Location','best');
    grid on;
    
    % Show numeric error
    errV = Vmpp_pred - Vmpp_meas;
    errPct = errV / Vmpp_meas * 100;
    fprintf('Error: ΔV = %.2f V (%.2f%%)\n\n', errV, errPct);
end

fprintf('Interactive session ended.\n');
