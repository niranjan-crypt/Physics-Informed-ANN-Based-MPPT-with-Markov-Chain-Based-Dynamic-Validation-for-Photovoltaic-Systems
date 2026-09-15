%% === Step 1: Load and preprocess full dataset ===
clc; clear; close all;

% Load your original dataset
data = readtable('clean_dataset.csv');

% Display original size
fprintf('Original dataset size: %d rows\n', height(data));

% Filter only valid daytime samples (irradiance > 10)
data = data(data.Irradiance > 10, :);
fprintf('Filtered daytime dataset size: %d rows\n', height(data));

% Optional: remove any missing or NaN rows
data = rmmissing(data);

% Show the range of key parameters
fprintf('Irradiance range: %.2f–%.2f W/m²\n', min(data.Irradiance), max(data.Irradiance));
fprintf('Temperature range: %.2f–%.2f °C\n', min(data.Temperature), max(data.Temperature));
fprintf('Voltage range: %.2f–%.2f V\n', min(data.Voltage), max(data.Voltage));
fprintf('Current range: %.2f–%.2f A\n', min(data.Current), max(data.Current));

% Compute Power (for reference / plotting)
data.Power = data.Voltage .* data.Current;

% Save the filtered 28k dataset
writetable(data, 'daytime_dataset.csv');
fprintf('Saved cleaned dataset as daytime_dataset.csv\n');
