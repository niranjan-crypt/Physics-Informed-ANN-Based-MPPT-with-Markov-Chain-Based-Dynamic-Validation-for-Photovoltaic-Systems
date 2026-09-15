% Load the dataset
data = readtable('raw_data.csv');

% Keep only required columns
cleanData = data(:, {'Temperature','Irradiance','Voltage','Current','Power'});

% Save as a new clean CSV
writetable(cleanData, 'clean_dataset.csv');
