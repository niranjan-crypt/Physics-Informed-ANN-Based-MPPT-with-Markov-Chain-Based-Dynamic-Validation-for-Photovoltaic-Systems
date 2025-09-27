% 1) see what table column names MATLAB actually uses
data = readtable('clean_dataset.csv','VariableNamingRule','preserve');
data.Properties.VariableNames

% 2) build X and T the simple way and check sizes
X = table2array(data(:, {'Temperature','Irradiance','Voltage','Current'}));
T = table2array(data(:, {'Power'}));

size(X)   % likely [N x 4]
size(T)   % likely [N x 1]

% If X is N x inputs and T is N x 1, transpose them before training:
X2 = X';  % becomes [inputs x samples]
T2 = T';  % becomes [1 x samples]
size(X2), size(T2)



