load('ANN_MPPT_Model.mat');

% Example ANN predictions over time
Vmpp_pred = [27.0 27.2 27.3 27.25 27.35];
Power = [220 225 226 224 227];   % measured powers

% Build transition probabilities based on trend
states = [-1 0 1];  % decrease, same, increase
transitions = zeros(3);

for i = 2:length(Vmpp_pred)
    dVmpp = sign(Vmpp_pred(i) - Vmpp_pred(i-1));      % -1, 0, 1
    dP    = sign(Power(i) - Power(i-1));              % power response

    % learn transition: if dVmpp causes positive dP, increase prob for that
    if dP >= 0
        transitions(2+dVmpp,3) = transitions(2+dVmpp,3)+1;
    else
        transitions(2+dVmpp,1) = transitions(2+dVmpp,1)+1;
    end
end

% Normalize to get probability matrix
P = transitions ./ sum(transitions,2);

% Apply correction
lastChange = sign(Vmpp_pred(end) - Vmpp_pred(end-1));
if P(2+lastChange,3) > 0.5
    Vmpp_next = Vmpp_pred(end) + 0.05;  % increase Vmpp
elseif P(2+lastChange,1) > 0.5
    Vmpp_next = Vmpp_pred(end) - 0.05;  % decrease Vmpp
else
    Vmpp_next = Vmpp_pred(end);         % hold Vmpp
end

fprintf('\nMarkov refined Vmpp_next = %.3f V\n', Vmpp_next);
