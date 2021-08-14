%% D_2DBB_PD
%
% Designs a proportional-derivative (PD) position controller for 2 DOF
% Ball Balancer plant based on the desired overshoot and settling time.
%
% ************************************************************************
% Input paramters:
% K         Model steady-state gain                 (rad/s/V)
% tau       Model time constant                     (s)
% PO        Percentage overshoot specification      (%)
% ts        Settling time specifications            (s)
% c_ts      Settling time percentage                (%)
%
% ************************************************************************
% Output parameters:
% kp        Proportional gain                       (V/rad)
% kv        Velocity gain                           (V.s/rad)
%
% Copyright (C) 2013 Quanser Consulting Inc.
% Quanser Consulting Inc.
%%
%
function [ kp, kv ] = d_2dbb_pd( K_bb, PO, ts, c_ts )
    % Damping ratio from overshoot specification.
    zeta = -log(PO/100) * sqrt( 1 / ( ( log(PO/100) )^2 + pi^2 ) );
    % Natural frequency from specifications (rad/s)
    wn = -log( c_ts * (1-zeta^2)^(1/2) ) / (zeta * ts);
    % Proportional gain (rad/m)
    kp = wn^2/K_bb;
    % Velocity gain (rad.s/m)   
    kv = 2*zeta*wn/K_bb;
end