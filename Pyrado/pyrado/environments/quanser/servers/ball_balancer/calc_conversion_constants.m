%% CALC_CONVERSION_CONSTANTS
%
% Calculates useful conversion factors w.r.t. units.
%
% Copyright (C) 2007 Quanser Consulting Inc.
% Quanser Consulting Inc.
%%
%
function [ K_R2D, K_D2R, K_IN2M, K_M2IN, K_RDPS2RPM, K_RPM2RDPS, K_OZ2N, K_N2OZ, K_LBS2N, K_N2LBS, K_G2MS, K_MS2G ] = calc_conversion_constants ()
    % from radians to degrees
    K_R2D = 180 / pi;
    % from degrees to radians
    K_D2R = 1 / K_R2D;
    % from Inch to Meter
    K_IN2M = 0.0254;
    % from Meter to Inch
    K_M2IN = 1 / K_IN2M;
    % from rad/s to RPM
    K_RDPS2RPM = 60 / ( 2 * pi );
    % from RPM to rad/s
    K_RPM2RDPS = 1 / K_RDPS2RPM;
    % from oz-force to N
    K_OZ2N = 0.2780139;
    % from N to oz-force
    K_N2OZ = 1 / K_OZ2N;
    % Pound to Newton (N/lbs)
    K_LBS2N = 4.4482216;
    % Newton to Pound (lbs/N/)
    K_N2LBS = 1 / K_LBS2N;
    % from gravity_const to m/s^2
    K_G2MS = 9.81;
    % from m/s^2 to gravity_const
    K_MS2G = 1 / K_G2MS;
end