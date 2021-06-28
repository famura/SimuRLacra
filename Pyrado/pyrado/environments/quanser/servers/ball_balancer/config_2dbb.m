%% CONFIG_2DBB
%
% CONFIG_2DBB sets the model variables of the Quanser 2D Ball Balance
% plant.
%
% ************************************************************************
% Output parameters:
% L_tbl     Length (and width) of balance table                 (m)
% r_arm     Distance between SRV02 output gear shaft and 
%           coupled joint                                       (m)
% J_b       Moment of inertia of ball.                          (kg.m^2)
% gravity_const         Gravitational constant                              (m/s^2)
% r_b       Radius of ball                                      (m)
% m_b       Mass of ball                                        (kg)
% THETA_MIN Minimum SRV02 Load Angle                            (rad)
% THETA_MAX Maximum SRV02 Load Angle                            (rad)
%
% Copyright (C) 2007 Quanser Consulting Inc.
% Quanser Consulting Inc.
%%
% 
function [ L_tbl, r_arm, r_b, m_b, J_b, gravity_const, THETA_MIN, THETA_MAX] = config_2dbb( )
    % Calculate useful conversion factors
    [ K_R2D, K_D2R, K_IN2M, K_M2IN, K_RDPS2RPM, K_RPM2RDPS, K_OZ2N, K_N2OZ, K_LBS2N, K_N2LBS, K_G2MS, K_MS2G ] = calc_conversion_constants ();    
    % Table width and length (m)
    L_tbl = 27.5 / 100; 
    % Distance between SRV02 output gear shaft and coupled joint (m)
    r_arm = 1 * K_IN2M;
    % Gravitational constant (m/s^2)
    gravity_const = 9.81;
    % Radius of ball (m)
    r_b = 39.25 / 2 / 1000;
	% Mass of ball (kg)
    m_b = 0.003;
    % Moment of inertia of ball (kg.m^2)
    J_b = 2/5 * m_b * r_b^2;
    % Minimum SRV02 Load Angle (rad)
    THETA_MIN = - 30.0 * K_D2R;
    % Maximum SRV02 Load Angle (rad)
    THETA_MAX = - THETA_MIN;
end