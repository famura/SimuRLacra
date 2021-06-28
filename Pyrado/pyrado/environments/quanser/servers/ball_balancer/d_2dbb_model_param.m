%% D_2DBB_MODEL_PARAM
%
% Calculates the model gain for the X and Y axes in the 2D Ball Balancer 
% experiment.
%
% ************************************************************************
% Input parameters:
% r_arm     Distance between SRV02 output gear shaft and 
%           coupled joint                                       (m)
% L_tbl     Balance table length and width                      (m)
% r_b       Radius of ball                                      (m)
% m_b       Mass of ball                                        (kg)
% J_b       Moment of inertia of ball.                          (kg.m^2)
% g         Gravitational constant                              (m/s^2)
%
% ************************************************************************
% Output paramters:
% K_bb      Model gain                                          (m/s^2/rad)
%
% Copyright (C) 2007 Quanser Consulting Inc.
% Quanser Consulting Inc.
%%
%
function [ K_bb ] = d_2dbb_model_param(r_arm, L_tbl, r_b, m_b, J_b, g)
    % Model gain (m/s^2/rad)
%     K_bb = 2 * m_b * g * r_arm * r_b^2 / L_tbl / ( m_b * r_b^2 + J_b );
    K_bb = 6/5*g*r_arm/L_tbl;
end
