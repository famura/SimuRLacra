%% D_MODEL_PARAM
%
% Calculates the first-order model parameters, K and tau, of the Quanser
% SRV02 plant.
%
% ************************************************************************
% Input parameters:
% Rm        Motor armaturce resistance                          (ohm)
% kt        Motor torque constant                               (N.m/A)
% km        Motor back-EMF constant                             (V.s/rad)
% Kg        Total gear ratio
% eta_g     Gearbox efficiency
% eta_m     Motor efficiency
% Beq       Equivalent viscous damping coefficient w.r.t. load  (N.m.s/rad)
% Jeq       Equivalent moment of inertia w.r.t. load            (kg.m^2)
%
% ************************************************************************
% Output paramters:
% K         Model steady-state gain                             (rad/s/V)
% tau       Model time constant                                  (s)
%
% Copyright (C) 2007 Quanser Consulting Inc.
% Quanser Consulting Inc.
%%
%
function [K,tau] = d_model_param(Rm, kt, km, Kg, eta_g, Beq, Jeq, eta_m, daqb)        
    if strcmp(upper(daqb),'Q3')        
        % Actuator gain (N.m/A)
        Am = eta_g*eta_m*kt*Kg;
        % Steady-state gain (rad/s/A)
        K = Am / Jeq; % Am / Beq;
        % Time constant (s)
        tau = 1; % Jeq / Beq;
    else
        % Viscous damping relative to motor
        Beq_v = ( Beq*Rm + eta_g*eta_m*km*kt*Kg^2 ) / Rm;
        % Actuator gain (N.m/V)
        Am = eta_g*eta_m*kt*Kg / Rm;
        % Steady-state gain (rad/s/V)
        K = Am / Beq_v;
        % Time constant (s)
        tau = Jeq / Beq_v;
    end
end
