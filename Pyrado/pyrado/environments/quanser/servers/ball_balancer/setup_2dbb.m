%% SETUP_2DBB
%
% Sets the necessary parameters to run the Quanser 2 DOF Ball Balancer
% experiment. 
% 
% Copyright (C) 2013 Quanser Consulting Inc.
%
clear all;
%
%% SRV02 Configuration
% External Gear Configuration: set to 'HIGH' or 'LOW'
EXT_GEAR_CONFIG = 'HIGH';
% Encoder Type: set to 'E' or 'EHR'
ENCODER_TYPE = 'E';
% Is SRV02 equipped with Tachometer? (i.e. option T): set to 'YES' or 'NO'
TACH_OPTION = 'YES';
% Type of Load: set to 'NONE', 'DISC', or 'BAR'
LOAD_TYPE = 'NONE';
% Amplifier Gain used: set to 3 when using VoltPAQ-X2 
% (or to 1 with VoltPAQ-X1)
K_AMP = 1;
% Amplifier Type: set to 'VoltPAQ'
AMP_TYPE = 'VoltPAQ';
% Digital-to-Analog Maximum Voltage (V)
VMAX_DAC = 10;
%
%% Lab Configuration
% Type of controller: set it to 'AUTO', 'MANUAL'
CONTROL_TYPE = 'AUTO';
% CONTROL_TYPE = 'MANUAL';
%
%% Control specifications
% 2DBB Position Control Specifications
% Settling time percentage
c_ts = 0.04;
% Settling time (s)
ts = 3.0; % 2.5 s
% Percentage overshoot (%)
PO = 10;
%
%% System Parameters
% Sets model variables according to the user-defined system configuration
[ Rm, kt, km, Kg, eta_g, Beq, Jm, Jeq, eta_m, K_POT, K_TACH, K_ENC, VMAX_AMP, IMAX_AMP ] = config_srv02( EXT_GEAR_CONFIG, ENCODER_TYPE, TACH_OPTION, AMP_TYPE, LOAD_TYPE );
% Load 2DBB model parameters.
[ L_tbl, r_arm, r_b, m_b, J_b, g, THETA_MIN, THETA_MAX ] = config_2dbb( );
% Load model parameters based on SRV02 configuration.
[ K, tau ] = d_model_param(Rm, kt, km, Kg, eta_g, Beq, Jeq, eta_m, AMP_TYPE);
%
%% Filter Parameters
% 2DBB High-pass filter in PD control used to compute velocity
% Cutoff frequency (rad/s)
wf = 2 * pi * 2.5;
%
%% Calculate Control Parameters
if strcmp ( CONTROL_TYPE , 'MANUAL' )
    % Calculate Balance Table model gain.
    K_bb = 0;
    % Design Balance Table PV Gains
    kp = 0;
    kd = 0;
    %
elseif strcmp ( CONTROL_TYPE , 'AUTO' )
    % Calculate Balance Table model gain.
    [ K_bb ] = d_2dbb_model_param(r_arm, L_tbl, r_b, m_b, J_b, g);
    % Design Balance Table PD Gains
    [ kp, kd ] = d_2dbb_pd( K_bb, PO, ts, c_ts );
end
%
%% Display
disp( ' ' );
disp( 'Balance Table model parameter: ' );
disp( [ '   K_bb = ' num2str( K_bb, 3 ) ' m/s^2/rad' ] );
disp( 'Balance Table Specifications: ' );
disp( [ '   ts = ' num2str( ts, 3 ) ' s' ] );
disp( [ '   PO = ' num2str( PO, 3 ) ' %' ] );
disp( 'Balance Table PID Gains: ' );
disp( [ '   kp_bb = ' num2str( kp, 3 ) ' rad/m' ] );
disp( [ '   kd_bb = ' num2str( kd, 3 ) ' rad.s/m' ] );
%