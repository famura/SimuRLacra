% SETUP_IP02_SIP
%
% IP02 Single Inverted Pendulum (SIP) Control Lab: 
% Design of a LQR position controller
% 
% SETUP_IP02_SIP sets the SIP and IP02 
% model parameters accordingly to the user-defined configuration.
% SETUP_IP02_SIP can also set the controllers' parameters, 
% accordingly to the user-defined desired specifications.
%
% Copyright (C) 2012 Quanser Consulting Inc.
% Quanser Consulting Inc.

clear all;

% ############### USER-DEFINED IP02 with SIP CONFIGURATION ###############

% if IP02: Type of Cart Load: set to 'NO_LOAD', 'WEIGHT'
 IP02_LOAD_TYPE = 'NO_LOAD';
% IP02_LOAD_TYPE = 'WEIGHT';
% Type of single pendulum: set to 'LONG_24IN', 'MEDIUM_12IN'
 PEND_TYPE = 'LONG_24IN'; 
% PEND_TYPE = 'MEDIUM_12IN'; 
% Turn on or off the safety watchdog on the cart position: set it to 1 , or 0 
X_LIM_ENABLE = 1;       % safety watchdog turned ON
%X_LIM_ENABLE = 0;      % safety watchdog turned OFF
% Safety Limits on the cart displacement (m)
X_MAX = 0.3;            % cart displacement maximum safety position (m)
X_MIN = - X_MAX;        % cart displacement minimum safety position (m)
% Turn on or off the safety watchdog on the pendulum angle: set it to 1 , or 0 
ALPHA_LIM_ENABLE = 1;       % safety watchdog turned ON
%ALPHA_LIM_ENABLE = 0;      % safety watchdog turned OFF
% Safety Limits on the pendulum angle (deg)
global ALPHA_MAX ALPHA_MIN
ALPHA_MAX = 20;            % pendulum angle maximum safety position (deg)
ALPHA_MIN = - ALPHA_MAX;   % pendulum angle minimum safety position (deg)
% Amplifier Gain: set VoltPAQ amplifier gain to 1
K_AMP = 1;
% Amplifier Type: set to 'VoltPAQ' or 'Q3'
AMP_TYPE = 'VoltPAQ';
% AMP_TYPE = 'Q3';
% Digital-to-Analog Maximum Voltage (V); for MultiQ cards set to 10
VMAX_DAC = 10;

% ############### USER-DEFINED CONTROLLER DESIGN ###############
% Type of Controller: set it to 'LQR_AUTO', 'MANUAL'  
%CONTROLLER_TYPE = 'LQR_AUTO';    % LQR controller design: automatic mode
CONTROLLER_TYPE = 'MANUAL';    % controller design: manual mode
% Initial Condition on alpha, i.e. pendulum angle at t = 0 (deg)
global IC_ALPHA0
% IC_ALPHA0 = 0.1;
IC_ALPHA0 = 0;
% conversion to radians
IC_ALPHA0 = IC_ALPHA0 / 180 * pi;
% Initialization of Simulink diagram parameters

    % Cart Encoder Resolution
    global K_EC K_EP
    
    % Specifications of a second-order low-pass filter
    wcf = 2 * pi * 10.0;  % filter cutting frequency
    zetaf = 0.9;        % filter damping ratio

% ############### END OF USER-DEFINED CONTROLLER DESIGN ###############


% variables required in the Simulink diagrams
global VMAX_AMP IMAX_AMP

% Set the model parameters accordingly to the user-defined IP02 system configuration.
% These parameters are used for model representation and controller design.
[ Rm, Jm, Kt, eta_m, Km, Kg, eta_g, Mc, r_mp, Beq ] = config_ip02( IP02_LOAD_TYPE, AMP_TYPE );

% Set the model parameters for the single pendulum accordingly to the user-defined system configuration.
% [ g, Mp, Lp, lp, Jp, Bp ] = config_sp( PEND_TYPE );

% Lumped Mass of the Cart System (accounting for the rotor inertia)
% Jeq = Mc + eta_g * Kg^2 * Jm / r_mp^2;

% Self-erecting SIP
% [Er, a_max] = d_swing_up(eta_m, eta_g, Kg, Kt, Rm, r_mp, Jeq, Mp, lp);
% epsilon = 10.0 * pi / 180;

% For the following state vector: X = [ xc; alpha; xc_dot; alpha_dot ]
% Initialization of the State-Space Representation of the Open-Loop System
% Call the following Maple-generated file to initialize the State-Space Matrices: A, B, C, and D
% ABCD Eqns relative to Fc
% [ A, B, C, D ] = SIP_ABCD_eqns(Rm, Kt, eta_m, Km, Kg, eta_g, Jeq, Mp, Bp, lp, g, Jp, r_mp, Beq);
%

% ############### LQR CONTROL ###############
% if strcmp ( CONTROLLER_TYPE, 'MANUAL' )
%     Q = diag([35 350 0.1 0.1]);
%     R = 0.02;
%     
%     [ K, S, EIG_CL ] = lqr( A, B, Q, R );
%     
% elseif  strcmp ( CONTROLLER_TYPE, 'LQR_AUTO')
%     [ K ] = d_ip02_sip_lqr( A, B, C, D, PEND_TYPE, IP02_LOAD_TYPE, AMP_TYPE);
%     % Display the calculated gains
%     disp( ' ' )
%     % disp( 'Calculated LQR controller gain elements: ' )
%         disp( [ 'K(1) = ' num2str( K(1) ) ' V/m' ] )
%         disp( [ 'K(2) = ' num2str( K(2) ) ' V/rad' ] )
%         disp( [ 'K(3) = ' num2str( K(3) ) ' V.s/m' ] )
%         disp( [ 'K(4) = ' num2str( K(4) ) ' V.s/rad' ] )
% 
% else
%     error( 'Error: Please set the type of controller that you wish to implement.' )
% end
