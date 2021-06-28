%% CONFIG_SRV02
%
% Accepts the user-defined configuration of the Quanser SRV02 plant and 
% sets its model variables accordingly.
%
% ************************************************************************
% Input parameters:
% EXT_GEAR_CONFIG   External gear configuration: 'LOW' or 'HIGH'
% ENCODER_TYPE      Encoder type: 'E' or 'EHR'
% TACH_OPTION       Tachometer option (e.g. SRV02-T): 'YES' or 'NO'
% AMP_TYPE          Universal Power Module type, e.g. 'VoltPAQ'
% LOAD_TYPE         Type of load: 'NONE', 'DISC', or 'BAR'
%
% ************************************************************************
% Output parameters:
% Rm        Motor armaturce resistance                          (ohm)
% kt        Motor torque constant                               (N.m/A)
% km        Motor back-EMF constant                             (V.s/rad)
% Kg        Total gear ratio
% eta_g     Gearbox efficiency
% eta_m     Motor efficiency
% Beq       Equivalent viscous damping coefficient w.r.t. load  (N.m.s/rad)
% Jm        Motor armature moment of inertia                    (kg.m^2)
% Jeq       Equivalent moment of inertia w.r.t. load            (kg.m^2)
% K_POT     Potentiometer Sensitivity                           (rad/V)
% K_TACH    Tachometer Sensitivity                              (rad/s/V)
% K_ENC     Encoder Resolution                                  (rad/count)
% VMAX_AMP  Amplifier Maximum Output Voltage                          (V)
% IMAX_AMP  Amplifier Maximum Output Current                          (A)
%
% Copyright (C) 2010 Quanser Consulting Inc.
% Quanser Consulting Inc.
%%
% 
function [ Rm, kt, km, Kg, eta_g, Beq, Jm, Jeq, eta_m, K_POT, K_TACH, K_ENC, VMAX_AMP, IMAX_AMP ] = config_srv02( EXT_GEAR_CONFIG, ENCODER_TYPE, TACH_OPTION, AMP_TYPE, LOAD_TYPE )
    % Set constant SRV02 parameters
    [ Rm, kt, km, Kgi, eta_g, eta_m ] = set_srv02_constants();
    % Calculate motor and gear inertias
    [ Jm, J24, J72, J120 ] = calc_motor_and_gear_inertias( TACH_OPTION );
    % Calculate SRV02 Model Parameters
    [ Kg, Beq, Jeq ] = calc_eq_damping_inerta( EXT_GEAR_CONFIG, LOAD_TYPE, Kgi, eta_g, Jm, J24, J72, J120 );
    % Set sensor and actuator calibration gains
    [K_POT, K_TACH, K_ENC, VMAX_AMP, IMAX_AMP] = srv02_calib ( ENCODER_TYPE, AMP_TYPE, Kg );
    
end
%
%
%% Sets constant SRV02 model parameters.
function [ Rm, kt, km, Kgi, eta_g, eta_m ] = set_srv02_constants( )
    % Calculate Useful Conversion Factors
    [ K_R2D, K_D2R, K_IN2M, K_M2IN, K_RDPS2RPM, K_RPM2RDPS, K_OZ2N, K_N2OZ, K_LBS2N, K_N2LBS, K_G2MS, K_MS2G ] = calc_conversion_constants ();
    %
    % Armature Resistance (Ohm)
    Rm = 2.6;
    % Motor Torque Constant (N.m/A)
    kt = 1.088 * K_OZ2N * K_IN2M; % = .00767
    % Continuous torque (N.m/A)
    M_e_max = 0.566 * K_OZ2N * K_IN2M; % = 0.566 oz.in (parameter not used)
    % Motor Back-EMF Constant (V.s/rd)
    km = 0.804 / 1000 * K_RDPS2RPM; % = .00767
    % Internal Gear Ratio (of the Planetary Gearbox)
    Kgi = 14;
    % Gearbox Efficiency
    eta_g = 0.90;
    % Motor ElectroMechanical Efficiency
    eta_m = 0.69;
end
%
%
%% Calculates the SRV02 motor and external gear inertias (kg.m^2)
function [ Jm, J24, J72, J120 ] = calc_motor_and_gear_inertias( TACH_OPTION )
    % Calculate Useful Conversion Factors
    [ K_R2D, K_D2R, K_IN2M, K_M2IN, K_RDPS2RPM, K_RPM2RDPS, K_OZ2N, K_N2OZ, K_LBS2N, K_N2LBS, K_G2MS, K_MS2G ] = calc_conversion_constants ();    
    %
    % Find motor shaft inertia.
    % Rotor Inertia (kg.m^2)
    Jm_rotor = 5.523e-5 * K_OZ2N * K_IN2M; % = 3.9e-7
    % Tachometer Armature Inertia, if any (kg.m^2)
    if strcmp ( TACH_OPTION, 'YES')
        Jtach = 1e-5 * K_OZ2N * K_IN2M; % = 7e-8
    else
        Jtach = 0;
    end
    % Motor Equivalent Inertia (kg.m^2)
    Jm = Jm_rotor + Jtach;
    %
    % External Gears Inertias (kg.m^2)
    % J24: 24-tooth Gear Inertia (on the Motor Shaft)
    m24 = 0.005; % mass (kg)
    r24 = 0.5 / 2 * 0.0254; % radius (m)
    J24 = m24 * r24^2 / 2;
    % J72: 72-tooth Gear Inertia (on the Potentiometer Shaft)
    m72 = 0.030; % mass (kg)
    r72 = 1.5 / 2 * 0.0254; % radius (m)
    J72 = m72 * r72^2 / 2;
    % J120: 120-tooth Gear Inertia (on the Load Shaft)
    m120 = 0.083; % mass (kg)
    r120 = 2.5 / 2 * 0.0254; % radius (m)
    J120 = m120 * r120^2 / 2;
end
%
%
%% Calculates SRV02 model parameters based on its configuration
function [ Kg, Beq, Jeq ] = calc_eq_damping_inerta( EXT_GEAR_CONFIG, LOAD_TYPE, Kgi, eta_g, Jm, J24, J72, J120 )    
    %
    % Moment of inertia of load attached to load shaft of SRV02
    if strcmp (LOAD_TYPE, 'NONE')
        Jl_ext = 0;
    elseif strcmp (LOAD_TYPE, 'DISC')
        Jl_ext = 1/2 * 0.04 * (0.05)^2; % J = mr^2 / 2
    elseif strcmp (LOAD_TYPE, 'BAR')
        Jl_ext = 1/12 * 0.038 * (0.1525)^2 + 0.038 * (0.0575)^2; % J = mL^2 / 12 + mR^2 (Moment about center then shifted)
    elseif strcmp (LOAD_TYPE, 'SLIP_RING')
        mtracker = 0.244; % mass (kg)
        rtracker = 2 * 0.0254; % radius (m)
        Jl_ext = mtracker * rtracker^2 / 2; % J = mr^2 / 2
    elseif strcmp (LOAD_TYPE, 'ROTFLEX')
        Jl_ext = 0.0005; % Calculated Experimentally
    elseif strcmp (LOAD_TYPE, 'FLEXGAGE')
        Jl_ext = 0.0002; % Calculated Experimentally
    elseif strcmp (LOAD_TYPE, 'ROTPEN')
        Jl_ext = 0.0015; % Calculated Experimentally
    elseif strcmp (LOAD_TYPE, 'GYRO')
        Jl_ext = 0.002; % Calculated Experimentally
    elseif ( strcmp (LOAD_TYPE, 'TORSION_1DOF') || strcmp (LOAD_TYPE, 'TORSION_2DOF') )
        % Found experimentally
        Jl_ext = 0.0001;    
    else
        error( 'LOAD_TYPE set not recognized.' )
        Jl_ext = 0;
    end    
    %
    % Moment of inertia and viscous damping based on external gear
    % configuration.
    if strcmp (EXT_GEAR_CONFIG, 'LOW')
        % Low Gear Configuration: (3x) 72-tooth gears
        Kge = 1;
        Kg = Kgi * Kge;
        % Equivalent moment of inertia including load (kg.m^2)
        Jg = 3 * J72;
        % Equivalent Viscous Damping Coefficient as seen at the Load (N.m.s/rd)                
        Beq = 15e-5;
        
    elseif strcmp (EXT_GEAR_CONFIG, 'HIGH')
        % High Gear Configuration: 
        % (1x) 24-tooth gear, (2x) 72-tooth gear, (1x) 120-tooth gear
        Kge = 5;
        Kg = Kgi * Kge;
        % Equivalent moment of inertia including load (kg.m^2)
        Jg = J24 + 2 * J72 + J120;
        % Equivalent Viscous Damping Coefficient as seen at the Load (N.m.s/rd)
        Beq = 15e-3;
    else
        error( 'EXT_GEAR_CONFIG set not recognized.' )
        Kge = 0;
        Kg = 0;
        Jg = 0;
        Beq = 0;
    end
    % Load moment of inertia: gears and external (kg.m^2)
    Jl = Jg + Jl_ext;
    % Equivalent moment of inertia including load (kg.m^2)    
    Jeq = Kg^2 * Jm * eta_g + Jl;
end
%
%
%% Sets sensor and amplifier calibration gains.
function [K_POT, K_TACH, K_ENC, VMAX_AMP, IMAX_AMP] = srv02_calib ( ENCODER_TYPE, AMP_TYPE, Kg )
    % Potentiometer Sensitivity (rd/V)
    K_POT = -(352 * pi / 180 / 10);
    % Tachometer Sensitivity (rd/s/V)
    K_TACH = -(1000 * 2 * pi / 60 / 1.5 / Kg);
    % Encoder Resolution, for a quadrature encoder, (rd/count)
    if strcmp (ENCODER_TYPE,'E')
        K_ENC = 2 * pi / ( 4 * 1024 );
    elseif strcmp (ENCODER_TYPE,'EHR')
        K_ENC = 2 * pi / ( 4 * 2048 );
    else
        error( 'ENCODER_TYPE set not recognized.' )
        K_ENC = 0;
    end

    % Set the amplifier Maximum Output Voltage (V) and Output Current (A)
    if strcmpi (AMP_TYPE, 'VoltPAQ')
        VMAX_AMP = 24;
        IMAX_AMP = 4;
    elseif strcmpi (AMP_TYPE, 'UPM_2405')
        VMAX_AMP = 24;
        IMAX_AMP = 5;
    elseif strcmpi (AMP_TYPE, 'UPM_1503')
        VMAX_AMP = 15;
        IMAX_AMP = 3;
    elseif strcmpi (AMP_TYPE, 'Q3')
        IMAX_AMP = 1.6;
        VMAX_AMP = 2.6 * IMAX_AMP; % Multiply times resistance of SRV02
    else
        error( 'AMP_TYPE set not recognized.' )
        VMAX_AMP = 0;
        IMAX_AMP = 0;
    end
end