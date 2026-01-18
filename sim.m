%% Transmon + SFQ pulse train + Crank–Nicolson Rabi simulation (AG)
clear; close all; clc;

%% Physical constants
h    = 6.62607015e-34;      % Planck constant [J·s]
hbar = h/(2*pi);
q=1.6e-19; % electron chrage in Coulombs 

%% Transmon parameters
fC        = 250e6;          % Charging energy frequency EC/h [Hz]
EC        = h * fC;         % EC in Joules
EJ_over_EC = 50;            % EJ/EC ratio (typical transmon)
EJ        = EJ_over_EC * EC; % Josephson energy [J]

%% Charge basis (|n> states)
n_cut = 20;                 % max |n| to keep
n     = -n_cut:n_cut;       % integer charge states
N     = numel(n);           % Hilbert space dimension

%% Build static transmon Hamiltonian H in charge basis
% H = 4EC n^2 - (EJ/2)(|n><n+1| + h.c.)
main_diag = 4 * EC * (n.^2);
off_diag  = -(EJ/2) * ones(1, N-1);

H = diag(main_diag) ...
    + diag(off_diag, 1) ...
    + diag(off_diag, -1);

%% Diagonalize H
[V, D]      = eig(H);                % columns of V: eigenvectors in charge basis in Joules
[Evals, ix] = sort(diag(D));         % sort eigenvalues 
V           = V(:, ix);              % reorder eigenvectors

% Energies in GHz
E_GHz = Evals / h / 1e9;

% Plot Energies and Potential 

figure(1)
for k=1:N
    Y=[Evals(k) Evals(k) Evals(k)]/1e-3/q; %% meV
    X=[-pi 0 pi]
    plot(X,Y,'k','linewidth',3) %Plot energy eigenvalues in meV
    xlabel('x','fontsize',15)
    hold on
    ylabel('E (meV)','fontsize',15)
    set(gca,'LineWidth',1.5,'FontSize',12);
    hold on
end
phi = linspace(-pi, pi, 400); % superconducting phase
U   = -EJ * cos(phi);         % Josephson potential [J]
plot(phi,U/q/1e-3,'r','linewidth',3) % Plot potential in meV
axis([-pi pi -0.06 0.05])

% Basic spectroscopic info
f01 = (Evals(2) - Evals(1)) / h / 1e9;   % |0> -> |1>, GHz
f12 = (Evals(3) - Evals(2)) / h / 1e9;   % |1> -> |2>, GHz
anh = f12 - f01;                         % anharmonicity, GHz

fprintf('Transmon parameters:\n');
fprintf('  EC/h  = %.3f GHz\n', fC / 1e9);
fprintf('  EJ/EC = %.1f\n', EJ_over_EC);
fprintf('  f01   = %.3f GHz\n', f01);
fprintf('  f12   = %.3f GHz\n', f12);
fprintf('  Anharmonicity (f12 - f01) = %.3f GHz\n\n', anh);

%% Choose a truncated eigenbasis (low levels only)
dim_sub = 6;                            % keep first few levels
E_sub   = Evals(1:dim_sub);
H0_eig  = diag(E_sub);                  % static Hamiltonian in eigenbasis

%% Build drive operator in eigenbasis
% Model SFQ pulses as a time-dependent term: H_drive(t) = A(t) * n_hat
% Charge operator n_hat in charge basis is diag(n)
n_op_charge = diag(n);
% Transform to eigenbasis
n_op_eig_full = V' * n_op_charge * V;
n_op_eig      = n_op_eig_full(1:dim_sub, 1:dim_sub);

%% Drive frequency and SFQ pulse parameters
f01_Hz   = f01 * 1e9;             % |0>->|1| in Hz
delta_f  = -1e6;                  % detuning (Hz)
f_drive  = f01_Hz + delta_f;      % drive repetition rate ~ resonant with 0-1 + detuning
T_drive  = 1 / f_drive;           % pulse repetition period [s]


Npulses  = 1000;                  % number of SFQ pulses in the train
t_total  = Npulses * T_drive;     % total simulation time

sigma    = 15e-12;                 % SFQ pulse width (std dev) ~ 2-20 ps
A0       = 0.01 * (Evals(2)-Evals(1)); % Pulse amplitude in Joules 

%% Time discretization and Crank-Nicolson setup
steps_per_period = 200;          % time steps per drive period
dt       = T_drive / steps_per_period;
Nt       = round(t_total / dt);
t_vec    = (0:Nt-1).' * dt;

% Precompute pulse centers
pulse_centers = (0:Npulses-1) * T_drive;

% Initial state: ground state |0> in eigenbasis
psi = zeros(dim_sub, 1);
psi(1) = 1.0;

% Storage for populations
P0 = zeros(Nt, 1);
P1 = zeros(Nt, 1);
P2 = zeros(Nt, 1);


% Helper identity
I_sub = eye(dim_sub);

%% Time evolution via Crank–Nicolson
At=zeros(Nt,1);
for it = 1:Nt
    it
    t  = t_vec(it);
    tm = t + dt/2;   % mid-point time for CN Hamiltonian

    % SFQ pulse train envelope: sum of Gaussians centered at pulse_centers
    % A(tm) = sum_k A0 * exp(-(tm - t_k)^2 / (2 sigma^2))
    % For speed, only sum nearby pulses (within ~4 sigma)
    
    dt_to_pulses = tm - pulse_centers;
    mask         = abs(dt_to_pulses) < 4*sigma;
    A_t          = sum(A0 * exp(-0.5 * (dt_to_pulses(mask)/sigma).^2));
    At(it) = A_t;

    % Total Hamiltonian in eigenbasis at mid-time
    H_mid = H0_eig + A_t * n_op_eig;

    % Crank–Nicolson update:
    % (I + i dt/(2ħ) H_mid) psi_{t+dt} = (Ir - i dt/(2ħ) H_mid) psi_t
    A_mat = (I_sub + 1i * dt/(2*hbar) * H_mid);
    B_mat = (I_sub - 1i * dt/(2*hbar) * H_mid);

    psi = A_mat \ (B_mat * psi);         % solve linear system

    % Normalize (guards against numerical drift)
    psi = psi / norm(psi);

    % Store populations in |0>,|1>,|2>
    P0(it) = abs(psi(1))^2;
    P1(it) = abs(psi(2))^2;
    P2(it) = abs(psi(3))^2;
end

%% Plot pulse sequence
figure(2)
plot(t_vec(1:1000)*1e9,At(1:1000)/q/1e3,'linewidth',3)
xlabel('Time (ns)');
ylabel('Pulse (meV)');
title('SFQ Pulse shape');
set(gca,'LineWidth',1.5,'FontSize',12);
grid on;

%% Plot populations: Rabi oscillations + leakage
figure(3) 
hold on; box on;
plot(t_vec*1e9, P0, 'LineWidth', 2);  % P0 vs time [ns]
plot(t_vec*1e9, P1, 'LineWidth', 2);  % P1
plot(t_vec*1e9, P2, 'LineWidth', 2);  % P2 (leakage)
xlabel('Time (ns)');
ylabel('Population');
legend('P_0','P_1','P_2','Location','best');
title(sprintf('SFQ-driven Rabi dynamics: f_{drive} = %.3f GHz (Δ = %.3f GHz)', ...
              f_drive/1e9, delta_f/1e9));
set(gca,'LineWidth',1.5,'FontSize',12);
grid on;