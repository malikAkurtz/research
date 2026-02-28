import numpy as np
import matplotlib.pyplot as plt
from utils import *

# -------------------------------------------------------------------------
#   Live State Evolution Plotter
# -------------------------------------------------------------------------

class LivePlotter:
    """
    Real-time visualization of Crank-Nicolson time evolution.

    Pass `plotter.update` as the `callback` argument to `crank_nicolson()`.

    Layout:
      Left        — Rotating-frame Bloch disk (X-Z plane).
                    Transforms to frame rotating at f_drive so only the
                    slow Rabi nutation is visible.  |0⟩ at north pole,
                    |1⟩ at south pole.  Resonant drive → clean arc.
                    Detuning → tilted ellipse / precession.
      Right top   — Qubit populations P_0, P_1
      Right bot   — Leakage P_2 on its own y-scale (typically ~10⁻² or less)
    """

    def __init__(self, basis: str, f_drive: float, dim_sub: int, n_cut: int, min_flux: float, max_flux: float, update_interval: int = 50):
        self.basis = basis
        self.n_cut = n_cut
        self.update_interval = update_interval
        self.dim_sub = dim_sub
        self.omega_d = 2 * np.pi * f_drive

        # Trajectory storage
        self.traj_x = []
        self.traj_y = []
        self.traj_z = []

        plt.ion()
        self.fig = plt.figure(figsize=(14, 6))

        # ---- Top Left: wavefunction in charge/fock basis ----
        self.ax_charge_state = self.fig.add_subplot(2, 3, 1)
        self.state_line,      = self.ax_charge_state.plot([], [], color="tab:red", linewidth=2, label=f'psi')
        if basis == "charge":
            self.ax_charge_state.set_xlabel(r'# Cooper Pairs (n)$')
        elif basis == "fock":
            self.ax_charge_state.set_xlabel(r'Energy Quanta (n)$')
        self.ax_charge_state.set_xlim(-self.n_cut, self.n_cut)
        self.ax_charge_state.set_ylabel('Probability')
        self.ax_charge_state.set_ylim(0.0, 1.0)
        self.ax_charge_state.axvline(0, color='black', linestyle='--', alpha=0.3)
        self.ax_charge_state.grid(True, alpha=0.3)
        self.ax_charge_state.legend(loc='upper right', fontsize=8)
        
        # ---- Right top: qubit populations ----
        self.ax_pop = self.fig.add_subplot(2, 3, 2)
        self.line_p0, = self.ax_pop.plot([], [], label=r'$P_0$', linewidth=1.5)
        self.line_p1, = self.ax_pop.plot([], [], label=r'$P_1$', linewidth=1.5)
        self.ax_pop.set_ylabel('Population')
        self.ax_pop.set_title('Qubit Populations')
        self.ax_pop.legend(loc='upper right', fontsize=8)
        self.ax_pop.set_ylim(-0.05, 1.05)
        self.ax_pop.grid(True, alpha=0.3)

        # ---- Right bottom: leakage on its own scale ----
        self.ax_leak = self.fig.add_subplot(2, 3, 3)
        self.line_p2, = self.ax_leak.plot([], [], color='tab:green',
                                           linewidth=1.5, label=r'$P_2$')
        self.ax_leak.set_xlabel('Time (ns)')
        self.ax_leak.set_ylabel('Probability')
        self.ax_leak.set_title('Leakage to Higher Levels')
        self.ax_leak.legend(loc='upper right', fontsize=8)
        self.ax_leak.grid(True, alpha=0.3)

        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # ---- Left Bottom: wavefunction in phase basis overlayed on phase vs potential energy ----
        self.ax_wave = self.fig.add_subplot(2, 2, 4)
        self.min_flux, self.max_flux = min_flux, max_flux
        self.node_phases = np.linspace(min_flux, max_flux, 400)

        self.wave_line, = self.ax_wave.plot([], [], color="tab:red", linewidth=2, label=f'psi')

        self.ax_wave.set_xlabel(r'Phase $\phi$')
        self.ax_wave.set_xlim(self.min_flux, self.max_flux)
        self.ax_wave.set_ylabel('Probability')
        self.ax_wave.set_ylim(0.0, 8.0)
        self.ax_wave.legend(loc='upper right', fontsize=8)
        
        # ---- Bloch Sphere representation of psi ----
        self.ax_bloch = self.fig.add_subplot(2, 2, 3, projection='3d')

        # Draw wireframe sphere
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 40)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(u.size), np.cos(v))
        self.ax_bloch.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.08, color='gray')

        # Reference points
        self.ax_bloch.scatter(0, 0, 1, color='black', s=20)   # |0⟩
        self.ax_bloch.scatter(0, 0, -1, color='black', s=20)   # |1⟩
        self.ax_bloch.text(0, 0, 1.15, r'$|0\rangle$', ha='center')
        self.ax_bloch.text(0, 0, -1.15, r'$|1\rangle$', ha='center')

        # State point and trajectory
        self.bloch_point = self.ax_bloch.scatter([], [], [], color='red', s=50)
        self.bloch_trail, = self.ax_bloch.plot([], [], [], color='red', alpha=0.3, linewidth=1)

        self.ax_bloch.set_xlim([-1, 1])
        self.ax_bloch.set_ylim([-1, 1])
        self.ax_bloch.set_zlim([-1, 1])
        self.ax_bloch.set_xlabel('X')
        self.ax_bloch.set_ylabel('Y')
        self.ax_bloch.set_zlabel('Z')
        self.ax_bloch.set_title('Bloch Sphere')
        

    def update(self, step, basis, C, omega, states, t_vec, P_0, P_1, P_2, psi):
        """Callback for crank_nicolson(). Redraws every update_interval steps."""
        if step % self.update_interval != 0 and step != len(t_vec) - 1:
            return

        s = step + 1
        t_ns = t_vec[:s] * 1e9
        t = t_vec[step]

        # --- Wavefunction charge/fock basis ---
        dim_sub = len(psi)
        truncated_states = states[:, :dim_sub]
        psi_original_basis = truncated_states @ psi
        probs = np.abs(psi_original_basis)**2
        if basis == "charge":
            n_vals = np.arange(-self.n_cut, self.n_cut + 1)
        elif basis == "fock":
            n_vals = np.arange(self.n_cut)
        self.state_line.set_data(n_vals, probs)

        # --- Qubit populations ---
        self.line_p0.set_data(t_ns, P_0[:s])
        self.line_p1.set_data(t_ns, P_1[:s])
        self.ax_pop.set_xlim(t_ns[0], t_ns[-1] if len(t_ns) > 1 else 1)

        # --- Leakage (auto-scaled y) ---
        self.line_p2.set_data(t_ns, P_2[:s])
        self.ax_leak.set_xlim(t_ns[0], t_ns[-1] if len(t_ns) > 1 else 1)
        self.ax_leak.relim()
        self.ax_leak.autoscale_view(scalex=False)
        
        # --- Phase basis projection ---
        if basis == "charge":
            psi_phase_basis  = charge_to_phase_basis(psi_original_basis.reshape(-1, 1), 
                                                 n_cut=self.n_cut, 
                                                 phases=self.node_phases
                                                 ).flatten()
        elif basis == "fock":
            psi_phase_basis  = fock_to_phase_basis(psi_original_basis.reshape(-1, 1), 
                                                 C=C,
                                                 omega=omega,
                                                 n_cut=self.n_cut, 
                                                 phases=self.node_phases
                                                 ).flatten()
        
        probs = np.abs(psi_phase_basis)**2
        self.wave_line.set_data(self.node_phases, probs)
            
        # --- Psi on Bloch Sphere
        azimuth, inclination = spherical_coords(state_energy_basis=psi)
        x, y, z              = spherical_to_rectangular(azimuth=azimuth, inclination=inclination)
        
        self.traj_x.append(x)
        self.traj_y.append(y)
        self.traj_z.append(z)

        self.bloch_point._offsets3d = ([x], [y], [z])
        self.bloch_trail.set_data_3d(self.traj_x, self.traj_y, self.traj_z)
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def finalize(self):
        """Switch back to blocking mode after evolution completes."""
        plt.ioff()
        plt.show()