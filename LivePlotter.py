import numpy as np
import matplotlib.pyplot as plt
from Circuit import Circuit, PHI_0, e
from utils import charge_to_phase_basis

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

    def __init__(self, circuit: Circuit, f_drive: float, dim_sub: int, n_cut: int, min_flux: float, max_flux: float, update_interval: int = 50):
        self.n_cut = n_cut
        self.update_interval = update_interval
        self.dim_sub = dim_sub
        self.omega_d = 2 * np.pi * f_drive

        # Trajectory storage
        self.traj_x = []
        self.traj_z = []

        plt.ion()
        self.fig = plt.figure(figsize=(14, 6))

        # ---- Top Left: rotating-frame Bloch disk (X-Z plane) ----
        self.ax_bloch = self.fig.add_subplot(2, 2, 1, aspect='equal')
        theta = np.linspace(0, 2 * np.pi, 100)
        self.ax_bloch.plot(np.cos(theta), np.sin(theta), 'k-', alpha=0.15, linewidth=0.8)
        self.ax_bloch.axhline(0, color='k', alpha=0.08, linewidth=0.5)
        self.ax_bloch.axvline(0, color='k', alpha=0.08, linewidth=0.5)
        self.ax_bloch.text(0, 1.15, r'$|0\rangle$', ha='center', fontsize=11)
        self.ax_bloch.text(0, -1.15, r'$|1\rangle$', ha='center', fontsize=11)
        self.ax_bloch.text(1.15, 0, r'$+X$', ha='left', fontsize=9, alpha=0.4)
        self.ax_bloch.text(-1.15, 0, r'$-X$', ha='right', fontsize=9, alpha=0.4)
        self.ax_bloch.set_xlim(-1.4, 1.4)
        self.ax_bloch.set_ylim(-1.4, 1.4)
        self.ax_bloch.set_title('Rotating Frame  (X-Z plane)')
        self.ax_bloch.set_xlabel(r'$\langle X \rangle$  (coherence)')
        self.ax_bloch.set_ylabel(r'$\langle Z \rangle$  (inversion)')
        self.ax_bloch.grid(True, alpha=0.1)
        # Line artists updated in-place (no cla needed → fast)
        self.bloch_trail, = self.ax_bloch.plot([], [], color='tab:blue',
                                                alpha=0.35, linewidth=0.8)
        self.bloch_point = self.ax_bloch.scatter([], [], color='tab:red',
                                                  s=60, zorder=5)

        # ---- Right top: qubit populations ----
        self.ax_pop = self.fig.add_subplot(2, 2, 2)
        self.line_p0, = self.ax_pop.plot([], [], label=r'$P_0$', linewidth=1.5)
        self.line_p1, = self.ax_pop.plot([], [], label=r'$P_1$', linewidth=1.5)
        self.ax_pop.set_ylabel('Population')
        self.ax_pop.set_title('Qubit Populations')
        self.ax_pop.legend(loc='upper right', fontsize=8)
        self.ax_pop.set_ylim(-0.05, 1.05)
        self.ax_pop.grid(True, alpha=0.3)

        # ---- Right bottom: leakage on its own scale ----
        self.ax_leak = self.fig.add_subplot(2, 2, 4)
        self.line_p2, = self.ax_leak.plot([], [], color='tab:red',
                                           linewidth=1.5, label=r'$P_2$ (leakage)')
        self.ax_leak.set_xlabel('Time (ns)')
        self.ax_leak.set_ylabel('Leakage')
        self.ax_leak.set_title('Leakage to Higher Levels')
        self.ax_leak.legend(loc='upper right', fontsize=8)
        self.ax_leak.grid(True, alpha=0.3)

        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # ---- Left Bottom: wavefunction in phase basis overlayed on eigenvalues ----
        self.ax_wave = self.fig.add_subplot(2, 2, 3)
        self.min_flux, self.max_flux = min_flux, max_flux
        self.node_phases = np.linspace(min_flux, max_flux, 400)

        # Plot static Potential Energy
        U = np.array([circuit.get_potential_energy(np.array([phi * PHI_0])) for phi in self.node_phases])
        self.ax_wave.plot(self.node_phases, U / e / 1e-3, 'k', linewidth=1.5, alpha=0.6, label='V(phi)')

        # Create line objects for the 3 lowest wavefunctions
        # We will shift these vertically by their energy levels in the update loop
        self.wave_line, = self.ax_wave.plot([], [], color="tab:red", linewidth=2, label=f'psi')

        self.ax_wave.set_xlabel(r'Phase $\phi$')
        self.ax_wave.set_ylabel('Energy (meV)')
        self.ax_wave.set_ylim(min(U/e/1e-3), max(U/e/1e-3) * 1.2) # Room for wavefunctions
        self.ax_wave.legend(loc='upper right', fontsize=8)

    def _psi_to_rotating_bloch(self, psi, t):
        """Bloch X, Z in the frame rotating at f_drive."""
        c0 = psi[0]
        c1 = psi[1] * np.exp(1j * self.omega_d * t)
        x = 2 * np.real(np.conj(c0) * c1)
        z = np.abs(c0)**2 - np.abs(c1)**2
        return x, z

    def update(self, step, energies, states, t_vec, P_0, P_1, P_2, psi):
        """Callback for crank_nicolson(). Redraws every update_interval steps."""
        if step % self.update_interval != 0 and step != len(t_vec) - 1:
            return

        s = step + 1
        t_ns = t_vec[:s] * 1e9
        t = t_vec[step]

        # --- Rotating-frame Bloch disk ---
        bx, bz = self._psi_to_rotating_bloch(psi, t)
        self.traj_x.append(bx)
        self.traj_z.append(bz)
        self.bloch_trail.set_data(self.traj_x, self.traj_z)
        self.bloch_point.set_offsets([[bx, bz]])

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
        dim_sub = len(psi)
        truncated_states = states[:, :dim_sub]
        psi_charge_basis = truncated_states @ psi
        psi_phase_basis  = charge_to_phase_basis(psi_charge_basis.reshape(-1, 1), 
                                                 n_cut=self.n_cut, 
                                                 phases=self.node_phases
                                                 ).flatten()
        
        probs = np.abs(psi_phase_basis)**2
        scaling_factor = 0.01  # Adjust this to make waves taller/shorter
        self.wave_line.set_data(self.node_phases, (probs * scaling_factor))
            
            
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def finalize(self):
        """Switch back to blocking mode after evolution completes."""
        plt.ioff()
        plt.show()