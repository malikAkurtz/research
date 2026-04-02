import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from utils import (
    create_free_evolution_unitary,
    create_kick_unitary,
    create_ladder_operators,
    measure_populations,
    spherical_coords,
    spherical_to_rectangular,
)


def main():
    omega_q = 2 * np.pi * 5e9       # [Rad/s]
    alpha   = 2 * np.pi * (-250e6)  # [Rad/s]
    theta   = 0.03
    dim     = 3

    T   = 2 * np.pi / omega_q  # qubit period
    H_0 = np.diag([0, omega_q, 2 * omega_q + alpha])
    
    a, a_dagger = create_ladder_operators(dim=dim)
    U_kick = create_kick_unitary(K=(a_dagger - a), theta=theta)
    
    gate = "H"
    
    U_free_1 = create_free_evolution_unitary(H=H_0, t=1*(T/4))
    U_free_2 = create_free_evolution_unitary(H=H_0, t=2*(T/4))
    U_free_3 = create_free_evolution_unitary(H=H_0, t=3*(T/4))
    U_free_4 = create_free_evolution_unitary(H=H_0, t=4*(T/4))
        
        
    theta_target = np.pi / 2
    N = int(np.round(theta_target / theta)) +  1

    # ---- Pre-compute all states ----
    psi = np.array([1, 0, 0], dtype=complex)

    all_P0, all_P1, all_P2 = [], [], []
    all_x, all_y, all_z    = [], [], []

    for i in range(N):
        if gate == "X":
            psi = U_free_1 @ psi
            psi = U_kick @ psi
            psi = U_free_3 @ psi
        elif gate == "Y":
            psi = U_kick @ psi
            psi = U_free_4 @ psi
        elif gate == "Z":
            psi = U_free_4 @ psi
        elif gate == "H":
            psi = U_kick @ psi
            psi = U_free_4 @ psi
        
        if gate == "H" and i == N-1:
            psi = U_free_2 @ psi

        pops = measure_populations(psi)
        all_P0.append(pops[0])
        all_P1.append(pops[1])
        all_P2.append(pops[2])

        azimuth, inclination = spherical_coords(psi)
        x, y, z = spherical_to_rectangular(azimuth, inclination)
        all_x.append(x)
        all_y.append(y)
        all_z.append(z)

    kicks = np.arange(1, N + 1)

    # ---- Set up figure ----
    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor("#0f1117")

    # Bloch sphere (left)
    ax_bloch = fig.add_subplot(121, projection="3d")
    # Population plot (right)
    ax_pop = fig.add_subplot(122)

    # ---- Draw static Bloch sphere wireframe ----
    def draw_bloch_wireframe(ax):
        ax.set_facecolor("#0f1117")

        # Wireframe sphere
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 20)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(xs, ys, zs, color="gray", alpha=0.07, linewidth=0.4)

        # Equator and meridians
        circle = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(circle), np.sin(circle), np.zeros_like(circle),
                color="gray", alpha=0.2, linewidth=0.6)
        ax.plot(np.cos(circle), np.zeros_like(circle), np.sin(circle),
                color="gray", alpha=0.2, linewidth=0.6)
        ax.plot(np.zeros_like(circle), np.cos(circle), np.sin(circle),
                color="gray", alpha=0.2, linewidth=0.6)

        # Axes
        ax.plot([-1.4, 1.4], [0, 0], [0, 0], color="#ef4444", alpha=0.5, linewidth=1)
        ax.plot([0, 0], [-1.4, 1.4], [0, 0], color="#22c55e", alpha=0.5, linewidth=1)
        ax.plot([0, 0], [0, 0], [-1.4, 1.4], color="#3b82f6", alpha=0.5, linewidth=1)

        ax.text(1.55, 0, 0, "X", color="#ef4444", fontsize=10, fontweight="bold")
        ax.text(0, 1.55, 0, "Y", color="#22c55e", fontsize=10, fontweight="bold")
        ax.text(0, 0, 1.55, "|0⟩", color="#3b82f6", fontsize=10, fontweight="bold")
        ax.text(0, 0, -1.55, "|1⟩", color="#3b82f6", fontsize=10, fontweight="bold")

        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.set_axis_off()
        ax.set_title("Bloch Sphere Evolution via SFQ", color="#f59e0b",
                      fontsize=13, fontweight="bold", pad=10)

    def style_pop_axes(ax):
        ax.set_facecolor("#0f1117")
        ax.set_xlim(0, N + 1)
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel("Kick number", color="#94a3b8", fontsize=11)
        ax.set_ylabel("Population", color="#94a3b8", fontsize=11)
        ax.set_title("State Populations", color="#f59e0b",
                      fontsize=13, fontweight="bold", pad=10)
        ax.tick_params(colors="#64748b")
        for spine in ax.spines.values():
            spine.set_color("#1e293b")
        ax.grid(True, alpha=0.15, color="#334155")

    # How many kicks per animation frame (speed up)
    kicks_per_frame = 1

    def init():
        ax_bloch.cla()
        draw_bloch_wireframe(ax_bloch)

        ax_pop.cla()
        style_pop_axes(ax_pop)

        return []

    def update(frame):
        idx = min(frame * kicks_per_frame, N)

        # -- Bloch sphere --
        ax_bloch.cla()
        draw_bloch_wireframe(ax_bloch)

        if idx > 0:
            # Trajectory
            ax_bloch.plot(
                all_x[:idx], all_y[:idx], all_z[:idx],
                color="#f59e0b", linewidth=1.8, alpha=0.85
            )
            # Start point
            ax_bloch.scatter(
                [all_x[0]], [all_y[0]], [all_z[0]],
                color="#3b82f6", s=60, edgecolors="white", linewidths=1.5, zorder=10
            )
            # Current point
            ax_bloch.scatter(
                [all_x[idx - 1]], [all_y[idx - 1]], [all_z[idx - 1]],
                color="#22c55e", s=80, edgecolors="white", linewidths=1.5, zorder=10
            )

        ax_bloch.view_init(elev=20, azim=30 + frame * 0.5)

        # -- Population plot --
        ax_pop.cla()
        style_pop_axes(ax_pop)

        if idx > 0:
            k = kicks[:idx]
            ax_pop.plot(k, all_P0[:idx], color="#3b82f6", linewidth=2, label="|0⟩")
            ax_pop.plot(k, all_P1[:idx], color="#22c55e", linewidth=2, label="|1⟩")
            ax_pop.plot(k, all_P2[:idx], color="#ef4444", linewidth=1.5, linestyle="--",
                        label="|2⟩ (leakage)")
            ax_pop.legend(loc="center right", fontsize=10,
                          facecolor="#1a1d2e", edgecolor="#334155", labelcolor="#e2e8f0")

            # Kick counter text
            angle_so_far = idx * theta
            ax_pop.text(
                0.02, 0.95,
                f"Kick {idx}/{N}  |  θ_total = {angle_so_far:.2f} rad ({np.degrees(angle_so_far):.1f}°)",
                transform=ax_pop.transAxes,
                color="#f59e0b", fontsize=10, verticalalignment="top",
                fontfamily="monospace"
            )

        return []

    num_frames = N // kicks_per_frame + 1
    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=num_frames, interval=30, blit=False
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()