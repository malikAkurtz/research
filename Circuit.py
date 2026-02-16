###############################################################################
#                                                                             #
#   Circuit.py                                                                #
#                                                                             #
#   Core module for superconducting circuit simulation using the method of    #
#   nodes. Builds a circuit graph from a user-defined topology, constructs    #
#   capacitance & inductance matrices, performs charge-basis quantization,    #
#   diagonalization, Crank-Nicolson time evolution, and gate fidelity         #
#   analysis. Also provides plotting utilities for wavefunctions, potential   #
#   energy landscapes, pulse sequences, and Rabi oscillations.               #
#                                                                             #
###############################################################################

import numpy as np
from scipy.constants import h, hbar, e
from typing import Optional
from scipy.signal import find_peaks
from Elements import Node, Capacitor, Inductor, JosephsonJunction, Graph

# ---- Fundamental constant ----
PHI_0 = hbar / (2 * e)  # Reduced flux quantum [Wb]

###############################################################################
#                                                                             #
#   Circuit Class                                                             #
#                                                                             #
###############################################################################

class Circuit:
    """
    Circuit class to build and store relevant data for a circuit
    using the method of nodes.
    """

    # =====================================================================
    #   Constructor
    # =====================================================================

    def __init__(self, graph_rep: dict):
        self.labels                                           = graph_rep["nodes"] # ["a", "b", "c", etc.]
        self.circuit_graph                                    = Circuit._build_master_graph(graph_rep) # collection of edges,vertices
        self.capacitive_sub_graph, self.inductive_sub_graph   = self._build_sub_graphs() # capactive edges,vertices + inductive edges,vertices
        self.active_nodes, self.passive_nodes                 = self._partition_nodes()
        self.N                                                = len(self.active_nodes)
        self.P                                                = self.N + len(self.passive_nodes) + 1 # for ground
        self.capacitance_matrix, self.inv_inductance_matrix   = self._build_matrices()
        self.inv_capacitance_matrix                           = np.linalg.inv(self.capacitance_matrix)
        self.offset_dict                                      = graph_rep["external_flux"]
        self.omega_squared                                    = self._build_omega_squared()
        self.normal_modes_squared, self.normal_vecs_squared   = np.linalg.eig(self.omega_squared)
        self.josephson_junctions                              = graph_rep["josephson_junctions"]

        # --- Placeholders (populated by later pipeline stages) ---
        self.n_hat, self.H_hat                                = None, None
        self.energies, self.states                            = None, None
        self.n_hat_energy                                     = None
        self.t_vec, self.At_vec, self.P_0, self.P_1, self.P_2 = None, None, None, None, None
        self.rabi_period                                      = None
        self.fidelity                                         = None
        self.U                                                = None

    # =====================================================================
    #   Graph Construction
    # =====================================================================

    @staticmethod
    def _build_master_graph(graph_rep: dict) -> Graph:
        """
        Build the full circuit graph from the user-supplied dictionary.
        Creates Node/Branch objects and wires them together.
        Also breaks symmetry by adding a tiny capacitor to ground for
        any node that lacks a capacitive branch.
        """
        gnd = Node(label="gnd")

        nodes    = {"gnd" : gnd}
        branches = []

        # --- Create node objects ---
        for i in range(len(graph_rep["nodes"])):
            node_label = graph_rep["nodes"][i]
            nodes[node_label] = Node(label=node_label)

        # --- Create and assign capacitive branches ---
        for i in range(len(graph_rep['capacitors'])):
            entry = graph_rep['capacitors'][i]
            node1 = nodes[entry[0]]
            node2 = nodes[entry[1]]
            value = entry[2]

            capacitor = Capacitor(value=value, nodes=[node1, node2])

            branches.append(capacitor)
            node1.branches.append(capacitor)
            node2.branches.append(capacitor)

        # --- Create and assign inductive branches ---
        for i in range(len(graph_rep['inductors'])):
            entry = graph_rep['inductors'][i]
            node1 = nodes[entry[0]]
            node2 = nodes[entry[1]]
            value = entry[2]

            inductor = Inductor(value=value, nodes=[node1, node2])

            branches.append(inductor)
            node1.branches.append(inductor)
            node2.branches.append(inductor)

        # --- Create and assign Josephson Junction branches ---
        for i in range(len(graph_rep["josephson_junctions"])):
            entry = graph_rep["josephson_junctions"][i]
            node1 = nodes[entry[0]]
            node2 = nodes[entry[1]]
            EJ = entry[2]
            CJ = entry[3]

            josephson_junction = JosephsonJunction(EJ=EJ, CJ=CJ, nodes=[node1, node2])

            branches.append(josephson_junction)
            node1.branches.append(josephson_junction)
            node2.branches.append(josephson_junction)

        # --- Break symmetry ---
        # Add a tiny parasitic capacitor to ground for nodes with no
        # capacitive connection (prevents singular capacitance matrix)
        for node in nodes.values():
            if node.label == "gnd":
                continue
            has_capacitive_branch = False
            for branch in node.branches:
                if isinstance(branch, (Capacitor, JosephsonJunction)):
                    has_capacitive_branch = True
                    break
            if not has_capacitive_branch:
                capacitor = Capacitor(value=1e-20, nodes=[node, nodes["gnd"]])

                branches.append(capacitor)
                node.branches.append(capacitor)
                nodes["gnd"].branches.append(capacitor)

        return Graph(nodes, branches)

    # =====================================================================
    #   Sub-Graph Construction
    # =====================================================================

    def _build_sub_graphs(self) -> dict:
        """
        Split the master graph into capacitive and inductive sub-graphs.
        Josephson junctions appear in both (they have C and nonlinear L).
        """
        nodes    = self.circuit_graph.vertices
        branches = self.circuit_graph.edges

        capacitive_branches = []
        inductive_branches  = []

        for branch in branches:
            if isinstance(branch, (Capacitor, JosephsonJunction)):
                capacitive_branches.append(branch)
            if isinstance(branch, (Inductor, JosephsonJunction)):
                inductive_branches.append(branch)

        return Graph(nodes, capacitive_branches), Graph(nodes, inductive_branches)

    # =====================================================================
    #   Node Partitioning
    # =====================================================================

    def _partition_nodes(self):
        """
        Classify nodes as active (connected to an inductor/JJ) or passive.
        Assumes symmetry has already been broken in _build_master_graph.
        """
        active_nodes    = []
        passive_nodes   = []

        for node_label, node in self.circuit_graph.vertices.items():
            if node_label == "gnd":
                continue
            capacitive_degree, inductive_degree = node._get_degree()
            if (inductive_degree > 0):
                active_nodes.append(node)
            else:
                passive_nodes.append(node)

        return active_nodes, passive_nodes

    # =====================================================================
    #   Matrix Construction (Capacitance & Inverse Inductance)
    # =====================================================================

    def _build_matrices(self):
        """
        Build the reduced (ground-row/column removed) capacitance and
        inverse-inductance matrices from the circuit graph topology.
        Uses the standard graph-Laplacian approach.
        """
        capacitance_matrix      = np.zeros((self.P, self.P))
        inv_inductance_matrix   = np.zeros((self.P, self.P))

        for branch in self.circuit_graph.edges:
            node1_label = branch.nodes[0].label
            node2_label = branch.nodes[1].label

            # Map node label -> matrix index (ground = 0)
            if node1_label == "gnd":
                j = 0
            else:
                j = self.labels.index(node1_label) + 1

            if node2_label == "gnd":
                k = 0
            else:
                k = self.labels.index(node2_label) + 1

            # Populate off-diagonal entries
            if isinstance(branch, (Capacitor, JosephsonJunction)):
                capacitance_matrix[j][k] += -branch.C
                capacitance_matrix[k][j] += -branch.C
            else:
                inv_inductance_matrix[j][k] += -1 / branch.L
                inv_inductance_matrix[k][j] += -1 / branch.L

        # Fill diagonal so each row sums to zero (Laplacian property)
        for j in range(self.P):
            capacitance_matrix[j][j]    = -1 * np.sum(capacitance_matrix[j]).item()
            inv_inductance_matrix[j][j] = -1 * np.sum(inv_inductance_matrix[j]).item()

        # Remove ground node (row 0, col 0) to get reduced matrices
        capacitance_matrix = np.delete(capacitance_matrix, 0, axis=0)
        capacitance_matrix = np.delete(capacitance_matrix, 0, axis=1)

        inv_inductance_matrix = np.delete(inv_inductance_matrix, 0, axis=0)
        inv_inductance_matrix = np.delete(inv_inductance_matrix, 0, axis=1)

        return capacitance_matrix, inv_inductance_matrix

    # =====================================================================
    #   Classical Mechanics Helpers
    # =====================================================================

    def _get_node_flux_dot(self, node_charge):
        """Convert node charges to node flux derivatives (voltages)."""
        return self.inv_capacitance_matrix @ node_charge

    def get_kinetic_energy(self, node_charge):
        """T = (1/2) * dPhi^T * C * dPhi"""
        node_flux_dot = self._get_node_flux_dot(node_charge)
        return (node_flux_dot.T @ self.capacitance_matrix @ node_flux_dot) / 2

    def get_potential_energy(self, node_flux: np.ndarray):
        """
        Total potential energy = linear (inductive) + nonlinear (Josephson).
        Includes external flux offsets for loops with applied magnetic flux.
        """
        # --- Linear inductive term ---
        if self.inv_inductance_matrix.size > 0 and np.any(self.inv_inductance_matrix != 0):
            term_1 = (node_flux.T @ self.inv_inductance_matrix @ node_flux) / 2
        else:
            term_1 = 0

        # --- External flux offset correction ---
        term_2 = 0
        for (node1_label, node2_label), offset in self.offset_dict.items():
                j = self.labels.index(node1_label)
                k = self.labels.index(node2_label)

                inductance = -1 / self.inv_inductance_matrix[j][k]
                term_2 += ((node_flux[j] - node_flux[k]) * offset) / inductance

        linear_term = term_1 + term_2

        # --- Nonlinear Josephson term: -EJ * cos(phi) ---
        non_linear_term = 0
        phi0 = hbar / (2 * e)

        for jj in self.josephson_junctions:
            node1_label = jj[0]
            node2_label = jj[1]
            EJ          = jj[2]

            node1_flux = node_flux[self.labels.index(node1_label)] if node1_label != "gnd" else 0
            node2_flux = node_flux[self.labels.index(node2_label)] if node2_label != "gnd" else 0

            branch_flux = node1_flux - node2_flux
            phi = branch_flux / phi0

            non_linear_term += -EJ * np.cos(phi)

        return linear_term + non_linear_term

    def get_lagrangian(self, node_flux, node_charge):
        """L = T - V"""
        return self.get_kinetic_energy(node_charge) - self.get_potential_energy(node_flux)

    # =====================================================================
    #   Normal Mode Analysis
    # =====================================================================

    def _build_omega_squared(self):
        """
        Compute omega^2 = C^{-1} * L^{-1} for normal mode decomposition.
        Returns zeros for purely nonlinear (no linear inductor) circuits.
        """
        if np.any(self.inv_inductance_matrix != 0):
            return self.inv_capacitance_matrix @ self.inv_inductance_matrix
        else:
            return np.zeros((self.N, self.N))

    def get_hamiltonian(self, node_flux, node_charge):
        """H = (1/2) * Q^T * C^{-1} * Q + V(Phi)"""
        return 0.5 * (node_charge.T @ self.inv_capacitance_matrix @ node_charge) + self.get_potential_energy(node_flux)

    @staticmethod
    def get_LJ(EJ: float):
        """Josephson inductance: L_J = Phi_0^2 / E_J"""
        return PHI_0**2 / EJ

    # =====================================================================
    #   Quantization (Charge Basis)
    # =====================================================================

    def _quantize(self, n_cut: int):
        """
        Build the charge operator n_hat and Hamiltonian H_hat in the
        charge basis for a single-node circuit with one Josephson junction.

        Truncates Hilbert space to Cooper pair numbers n in [-n_cut, n_cut].
        H = 4*EC*n^2 - (EJ/2)*(|n><n+1| + |n+1><n|)
        """
        if self.N != 1:
            raise NotImplementedError("Quantization currently supports single-node circuits only")

        C_sum = self.capacitance_matrix[0][0]

        if self.josephson_junctions:
            EJ = self.josephson_junctions[0][2]

        # Cost to add one Cooper pair of charge
        EC = e**2 / (2*C_sum)

        # Define Hilbert space
        # e.g. [-20, -19, ..., 0, ..., 19, 20] for n_cut = 20
        # each n represents n Cooper pairs on the island
        n_vals = np.arange(-n_cut, n_cut + 1)
        dim = len(n_vals)
        n_hat = np.diag(n_vals)

        # Build Hamiltonian Operator
        H = (4 * EC * (n_hat @ n_hat)) - 0.5 * EJ * (np.diag(np.ones(dim-1), 1) + np.diag(np.ones(dim-1), -1))

        self.n_hat = n_hat
        self.H_hat = H

    # =====================================================================
    #   Diagonalization
    # =====================================================================

    def _diagonalize(self):
        """Diagonalize H_hat to obtain energy eigenvalues and eigenstates."""
        self.energies, self.states = np.linalg.eigh(self.H_hat)

    # =====================================================================
    #   Basis Change (Charge -> Energy)
    # =====================================================================

    def _change_basis(self):
        """Transform the charge operator into the energy eigenbasis:
           n_hat_energy = S^dag * n_hat * S
        """
        self.n_hat_energy = np.conjugate(self.states).T @ self.n_hat @ self.states

    # =====================================================================
    #   Rabi Period Estimation
    # =====================================================================

    def _calculate_rabi_period(self, dim_sub: int, init_state: int, detuning: float, N_pulses: int, amplitude_scale: float, sigma: float, lambda_drag: float, steps_per_period: int):
        """
        Run a long Crank-Nicolson simulation and extract the Rabi period
        from the peaks of the smoothed P_1 population oscillation.
        """
        self.crank_nicolson(
            dim_sub=dim_sub,
            init_state=init_state,
            detuning=detuning,
            N_pulses=N_pulses,
            amplitude_scale=amplitude_scale,
            sigma=sigma,
            lambda_drag=lambda_drag,
            steps_per_period=steps_per_period
            )

        # Smooth P_1 and find peaks to determine Rabi period
        window = 1000
        P1_smooth = np.convolve(self.P_1, np.ones(window)/window, mode='same')
        peaks, _ = find_peaks(P1_smooth, distance=10000)

        if len(peaks) >= 2:
            self.rabi_period = self.t_vec[peaks[1]] - self.t_vec[peaks[0]]
            return self.rabi_period
        else:
            return None

    # =====================================================================
    #   Drive Parameter Helpers
    # =====================================================================

    def _drive_params(self, detuning):
        """Compute drive frequency and period from qubit transition + detuning."""
        f0, f1 = self.energies[0:2] / h / 1e9
        f01_Hz = (f1 - f0) * 1e9
        f_drive = f01_Hz + detuning
        T_drive = 1 / f_drive
        return f01_Hz, f_drive, T_drive

    # =====================================================================
    #   Crank-Nicolson Time Evolution
    # =====================================================================

    def crank_nicolson(self, dim_sub: int, init_state: int, detuning: float, N_pulses: int, amplitude_scale: float, sigma: float, lambda_drag: float, steps_per_period: int):
        """
        Time-evolve an initial energy eigenstate under an SFQ Gaussian
        pulse train using the Crank-Nicolson (implicit midpoint) method.

        Parameters
        ----------
        dim_sub          : number of lowest energy levels to keep
        init_state       : index of the initial eigenstate (0 = ground)
        detuning         : drive frequency offset from f_01  [Hz]
        N_pulses         : total number of SFQ pulses
        amplitude_scale  : pulse amplitude as fraction of (E1 - E0)
        sigma            : Gaussian pulse width              [s]
        steps_per_period : time steps per drive period

        Returns
        -------
        psi : final state vector in the truncated energy basis
        """

        # --- Truncate to dim_sub lowest levels ---
        truncated_energies = self.energies[:dim_sub]
        H_0 = np.diag(truncated_energies)            # Free Hamiltonian
        n_op = self.n_hat_energy[:dim_sub, :dim_sub]  # Charge operator (drive coupling)

        # --- Drive frequency from qubit transition + detuning ---
        f01_Hz, f_drive, T_drive = self._drive_params(detuning)
        
        f0, f1, f2 = self.energies[0:3] / hbar     # [rad/s]
        alpha = (f2 - f1) - (f1 - f0)

        # --- Time grid ---
        T   = N_pulses * T_drive                      # Total simulation time [s]
        dt  = T_drive / steps_per_period              # Time step             [s]
        N_t = round(T / dt)                           # Number of time steps
        t_vec = np.arange(N_t) * dt

        # --- Pulse amplitude ---
        A_0 = amplitude_scale * (truncated_energies[1]-truncated_energies[0])  # [J]

        # --- Precompute Gaussian pulse centers ---
        pulse_centers = np.arange(N_pulses) * T_drive

        # --- Initial state: energy eigenstate |init_state> ---
        psi = np.zeros(dim_sub);
        psi[init_state] = 1.0;

        # --- Pre-allocate result arrays ---
        At_vec = np.zeros(N_t)      # Pulse envelope at each time step
        At_dot_vec = np.zeros(N_t)  # Derivative of Pulse envelope as each time step
        P_0 = np.zeros(N_t)         # Population of |0>
        P_1 = np.zeros(N_t)         # Population of |1>
        P_2 = np.zeros(N_t)         # Population of |2> (leakage)

        I = np.eye(dim_sub)         # Identity matrix
        
        # Quadrature coupling operator for DRAG correction (1-2 subspace only)
        # Hermitian σ_y-like operator: -i|1><2| + i|2><1| scaled by charge matrix element
        n_12_y = np.zeros((dim_sub, dim_sub), dtype=complex)
        n_12_y[1, 2] = -1j * n_op[1, 2]
        n_12_y[2, 1] =  1j * n_op[2, 1]

        # --- Main Crank-Nicolson loop ---
        for i in range(N_t):
            t = t_vec[i]
            t_mid = t + (dt / 2)  # Midpoint for implicit scheme

            # SFQ pulse train envelope: sum of Gaussians near t_mid
            # (only include pulses within ~4 sigma for efficiency)
            dt_to_pulses = t_mid - pulse_centers
            mask = np.abs(dt_to_pulses) < 4 * sigma
            
            At      = A_0 * np.sum(np.exp(-0.5 * (dt_to_pulses[mask] / sigma)**2))
            At_dot  = A_0 * np.sum((-1 * dt_to_pulses[mask] / (sigma**2)) * np.exp(-0.5 * (dt_to_pulses[mask] / sigma)**2))
            
            At_vec[i]     = At
            At_dot_vec[i] = At_dot

            # Full Hamiltonian at midpoint: H_0 + A(t) * n_op
            H_mid = H_0 + (At * n_op) + lambda_drag * (At_dot / alpha) * n_12_y

            # Crank-Nicolson matrices:
            #   A * psi(t+dt) = B * psi(t)
            #   A = I + (i*dt/2hbar)*H,   B = I - (i*dt/2hbar)*H
            A = I + (((1j*dt) / (2*hbar)) * H_mid)
            B = I - (((1j*dt) / (2*hbar)) * H_mid)

            # Solve for psi(t+dt)
            psi = np.linalg.solve(A, B @ psi)

            At_vec[i] = At
            P_0[i]    = np.abs(psi[0])**2  # Ground state population
            P_1[i]    = np.abs(psi[1])**2  # First excited state population
            P_2[i]    = np.abs(psi[2])**2  # Second excited state (leakage)

        self.t_vec, self.At_vec, self.P_0, self.P_1, self.P_2 = t_vec, At_vec, P_0, P_1, P_2

        return psi

    # =====================================================================
    #   Unitary Gate Construction
    # =====================================================================

    def _build_unitary(self, d: int, dim_sub: int, detuning: float, amplitude_scale: float, sigma: float, lambda_drag: float, steps_per_period: int):
        """
        Construct the d-column unitary by evolving each computational
        basis state |0>, |1>, ..., |d-1> for a pi-pulse duration
        (half a Rabi period) and collecting the resulting state vectors.
        """
        self.U = []

        t_pi = self.rabi_period / 2  # Pi-pulse duration [s]

        # Recompute drive parameters
        f01_Hz, f_drive, T_drive = self._drive_params(detuning)

        N_pi = round(t_pi / T_drive)  # Number of pulses for a pi-rotation

        # Evolve each basis state and collect final states as columns
        for i in range(d):
            final_state = self.crank_nicolson(
                dim_sub=dim_sub,
                init_state=i,
                detuning=detuning,
                N_pulses=N_pi,
                amplitude_scale=amplitude_scale,
                sigma=sigma,
                lambda_drag=lambda_drag,
                steps_per_period=steps_per_period
            )

            # Appending eigenstates as rows, so will need to tranpose
            self.U.append(final_state)

        self.U = np.array(self.U).T  # Transpose so columns = evolved states

    # =====================================================================
    #   Gate Fidelity
    # =====================================================================

    def _calculate_fidelity(self, d: int, U_target: Optional[np.ndarray] = np.array([[0, 1], [1, 0]])):
        """
        Average gate fidelity:  F = |Tr(U_target^dag * U_actual)| / d
        Default target is the Pauli-X gate.
        """
        return np.abs(np.trace(U_target.conjugate().T @ self.U[:d, :d])) / d

    # =====================================================================
    #   Connectivity Display
    # =====================================================================

    def connectivity(self):
        """Return a human-readable string of the circuit's node-branch topology."""
        s = ""
        s += "--- Circuit Connectivity -\n"
        for label, node in self.circuit_graph.vertices.items():
            s += (f"Node '{label}':\n")
            for branch in node.branches:
                other_node = [n for n in branch.nodes if n != node]
                connection = f"connected to {other_node[0].label}" if other_node else "grounded"
                if isinstance(branch, Capacitor):
                    s += (f"  - {"Capacitor"} ({branch.C:.2e}) {connection}\n")
                elif isinstance(branch, Inductor):
                    s += (f"  - {"Inductor"} ({branch.L:.2e}) {connection}\n")
                else:
                    s += (f"  - {"Josephson Junction"} ({branch.EJ:.2e}) ({branch.C:.2e}) {connection}\n")
        return s

    def __repr__(self):
        return self.connectivity()
