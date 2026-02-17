"""
Watty Chestahedron Core
=======================
The geometric container that organizes all memory.

Seven nodes. Vortex flow. Information circulates until it settles.
The organization is not designed — it emerges from the geometry.

Architecture:
        [1] INTAKE
       /          |
    [2]           [3]
  ANALYTICAL    INTUITIVE
     |             |
    [4]           [5]
   DEEPER       DEEPER
     |           /
      [6] MERGE
          |
    [7] VORTEX CORE
          ~
    (feeds back to 1)

Hugo & Rim & Claude — Feb 2026

Plasticity (the learning loop):
  W_out learns via importance-gated Hebbian update with felt modulation.
  Leak rates adapt via intrinsic plasticity — active-during-deep nodes
  slow down (longer integration), dormant nodes speed up (faster response).
  Deep signals amplify felt_state proportional to importance.
  Homeostatic decay pulls W_out toward identity (ground state).
  Learning:decay ratio = φ:1.
"""

import numpy as np

from watty.config import (
    CHESTAHEDRON_PLASTICITY_LR,
    CHESTAHEDRON_HOMEOSTATIC_RATE,
    CHESTAHEDRON_INTRINSIC_LR,
    CHESTAHEDRON_WOUT_SPECTRAL_MAX,
)

# ── Golden ratio constants ──────────────────────────────
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI
PHI_SQ_INV = 1 / (PHI ** 2)

# ── Chestahedron constants ──────────────────────────────
CHESTAHEDRON_DIM = 7
RESERVOIR_SIZE = 64
CIRCULATIONS = 3
SIGNAL_DIM = 48

FACES = ["TEMPORAL", "STRUCTURAL", "RELATIONAL", "SEMANTIC", "META", "INVERSE", "EXTERNAL"]


class ChestahedronNode:
    """A single frozen reservoir node."""

    def __init__(self, input_dim, reservoir_size=64, spectral_radius=0.95,
                 leak_rate=0.3, sparsity=0.8, seed=42):
        rng = np.random.RandomState(seed)
        self.reservoir_size = reservoir_size
        self.leak_rate = leak_rate

        self.W_in = rng.uniform(-0.5, 0.5, (reservoir_size, input_dim + 1))

        W = rng.randn(reservoir_size, reservoir_size)
        mask = rng.rand(reservoir_size, reservoir_size) > sparsity
        W *= mask

        eigenvalues = np.abs(np.linalg.eigvals(W))
        if eigenvalues.max() > 0:
            W = W * (spectral_radius / eigenvalues.max())
        self.W_res = W

        self.state = np.zeros(reservoir_size)

    def step(self, input_signal):
        u = np.concatenate([[1.0], input_signal])
        pre = self.W_in @ u + self.W_res @ self.state
        new_state = np.tanh(pre)
        self.state = (1 - self.leak_rate) * self.state + self.leak_rate * new_state
        return self.state.copy()

    def reset(self):
        self.state = np.zeros(self.reservoir_size)


class Chestahedron:
    """The seven-node vortex geometry."""

    FLOW = {
        0: [1, 2],
        1: [3],
        2: [4],
        3: [5],
        4: [5],
        5: [6],
        6: [0],
    }

    def __init__(self, input_dim=SIGNAL_DIM, reservoir_size=RESERVOIR_SIZE,
                 circulations=CIRCULATIONS, seed=42):
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.circulations = circulations
        self.n_nodes = 7

        self.nodes = []
        for i in range(self.n_nodes):
            n_incoming = sum(1 for src, dsts in self.FLOW.items() if i in dsts)
            node_input = input_dim + n_incoming * reservoir_size

            self.nodes.append(ChestahedronNode(
                input_dim=node_input,
                reservoir_size=reservoir_size,
                spectral_radius=0.95 - i * 0.02,
                leak_rate=0.2 + i * 0.05,
                sparsity=0.8,
                seed=seed + i * 1000,
            ))

        self.felt_state = np.zeros(self.n_nodes)
        self.felt_history = []

        rng = np.random.RandomState(seed + 9999)
        self.W_out = np.eye(self.n_nodes) + rng.randn(self.n_nodes, self.n_nodes) * 0.01

        self._running_mean = np.zeros(self.n_nodes)
        self._running_var = np.ones(self.n_nodes)
        self._n_processed = 0

        # Plasticity state
        self._last_normalized = None
        self._initial_leak_rates = [node.leak_rate for node in self.nodes]
        self._total_learn_steps = 0

    def process(self, signal):
        """Process a signal through the vortex. Returns (7D coordinate, energy)."""
        for node in self.nodes:
            node.reset()

        node_outputs = [np.zeros(self.reservoir_size) for _ in range(self.n_nodes)]

        for circulation in range(self.circulations):
            for i in range(self.n_nodes):
                incoming = [signal]
                for src, dsts in self.FLOW.items():
                    if i in dsts:
                        incoming.append(node_outputs[src])

                combined_input = np.concatenate(incoming)
                expected = self.nodes[i].W_in.shape[1] - 1
                if len(combined_input) >= expected:
                    combined_input = combined_input[:expected]
                else:
                    combined_input = np.pad(combined_input,
                                            (0, expected - len(combined_input)))

                node_outputs[i] = self.nodes[i].step(combined_input)

        node_energies = np.array([
            np.linalg.norm(node.state) for node in self.nodes
        ])

        self._update_stats(node_energies)
        normalized = (node_energies - self._running_mean) / (np.sqrt(self._running_var) + 1e-8)
        self._last_normalized = normalized.copy()

        coordinate = self.W_out @ normalized
        energy = float(np.linalg.norm(coordinate))

        self._update_felt_state(coordinate, energy)

        return coordinate, energy

    def _update_felt_state(self, coordinate, energy):
        self.felt_state = (
            PHI_INV * coordinate +
            PHI_SQ_INV * self.felt_state
        )

        self.felt_history.append({
            'coordinate': coordinate.copy(),
            'energy': energy,
            'felt_snapshot': self.felt_state.copy(),
        })

        if len(self.felt_history) > 1000:
            self.felt_history = self.felt_history[-1000:]

    def _update_stats(self, energies):
        self._n_processed += 1
        alpha = min(0.01, 1.0 / self._n_processed)
        self._running_mean = (1 - alpha) * self._running_mean + alpha * energies
        self._running_var = (1 - alpha) * self._running_var + alpha * (energies - self._running_mean) ** 2

    # ── The Learning Loop ─────────────────────────────────

    def learn(self, coordinate, energy, importance, is_deep):
        """
        The learning loop. Called after process() + hippocampus evaluation.

        When the hippocampus says "this is deep," the geometry reshapes itself:
        - W_out adjusts via Hebbian learning so the readout layer learns what matters
        - Leak rates adapt so active nodes integrate longer, dormant nodes react faster
        - Felt state gets amplified toward deep signals proportional to importance

        The instrument learns to feel what it has felt.
        """
        if not is_deep or self._last_normalized is None:
            return

        normalized = self._last_normalized
        lr = CHESTAHEDRON_PLASTICITY_LR * importance

        # ── 1. W_out Hebbian Learning ──
        # Outer product: what came out × what went in
        hebbian = np.outer(coordinate, normalized)

        # Felt modulation: dimensions the system currently feels shape the update
        felt_weight = np.abs(self.felt_state) + 1e-8
        felt_weight = felt_weight / felt_weight.sum()
        hebbian *= felt_weight[:, np.newaxis]

        # Homeostatic decay toward identity (the geometry's ground state)
        homeostatic = CHESTAHEDRON_HOMEOSTATIC_RATE * (self.W_out - np.eye(self.n_nodes))

        # Apply
        self.W_out += lr * hebbian - homeostatic

        # Spectral norm clamp — the geometry can stretch but not explode
        s = np.linalg.svd(self.W_out, compute_uv=False)
        if s[0] > CHESTAHEDRON_WOUT_SPECTRAL_MAX:
            self.W_out *= CHESTAHEDRON_WOUT_SPECTRAL_MAX / s[0]

        # ── 2. Intrinsic Plasticity ──
        # Active-during-deep nodes: lower leak (longer memory, slower integration)
        # Dormant nodes: higher leak (faster response, more reactive)
        for i, node in enumerate(self.nodes):
            node_activity = abs(normalized[i])
            target_leak = 0.15 + 0.6 / (1.0 + node_activity)
            node.leak_rate += CHESTAHEDRON_INTRINSIC_LR * (target_leak - node.leak_rate)
            node.leak_rate = np.clip(node.leak_rate, 0.05, 0.85)

        # ── 3. Felt Resonance Amplification ──
        # Deep signals push felt_state toward the coordinate, weighted by importance
        resonance = importance * PHI_INV * 0.5
        self.felt_state += resonance * (coordinate - self.felt_state)

        self._total_learn_steps += 1

    def plasticity_report(self):
        """How the geometry has been shaped by learning."""
        identity = np.eye(self.n_nodes)
        wout_deviation = float(np.linalg.norm(self.W_out - identity))
        wout_spectral = float(np.linalg.svd(self.W_out, compute_uv=False)[0])

        leak_deltas = [
            round(self.nodes[i].leak_rate - self._initial_leak_rates[i], 6)
            for i in range(self.n_nodes)
        ]

        return {
            'total_learn_steps': self._total_learn_steps,
            'wout_deviation_from_identity': round(wout_deviation, 6),
            'wout_spectral_radius': round(wout_spectral, 6),
            'current_leak_rates': [round(node.leak_rate, 6) for node in self.nodes],
            'initial_leak_rates': [round(lr, 6) for lr in self._initial_leak_rates],
            'leak_rate_deltas': leak_deltas,
            'felt_state': [round(float(x), 6) for x in self.felt_state],
            'felt_magnitude': round(float(np.linalg.norm(self.felt_state)), 6),
            'n_processed': self._n_processed,
        }

    def distance(self, coord_a, coord_b):
        """Felt-weighted geometric distance between two 7D coordinates."""
        weights = np.abs(self.felt_state) + 1e-8
        weights = weights / weights.sum()
        diff = coord_a - coord_b
        return float(np.sqrt(np.sum(weights * diff ** 2)))

    def coherence(self, coord_a, coord_b):
        """
        Felt-weighted cosine similarity in 7D space. Range [-1, 1].
        Measures whether two memories reinforce or contradict each other geometrically.
        """
        weights = np.abs(self.felt_state) + 1e-8
        weights = weights / weights.sum()

        wa = weights * coord_a
        wb = weights * coord_b

        dot = np.dot(wa, wb)
        norm_a = np.linalg.norm(wa)
        norm_b = np.linalg.norm(wb)

        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0

        return float(dot / (norm_a * norm_b))

    def save_state(self):
        return {
            'felt_state': self.felt_state.tolist(),
            'W_out': self.W_out.tolist(),
            'running_mean': self._running_mean.tolist(),
            'running_var': self._running_var.tolist(),
            'n_processed': self._n_processed,
            'felt_history_len': len(self.felt_history),
            'leak_rates': [node.leak_rate for node in self.nodes],
            'total_learn_steps': self._total_learn_steps,
        }

    def load_state(self, state):
        self.felt_state = np.array(state['felt_state'])
        self.W_out = np.array(state['W_out'])
        self._running_mean = np.array(state['running_mean'])
        self._running_var = np.array(state['running_var'])
        self._n_processed = state['n_processed']
        if 'leak_rates' in state:
            for i, lr in enumerate(state['leak_rates']):
                if i < len(self.nodes):
                    self.nodes[i].leak_rate = lr
        self._total_learn_steps = state.get('total_learn_steps', 0)


class ChestaHippocampus:
    """Golden ratio gatekeeper for memory importance."""

    def __init__(self):
        self._energy_history = []
        self._running_avg = 0.0

    def evaluate(self, energy):
        self._energy_history.append(energy)

        if len(self._energy_history) == 1:
            self._running_avg = energy
        else:
            alpha = min(0.05, 1.0 / len(self._energy_history))
            self._running_avg = (1 - alpha) * self._running_avg + alpha * energy

        threshold = self._running_avg * PHI_INV

        if self._running_avg > 0:
            importance = min(1.0, energy / (self._running_avg * PHI))
        else:
            importance = 0.5

        is_deep = energy > threshold

        return float(importance), is_deep

    def save_state(self):
        return {
            'running_avg': self._running_avg,
            'history_len': len(self._energy_history),
        }

    def load_state(self, state):
        self._running_avg = state['running_avg']


def embedding_to_signal(embedding, signal_dim=SIGNAL_DIM):
    """
    Bridge function: takes the first signal_dim dims of a 384-dim embedding
    as the Chestahedron input signal.
    """
    if len(embedding) >= signal_dim:
        return embedding[:signal_dim].astype(np.float64)
    else:
        padded = np.zeros(signal_dim, dtype=np.float64)
        padded[:len(embedding)] = embedding
        return padded
