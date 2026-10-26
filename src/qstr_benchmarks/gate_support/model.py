"""Two-qutrit Lindblad model and split propagation for gate calibration."""

import numpy as np

from .device_setup import build_setup


def build(target=0, resolution=800e9, destination=None):
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    ns = build_setup(resolution)
    generator = ns["generator"]
    devices, Qty = ns["devices"], ns["Qty"]
    response = devices.Response(
        name="resp",
        rise_time=Qty(value=0.3e-9, min_val=0.05e-9, max_val=0.6e-9, unit="s"),
        resolution=resolution,
        inputs=1,
        outputs=1,
    )
    # Include the finite-bandwidth Response component in each drive chain.
    all_devices = dict(generator.devices)
    all_devices["Response"] = response
    chains = {
        name: {
            "LO": [],
            "AWG": [],
            "DigitalToAnalog": ["AWG"],
            "Response": ["DigitalToAnalog"],
            "Mixer": ["LO", "Response"],
            "VoltsToHertz": ["Mixer"],
        }
        for name in ("d1", "d2")
    }
    generator = ns["Gnr"](devices=all_devices, chains=chains)
    pmap = ns["PMap"](instructions=ns["single_q_gates"], model=ns["model"], generator=generator)
    names = [f"{name}[{target}]" for name in ("rx90p", "ry90p", "rx90m", "ry90m")]
    line = f"d{target + 1}"
    optmap = [
        [(name, line, "gauss", param) for name in names]
        for param in ("amp", "delta", "freq_offset")
    ]
    pmap.set_opt_map(optmap)
    exp = ns["Exp"](pmap=pmap)
    exp.sim_res = resolution
    exp.set_opt_gates(names)
    ns.update(
        parameter_map=pmap,
        generator=generator,
        exp=exp,
        target=target,
        optmap=optmap,
        gate_names=names,
        resolution=resolution,
        dissipation_model="paper",
    )
    ns["initial_state"] = np.asarray(ns["model"].get_init_state()).reshape(-1)
    return ns


def reference_superoperators(ns, x):
    ns["parameter_map"].set_parameters_scaled(np.asarray(x, float))
    return {k: np.asarray(v) for k, v in ns["exp"].compute_propagators().items()}


class SplitPropagator:
    """Strang splitting with upstream signals/Hamiltonians/collapse operators.

    Each small Hamiltonian exponential is exact for the same PWC slice as C3.
    Constant dissipation is applied around groups of slices. This approximation
    is checked against full upstream Lindblad exponentials before production use.
    """

    def __init__(self, ns, split_ns=0.1, dissipation=True, parity=False, tensor_action=False):
        from scipy.linalg import expm

        self.ns, self.split_ns, self.dissipation = ns, float(split_ns), bool(dissipation)
        model = ns["model"]
        model.controllability = True
        h0, hc = model.get_Hamiltonians()
        self.h0 = np.asarray(h0)
        self.hc = {k: np.asarray(v) for k, v in hc.items()}
        self.dissipator = np.zeros((81, 81), complex)
        identity = np.eye(9)
        for collapse in model.get_Lindbladians():
            c = np.asarray(collapse)
            left = np.kron(c, identity)
            right = np.kron(identity, c.T)
            self.dissipator += left @ right.conj().T - 0.5 * (
                left.conj().T @ left + right @ right.conj().T
            )
        self.cache = {}
        self.expm = expm
        self.parity = bool(parity)
        self.tensor_action = bool(tensor_action)
        transform = np.asarray(model.transform)
        self.P = (
            transform.conj().T
            @ np.kron(np.diag([1.0, -1.0, 1.0]), np.diag([1.0, -1.0, 1.0]))
            @ transform
        )
        self.K = np.kron(self.P, self.P.conj())
        if self.parity and np.max(
            abs(self.K @ self.dissipator - self.dissipator @ self.K)
        ) > 1e-9 * max(1, np.max(abs(self.dissipator))):
            raise ValueError("Parity acceleration requires a parity-covariant dissipator")

    def __call__(self, x):
        ns = self.ns
        ns["parameter_map"].set_parameters_scaled(np.asarray(x, float))
        result = {}
        for name in ns["gate_names"]:
            if self.parity and name.startswith(("rx90m", "ry90m")):
                positive = name.replace("90m", "90p")
                result[name] = self.K @ result[positive] @ self.K.conj().T
                continue
            instr = ns["parameter_map"].instructions[name]
            signals = ns["generator"].generate_signals(instr)
            h = None
            ts = None
            for key, signal in signals.items():
                values = np.asarray(signal["values"]).reshape(-1)
                ts = np.asarray(signal["ts"]).reshape(-1)
                term = values[:, None, None] * self.hc[key]
                h = term if h is None else h + term
            h = h + self.h0
            dt = float(ts[1] - ts[0])
            energies, vectors = np.linalg.eigh(h)
            us = (vectors * np.exp(-1j * dt * energies)[:, None, :]) @ vectors.conj().transpose(
                0, 2, 1
            )
            width = max(1, int(round(self.split_ns * 1e-9 / dt))) if self.dissipation else len(us)
            channel = np.eye(81, dtype=complex)
            previous_duration = None
            for start in range(0, len(us), width):
                block = us[start : start + width]
                duration = len(block) * dt
                key = (len(block), dt)
                if key not in self.cache:
                    self.cache[key] = (
                        self.expm(self.dissipator * (duration / 2))
                        if self.dissipation
                        else np.eye(81)
                    )
                diss = self.cache[key]
                unitary = np.eye(9, dtype=complex)
                for u in block:
                    unitary = u @ unitary
                if self.tensor_action:
                    # D is fixed, so adjacent dissipative half steps merge.
                    if previous_duration is None:
                        channel = diss.copy()
                    else:
                        bridge_key = ("bridge", previous_duration, duration)
                        if bridge_key not in self.cache:
                            self.cache[bridge_key] = (
                                self.expm(self.dissipator * ((previous_duration + duration) / 2))
                                if self.dissipation
                                else np.eye(81)
                            )
                        channel = self.cache[bridge_key] @ channel
                    # Apply U rho U^dagger to the 81 input matrix units without
                    # multiplying the dense Kronecker superoperator.
                    states = channel.T.reshape(81, 9, 9)
                    states = unitary @ states @ unitary.conj().T
                    channel = states.reshape(81, 81).T
                    previous_duration = duration
                else:
                    super_u = np.kron(unitary, unitary.conj())
                    channel = diss @ super_u @ diss @ channel
            if self.tensor_action:
                channel = diss @ channel
            # Apply exactly the upstream Experiment's virtual frame correction.
            freqs, changes = {}, {}
            for line, controls in instr.comps.items():
                offset = 0.0
                for ctrl in controls.values():
                    if "freq_offset" in ctrl.params and ctrl.params["amp"] != 0.0:
                        offset = ctrl.params["freq_offset"].get_value()
                import tensorflow as tf

                freqs[line] = tf.cast(
                    controls["carrier"].params["freq"].get_value() + offset, tf.complex128
                )
                changes[line] = tf.cast(
                    controls["carrier"].params["framechange"].get_value(), tf.complex128
                )
            rotation = np.asarray(
                ns["model"].get_Frame_Rotation(
                    tf.constant(instr.t_end - instr.t_start, tf.complex128), freqs, changes
                )
            )
            result[name] = np.kron(rotation, rotation.conj()) @ channel
        return result


def circuits(ns, number, length, seed):
    state = np.random.get_state()
    try:
        np.random.seed(int(seed))
        seqs = ns["qt_utils"].single_length_RB(number, length, target=ns["target"])
    finally:
        np.random.set_state(state)
    return seqs


def circuit_probabilities(ns, operators, sequences):
    names = list(operators)
    lookup = {k: i for i, k in enumerate(names)}
    codes = np.full((len(sequences), max(map(len, sequences))), -1, dtype=np.int8)
    for i, seq in enumerate(sequences):
        codes[i, : len(seq)] = [lookup[k] for k in seq]
    initial = ns["initial_state"]
    states = np.tile(initial, (len(sequences), 1))
    for j in range(codes.shape[1]):
        for k, name in enumerate(names):
            select = np.flatnonzero(codes[:, j] == k)
            if len(select):
                states[select] = states[select] @ operators[name].T
    populations = states[:, np.arange(9) * 10].real
    # Upstream ConfusionMatrix is a tensor product of two 2x3 matrices.
    conf = np.ones((1, 1))
    for item in ns["conf_matrix"].params.values():
        row = np.asarray(item.get_value()).reshape(-1)
        conf = np.kron(conf, np.vstack((row, 1 - row)))
    measured = populations @ conf.T
    excited = [2, 3] if ns["target"] == 0 else [1, 3]
    failure = measured[:, excited].sum(axis=1)
    return failure, dict(
        max_trace_error=float(np.max(abs(populations.sum(axis=1) - 1))),
        minimum_population=float(populations.min()),
        max_readout_sum_error=float(np.max(abs(measured.sum(axis=1) - 1))),
    )
