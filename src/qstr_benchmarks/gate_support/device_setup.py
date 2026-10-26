"""Two-qutrit device configuration adapted from the C3 two_qubits notebook.

Upstream: q-optimize/c3, commit 48b7917eb8d84e1471ff9fb911129c354e752dbf.
Copyright the C3 contributors; Apache-2.0, see third_party/gate/c3_toolset/LICENSE.
Changes: separate relaxation/dephasing, enable thermal state and readout errors,
use configurable propagation resolution, widen frequency bounds to +/-10 MHz.
"""


def build_setup(resolution):
    # System imports
    import copy

    # Building blocks
    import c3.generator.devices as devices

    # Libs and helpers
    import c3.libraries.chip as chip
    import c3.libraries.envelopes as envelopes
    import c3.libraries.hamiltonians as hamiltonians
    import c3.libraries.tasks as tasks
    import c3.signal.gates as gates
    import c3.signal.pulse as pulse
    import c3.utils.qt_utils as qt_utils
    import numpy as np
    import tensorflow as tf

    # Main C3 objects
    from c3.c3objs import Quantity as Qty
    from c3.experiment import Experiment as Exp
    from c3.generator.generator import Generator as Gnr
    from c3.model import Model as Mdl
    from c3.parametermap import ParameterMap as PMap

    class PaperModel(Mdl):
        def update_Lindbladians(self):
            self.col_ops = []
            for sub in self.subsystems.values():
                self.col_ops.append(
                    tf.cast(
                        sub.collapse_ops["t1"] / tf.sqrt(sub.params["t1"].get_value()),
                        tf.complex128,
                    )
                )
                self.col_ops.append(
                    tf.cast(
                        sub.collapse_ops["t2star"]
                        * tf.sqrt(0.5 / sub.params["t2star"].get_value()),
                        tf.complex128,
                    )
                )

    # Upstream notebook cell 5
    qubit_lvls = 3
    freq_q1 = 5e9
    anhar_q1 = -210e6
    t1_q1 = 27e-6
    t2star_q1 = 39e-6
    qubit_temp = 50e-3

    q1 = chip.Qubit(
        name="Q1",
        desc="Qubit 1",
        freq=Qty(value=freq_q1, min_val=4.995e9, max_val=5.005e9, unit="Hz 2pi"),
        anhar=Qty(value=anhar_q1, min_val=-380e6, max_val=-120e6, unit="Hz 2pi"),
        hilbert_dim=qubit_lvls,
        t1=Qty(value=t1_q1, min_val=1e-6, max_val=90e-6, unit="s"),
        t2star=Qty(value=t2star_q1, min_val=10e-6, max_val=90e-3, unit="s"),
        temp=Qty(value=qubit_temp, min_val=0.0, max_val=0.12, unit="K"),
    )

    # Upstream notebook cell 7
    freq_q2 = 5.6e9
    anhar_q2 = -240e6
    t1_q2 = 23e-6
    t2star_q2 = 31e-6
    q2 = chip.Qubit(
        name="Q2",
        desc="Qubit 2",
        freq=Qty(value=freq_q2, min_val=5.595e9, max_val=5.605e9, unit="Hz 2pi"),
        anhar=Qty(value=anhar_q2, min_val=-380e6, max_val=-120e6, unit="Hz 2pi"),
        hilbert_dim=qubit_lvls,
        t1=Qty(value=t1_q2, min_val=1e-6, max_val=90e-6, unit="s"),
        t2star=Qty(value=t2star_q2, min_val=10e-6, max_val=90e-6, unit="s"),
        temp=Qty(value=qubit_temp, min_val=0.0, max_val=0.12, unit="K"),
    )

    # Upstream notebook cell 9
    coupling_strength = 20e6
    q1q2 = chip.Coupling(
        name="Q1-Q2",
        desc="coupling",
        comment="Coupling qubit 1 to qubit 2",
        connected=["Q1", "Q2"],
        strength=Qty(value=coupling_strength, min_val=-1 * 1e3, max_val=200e6, unit="Hz 2pi"),
        hamiltonian_func=hamiltonians.int_XX,
    )

    # Upstream notebook cell 11
    drive = chip.Drive(
        name="d1",
        desc="Drive 1",
        comment="Drive line 1 on qubit 1",
        connected=["Q1"],
        hamiltonian_func=hamiltonians.x_drive,
    )
    drive2 = chip.Drive(
        name="d2",
        desc="Drive 2",
        comment="Drive line 2 on qubit 2",
        connected=["Q2"],
        hamiltonian_func=hamiltonians.x_drive,
    )

    # Upstream notebook cell 13
    m00_q1 = 0.97  # Prop to read qubit 1 state 0 as 0
    m01_q1 = 0.04  # Prop to read qubit 1 state 0 as 1
    m00_q2 = 0.96  # Prop to read qubit 2 state 0 as 0
    m01_q2 = 0.05  # Prop to read qubit 2 state 0 as 1
    one_zeros = np.array([0] * qubit_lvls)
    zero_ones = np.array([1] * qubit_lvls)
    one_zeros[0] = 1
    zero_ones[0] = 0
    val1 = one_zeros * m00_q1 + zero_ones * m01_q1
    val2 = one_zeros * m00_q2 + zero_ones * m01_q2
    min_val = one_zeros * 0.8 + zero_ones * 0.0
    max_val = one_zeros * 1.0 + zero_ones * 0.2
    confusion_row1 = Qty(value=val1, min_val=min_val, max_val=max_val, unit="")
    confusion_row2 = Qty(value=val2, min_val=min_val, max_val=max_val, unit="")
    conf_matrix = tasks.ConfusionMatrix(Q1=confusion_row1, Q2=confusion_row2)

    # Upstream notebook cell 15
    init_temp = 50e-3
    init_ground = tasks.InitialiseGround(
        init_temp=Qty(value=init_temp, min_val=-0.001, max_val=0.22, unit="K")
    )

    # Upstream notebook cell 17
    model = PaperModel([q1, q2], [drive, drive2, q1q2], [conf_matrix, init_ground])

    # Upstream notebook cell 19
    model.set_lindbladian(True)
    model.set_dressed(True)

    # Upstream notebook cell 21
    sim_res = resolution  # Resolution for numerical simulation
    awg_res = 2e9  # Realistic, limited resolution of an AWG
    # Upstream notebook cell 27
    generator = Gnr(
        devices={
            "LO": devices.LO(name="lo", resolution=sim_res, outputs=1),
            "AWG": devices.AWG(name="awg", resolution=awg_res, outputs=1),
            "DigitalToAnalog": devices.DigitalToAnalog(
                name="dac", resolution=sim_res, inputs=1, outputs=1
            ),
            "Mixer": devices.Mixer(name="mixer", inputs=2, outputs=1),
            "VoltsToHertz": devices.VoltsToHertz(
                name="v_to_hz",
                V_to_Hz=Qty(value=1e9, min_val=0.9e9, max_val=1.1e9, unit="Hz/V"),
                inputs=1,
                outputs=1,
            ),
        },
        chains={
            "d1": {
                "LO": [],
                "AWG": [],
                "DigitalToAnalog": ["AWG"],
                "Mixer": ["LO", "DigitalToAnalog"],
                "VoltsToHertz": ["Mixer"],
            },
            "d2": {
                "LO": [],
                "AWG": [],
                "DigitalToAnalog": ["AWG"],
                "Mixer": ["LO", "DigitalToAnalog"],
                "VoltsToHertz": ["Mixer"],
            },
        },
    )

    # Upstream notebook cell 31
    t_final = 7e-9  # Time for single qubit gates
    sideband = 50e6
    gauss_params_single = {
        "amp": Qty(value=0.5, min_val=0.2, max_val=0.6, unit="V"),
        "t_final": Qty(value=t_final, min_val=0.5 * t_final, max_val=1.5 * t_final, unit="s"),
        "sigma": Qty(value=t_final / 4, min_val=t_final / 8, max_val=t_final / 2, unit="s"),
        "xy_angle": Qty(value=0.0, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"),
        "freq_offset": Qty(
            value=-sideband - 3e6, min_val=-60 * 1e6, max_val=-40 * 1e6, unit="Hz 2pi"
        ),
        "delta": Qty(value=-1, min_val=-5, max_val=3, unit=""),
    }

    # Upstream notebook cell 33
    gauss_env_single = pulse.EnvelopeDrag(
        name="gauss",
        desc="Gaussian comp for single-qubit gates",
        params=gauss_params_single,
        shape=envelopes.gaussian_nonorm,
    )

    # Upstream notebook cell 35
    nodrive_env = pulse.Envelope(
        name="no_drive",
        params={
            "t_final": Qty(value=t_final, min_val=0.5 * t_final, max_val=1.5 * t_final, unit="s")
        },
        shape=envelopes.no_drive,
    )

    # Upstream notebook cell 37
    lo_freq_q1 = 5e9 + sideband
    carrier_parameters = {
        "freq": Qty(value=lo_freq_q1, min_val=4.5e9, max_val=6e9, unit="Hz 2pi"),
        "framechange": Qty(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit="rad"),
    }
    carr = pulse.Carrier(
        name="carrier", desc="Frequency of the local oscillator", params=carrier_parameters
    )

    # Upstream notebook cell 39
    lo_freq_q2 = 5.6e9 + sideband
    carr_2 = copy.deepcopy(carr)
    carr_2.params["freq"].set_value(lo_freq_q2)

    # Upstream notebook cell 41
    rx90p_q1 = gates.Instruction(
        name="rx90p", targets=[0], t_start=0.0, t_end=t_final, channels=["d1", "d2"]
    )
    rx90p_q2 = gates.Instruction(
        name="rx90p", targets=[1], t_start=0.0, t_end=t_final, channels=["d1", "d2"]
    )

    rx90p_q1.add_component(gauss_env_single, "d1")
    rx90p_q1.add_component(carr, "d1")

    rx90p_q2.add_component(copy.deepcopy(gauss_env_single), "d2")
    rx90p_q2.add_component(carr_2, "d2")

    # Upstream notebook cell 43
    rx90p_q1.add_component(nodrive_env, "d2")
    rx90p_q1.add_component(copy.deepcopy(carr_2), "d2")
    rx90p_q1.comps["d2"]["carrier"].params["framechange"].set_value(
        (-sideband * t_final) * 2 * np.pi % (2 * np.pi)
    )

    rx90p_q2.add_component(nodrive_env, "d1")
    rx90p_q2.add_component(copy.deepcopy(carr), "d1")
    rx90p_q2.comps["d1"]["carrier"].params["framechange"].set_value(
        (-sideband * t_final) * 2 * np.pi % (2 * np.pi)
    )

    # Upstream notebook cell 45
    ry90p_q1 = copy.deepcopy(rx90p_q1)
    ry90p_q1.name = "ry90p"
    rx90m_q1 = copy.deepcopy(rx90p_q1)
    rx90m_q1.name = "rx90m"
    ry90m_q1 = copy.deepcopy(rx90p_q1)
    ry90m_q1.name = "ry90m"
    ry90p_q1.comps["d1"]["gauss"].params["xy_angle"].set_value(0.5 * np.pi)
    rx90m_q1.comps["d1"]["gauss"].params["xy_angle"].set_value(np.pi)
    ry90m_q1.comps["d1"]["gauss"].params["xy_angle"].set_value(1.5 * np.pi)
    single_q_gates = [rx90p_q1, ry90p_q1, rx90m_q1, ry90m_q1]

    ry90p_q2 = copy.deepcopy(rx90p_q2)
    ry90p_q2.name = "ry90p"
    rx90m_q2 = copy.deepcopy(rx90p_q2)
    rx90m_q2.name = "rx90m"
    ry90m_q2 = copy.deepcopy(rx90p_q2)
    ry90m_q2.name = "ry90m"
    ry90p_q2.comps["d2"]["gauss"].params["xy_angle"].set_value(0.5 * np.pi)
    rx90m_q2.comps["d2"]["gauss"].params["xy_angle"].set_value(np.pi)
    ry90m_q2.comps["d2"]["gauss"].params["xy_angle"].set_value(1.5 * np.pi)
    single_q_gates.extend([rx90p_q2, ry90p_q2, rx90m_q2, ry90m_q2])

    # Upstream notebook cell 47
    parameter_map = PMap(instructions=single_q_gates, model=model, generator=generator)
    return {
        "generator": generator,
        "devices": devices,
        "Qty": Qty,
        "Gnr": Gnr,
        "PMap": PMap,
        "Exp": Exp,
        "model": model,
        "single_q_gates": single_q_gates,
        "qt_utils": qt_utils,
        "conf_matrix": conf_matrix,
        "parameter_map": parameter_map,
    }
