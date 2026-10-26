#from qiskit import QuantumCircuit, execute, Aer
from qiskit import QuantumCircuit, transpile
#from qiskit_ibm_runtime.fake_provider import FakeAlgiers
from qiskit_aer import Aer

def append_zz_term(qc, q1, q2, gamma):
    qc.cx(q1, q2)
    qc.rz(2 * gamma, q2)
    qc.cx(q1, q2)


def get_cost_operator_circuit(G, gamma):
    N = G.number_of_nodes()
    qc = QuantumCircuit(N, N)
    for i, j in G.edges():
        append_zz_term(qc, i, j, gamma)
    return qc


def append_x_term(qc, q1, beta):
    qc.rx(2 * beta, q1)


def get_mixer_operator_circuit(G, beta):
    N = G.number_of_nodes()
    qc = QuantumCircuit(N, N)
    for n in G.nodes():
        append_x_term(qc, n, beta)
    return qc


def get_qaoa_circuit(G, beta, gamma):
    assert len(beta) == len(gamma)
    p = len(beta)  # infering number of QAOA steps from the parameters passed
    N = G.number_of_nodes()
    qc = QuantumCircuit(N, N)
    # first, apply a layer of Hadamards
    qc.h(range(N))
    # second, apply p alternating operators
    for i in range(p):
        qc.append(get_cost_operator_circuit(G, gamma[i]), range(N), range(N))
        qc.append(get_mixer_operator_circuit(G, beta[i]), range(N), range(N))
    # finally, do not forget to measure the result!
    qc.barrier(range(N))
    qc.measure(range(N), range(N))
    return qc


def invert_counts(counts):
    return {k[::-1]: v for k, v in counts.items()}


def maxcut_obj(x, G):
    cut = 0
    for i, j in G.edges():
        if x[i] != x[j]:
            # the edge is cut
            cut -= 1
    return cut


def return_objective_distribution(counts, G):
    objective_dict = dict()
    for meas, meas_count in counts.items():
        obj_for_meas = maxcut_obj(meas, G)
        if obj_for_meas in objective_dict:
            objective_dict[obj_for_meas] += meas_count
        else:
            objective_dict[obj_for_meas] = meas_count
    return objective_dict


def compute_maxcut_energy(counts, G):
    energy = 0
    total_counts = 0
    for meas, meas_count in counts.items():
        obj_for_meas = maxcut_obj(meas, G)
        energy += obj_for_meas * meas_count
        total_counts += meas_count
    return energy / total_counts

def compute_maxcut_variance(counts, G, mean):
    var = 0
    total_counts = 0
    for meas, meas_count in counts.items():
        obj_for_meas = maxcut_obj(meas, G)
        var += ((obj_for_meas - mean) ** 2) * meas_count
        total_counts += meas_count
    return var / (total_counts ** 2)


def Sampled_Simulator(G, p, theta, shots, seed=None):
    backend = Aer.get_backend("qasm_simulator")
    #backend = FakeAlgiers()
    beta = theta[:p]
    gamma = theta[p:]
    qc = get_qaoa_circuit(G, beta, gamma)
    if seed:
        #counts = execute(qc, backend, shots=shots).result().get_counts()
        new_circuit = transpile(qc, backend)
        counts = backend.run(new_circuit, shots=shots).result().get_counts()
    else:
        #counts = execute(qc, backend, seed_simulator=seed, shots=shots).result().get_counts()
        #new_circuit = transpile(qc, backend, seed_simulator=seed, shots=shots)
        #counts = backend.run(new_circuit).result().get_counts()
        new_circuit = transpile(qc, backend)
        counts = backend.run(new_circuit, seed_simulator=seed, shots=shots).result().get_counts()
    mean = compute_maxcut_energy(invert_counts(counts), G)
    var = compute_maxcut_variance(invert_counts(counts), G, mean)
    return mean, var

def Sampled_Simulator_dist(backend, G, p, theta, shots, seed=None):
    #backend = Aer.get_backend("qasm_simulator")
    #backend = FakeAlgiers()
    beta = theta[:p]
    gamma = theta[p:]
    qc = get_qaoa_circuit(G, beta, gamma)
    if seed:
        #counts = execute(qc, backend, shots=shots).result().get_counts()
        new_circuit = transpile(qc, backend)
        counts = backend.run(new_circuit, shots=shots).result().get_counts()
    else:
        #counts = execute(qc, backend, seed_simulator=seed, shots=shots).result().get_counts()
        #new_circuit = transpile(qc, backend, seed_simulator=seed, shots=shots)
        #counts = backend.run(new_circuit).result().get_counts()
        new_circuit = transpile(qc, backend)
        counts = backend.run(new_circuit, seed_simulator=seed, shots=shots).result().get_counts()
    mean = compute_maxcut_energy(invert_counts(counts), G)
    var = compute_maxcut_variance(invert_counts(counts), G, mean)
    obj_dict = return_objective_distribution(invert_counts(counts), G)
    return mean, var, obj_dict
