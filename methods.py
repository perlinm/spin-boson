import numpy as np
import qutip
import scipy

DEFAULT_TOLERANCE = 1e-8
DEFAULT_STEP = 1e-6


################################################################################
# operator definitions


def act_on(op: qutip.Qobj, target_index: int, num_spins: int) -> qutip.Qobj:
    ops = [qutip.qeye(2)] * num_spins + [qutip.qeye(num_spins + 1)]
    ops[target_index] = op
    return qutip.tensor(*ops)


def collective_qubit_op(qubit_op: qutip.Qobj, num_spins: int) -> qutip.Qobj:
    return sum(act_on(qubit_op, ss, num_spins) for ss in range(num_spins))


def qubit_lower(num_spins: int, target_index: int) -> qutip.Qobj:
    return act_on(qutip.destroy(2), target_index, num_spins)


def collective_lower(num_spins: int) -> qutip.Qobj:
    return collective_qubit_op(qutip.destroy(2), num_spins)


def collective_raise(num_spins: int) -> qutip.Qobj:
    return collective_qubit_op(qutip.destroy(2).dag(), num_spins)


def collective_Sz(num_spins: int) -> qutip.Qobj:
    return collective_qubit_op(qutip.sigmaz(), num_spins) / 2


def act_on_resonator(op: qutip.Qobj, num_spins: int) -> qutip.Qobj:
    return act_on(op, -1, num_spins)


def resonator_lower(num_spins: int) -> qutip.Qobj:
    return act_on_resonator(qutip.destroy(num_spins + 1), num_spins)


def resonator_num_op(num_spins: int) -> qutip.Qobj:
    return act_on_resonator(qutip.num(num_spins + 1), num_spins)


def get_hamiltonian(splitting: float, coupling: float, num_spins: int) -> qutip.Qobj:
    spin_term = splitting * collective_Sz(num_spins)
    resonator_term = splitting * resonator_num_op(num_spins)
    coupling_op = resonator_lower(num_spins) * collective_raise(num_spins)
    coupling_term = coupling * (coupling_op + coupling_op.dag())
    return spin_term + resonator_term + coupling_term


################################################################################
# state definitions


def get_all_down_state(num_spins: int) -> qutip.Qobj:
    states = [qutip.fock(2, 0)] * num_spins + [qutip.fock(num_spins + 1, 0)]
    return qutip.tensor(*states)


def get_dicke_state(num_spins: int, num_excitations: int) -> qutip.Qobj:
    state = get_all_down_state(num_spins)
    for _ in range(num_excitations):
        state = collective_raise(num_spins) * state
    return state / np.linalg.norm(state)


def get_ghz_state(num_spins: int) -> qutip.Qobj:
    return qutip.tensor(qutip.ghz_state(num_spins), qutip.fock(num_spins + 1, 0))


################################################################################
# miscellaneous simulation options


def get_jump_ops(num_spins: int, decay_res: float, decay_spin: float) -> list[qutip.Qobj]:
    ops = [np.sqrt(decay_spin) * qubit_lower(num_spins, ss) for ss in range(num_spins)]
    ops.append(np.sqrt(decay_res) * resonator_lower(num_spins))
    return ops


def get_expectation_ops(num_spins: int) -> list[qutip.Qobj]:
    return [act_on(qutip.num(2), ss, num_spins) for ss in range(num_spins)]


################################################################################
# Fisher info calculation


def get_fisher_info(rho: scipy.sparse.spmatrix, rhoprime: scipy.sparse.spmatrix, tol: float = 1e-8):
    vals, vecs = np.linalg.eigh(rho)
    rhodg = vecs.conj().T @ np.array(rhoprime.data.todense()) @ vecs

    # numerators and denominators
    nums = 2 * abs(rhodg) ** 2
    dens = vals[:, np.newaxis] + vals[np.newaxis, :]  # matrix M[i, j] = w[i] + w[j]

    include = ~np.isclose(dens, 0, atol=tol)  # matrix of booleans (True/False)
    return (nums[include] / dens[include]).sum()


def get_fisher_vals(
    times: np.ndarray,
    num_spins: int,
    splitting: float,
    coupling: float,
    decay_res: float,
    decay_spin: float,
    initial_state: qutip.Qobj,
    rel_diff_step: float = DEFAULT_STEP,
    options: qutip.Options = qutip.Options(store_states=True),
):
    diff_step = coupling * rel_diff_step if coupling else rel_diff_step
    jump_ops = get_jump_ops(num_spins, decay_res, decay_spin)
    expect_ops = get_expectation_ops(num_spins)

    result = qutip.mesolve(
        get_hamiltonian(splitting, coupling, num_spins),
        initial_state,
        times,
        jump_ops,
        expect_ops,
        options=options,
    )
    result_displaced = qutip.mesolve(
        get_hamiltonian(splitting, coupling + diff_step, num_spins),
        initial_state,
        times,
        jump_ops,
        expect_ops,
        options=options,
    )

    fisher_vals = np.zeros(len(times))
    scaled_fisher_vals = np.zeros(len(times))

    for tt, rho in enumerate(result.states):
        rho_displaced = result_displaced.states[tt]
        rhoprime = (rho_displaced - rho) / diff_step
        fisher_vals[tt] = get_fisher_info(rho, rhoprime)
        scaled_fisher_vals[tt] = np.real(1 - rho[0, 0]) * fisher_vals[tt]

    return fisher_vals, scaled_fisher_vals
