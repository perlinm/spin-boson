import functools
from typing import Sequence

import numpy as np
import qutip
import scipy

DEFAULT_INTEGRATION_METHOD = "qutip"
DEFAULT_RTOL = 1e-12
DEFAULT_ATOL = 1e-12
DEFAULT_DIFF_STEP = 1e-4

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
# generic simulation methods


@functools.cache
def get_hamiltonian(num_spins: int, splitting: float, coupling: float) -> qutip.Qobj:
    spin_term = splitting * collective_Sz(num_spins)
    resonator_term = splitting * resonator_num_op(num_spins)
    coupling_op = resonator_lower(num_spins) * collective_raise(num_spins)
    coupling_term = coupling * (coupling_op + coupling_op.dag())
    return spin_term + resonator_term + coupling_term


@functools.cache
def get_jump_ops(num_spins: int, decay_res: float, decay_spin: float) -> list[qutip.Qobj]:
    ops = [np.sqrt(decay_spin) * qubit_lower(num_spins, ss) for ss in range(num_spins)]
    ops.append(np.sqrt(decay_res) * resonator_lower(num_spins))
    return ops


@functools.cache
def get_identity_matrix(dim: int) -> scipy.sparse.spmatrix:
    return scipy.sparse.eye(dim)


def to_adjoint_rep(matrix: scipy.sparse.spmatrix) -> scipy.sparse.spmatrix:
    """Construct the adjoint representation of a matrix (in the Lie algebra sense)."""
    iden = get_identity_matrix(matrix.shape[0])
    return scipy.sparse.kron(matrix, iden) - scipy.sparse.kron(iden, matrix.T)


def to_dissipation_generator(jump_op: scipy.sparse.spmatrix) -> scipy.sparse.spmatrix:
    """
    Convert a jump operator 'J' into a generator of time evolution for a density matrix 'rho', such
    that the generator corresponding to 'J' acts on the vectorized version of 'rho' as
        J_generator @ rho_vec ~= J rho J^dag + 1/2 [J^dag J, rho]_+,
    where '[A, B]_+ = A B + B A', and '~=' denotes equality up to reshaping an array.
    """
    direct_term = scipy.sparse.kron(jump_op, jump_op.conj())
    identity = get_identity_matrix(jump_op.shape[0])
    op_JJ = jump_op.conj().T @ jump_op
    recycling_term = scipy.sparse.kron(op_JJ, identity) + scipy.sparse.kron(identity, op_JJ.T)
    return direct_term - recycling_term / 2


@functools.cache
def get_hamiltonian_superop(
    num_spins: int, splitting: float, coupling: float
) -> scipy.sparse.spmatrix:
    return to_adjoint_rep(get_hamiltonian(num_spins, splitting, coupling).data)


@functools.cache
def get_jump_superop(num_spins: int, decay_res: float, decay_spin: float) -> scipy.sparse.spmatrix:
    return sum(
        to_dissipation_generator(jump_op.data)
        for jump_op in get_jump_ops(num_spins, decay_res, decay_spin)
    )


def time_deriv(time: float, state: np.ndarray, generator: scipy.sparse.spmatrix) -> np.ndarray:
    return generator @ state


def get_states(
    times: Sequence[float],
    initial_state: qutip.Qobj,
    hamiltonian: qutip.Qobj,
    jump_ops: Sequence[qutip.Qobj],
    method: str = DEFAULT_INTEGRATION_METHOD,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> np.ndarray:
    dim = initial_state.shape[0]
    final_shape = (len(times), dim, dim)
    if method == "qutip":
        options = qutip.Options(rtol=rtol, atol=atol)
        result = qutip.mesolve(hamiltonian, initial_state, times, jump_ops, options=options)
        states = np.array([state.data.todense().ravel() for state in result.states])
        return states.reshape(final_shape)

    # construct initial state as a vectorized density matrix
    initial_state_array = initial_state.data.todense()
    initial_state_matrix = np.outer(initial_state_array, initial_state_array.conj())
    initial_state_matrix_vec = initial_state_matrix.astype(complex).ravel()

    # construct the generator of time evolution
    ham_superop = to_adjoint_rep(hamiltonian.data)
    jump_superop = sum(to_dissipation_generator(jump_op.data) for jump_op in jump_ops)
    generator = ham_superop + jump_superop

    # numerically integrate the initial state
    solution = scipy.integrate.solve_ivp(
        time_deriv,
        (times[0], times[-1]),
        initial_state_matrix_vec,
        t_eval=times,
        method=method,
        args=(generator,),
        rtol=rtol,
        atol=atol,
    )
    return solution.y.T.reshape(final_shape)


################################################################################
# Fisher info calculation


def get_QFI(state: np.ndarray, state_diff: np.ndarray, tol: float = DEFAULT_ATOL):
    vals, vecs = np.linalg.eigh(state)

    # numerators and denominators
    nums = 2 * abs(vecs.conj().T @ state_diff @ vecs) ** 2
    dens = vals[:, np.newaxis] + vals[np.newaxis, :]  # matrix M[i, j] = w[i] + w[j]

    include = ~np.isclose(dens, 0, atol=tol)  # matrix of booleans (True/False)
    return (nums[include] / dens[include]).sum()


def get_QFI_vals(
    times: Sequence[float],
    num_spins: int,
    splitting: float,
    coupling: float,
    decay_res: float,
    decay_spin: float,
    initial_state: qutip.Qobj,
    method: str = DEFAULT_INTEGRATION_METHOD,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    diff_step: float = DEFAULT_DIFF_STEP,
):
    hamiltonian_p = get_hamiltonian(num_spins, splitting, coupling + diff_step / 2)
    hamiltonian_m = get_hamiltonian(num_spins, splitting, coupling - diff_step / 2)
    jump_ops = get_jump_ops(num_spins, decay_res, decay_spin)

    def _get_states(hamiltonian: qutip.Qobj) -> np.ndarray:
        return get_states(times, initial_state, hamiltonian, jump_ops, method, rtol, atol)

    states_p = _get_states(hamiltonian_p)
    states_m = _get_states(hamiltonian_m)

    # compute the QFI
    vals_QFI = np.zeros(len(times))
    vals_QFI_SA = np.zeros(len(times))
    for tt, (state_p, state_m) in enumerate(zip(states_p, states_m)):
        state_avg = (state_p + state_m) / 2
        state_diff = (state_p - state_m) / diff_step

        vals_QFI[tt] = get_QFI(state_avg, state_diff)
        vals_QFI_SA[tt] = np.real(1 - state_avg[0, 0]) * vals_QFI[tt]

    return times, vals_QFI, vals_QFI_SA
