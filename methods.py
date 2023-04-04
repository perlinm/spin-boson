import functools
from typing import Any, Callable, Optional, Sequence, TypeVar

import numpy as np
import qutip
import scipy

DEFAULT_INTEGRATION_METHOD = "qutip"
DEFAULT_DIFF_STEP = 1e-4  # step size for finite-difference derivative
DEFAULT_RTOL = 1e-10  # relative/absolute error tolerance for numerical intgeration
DEFAULT_ATOL = 1e-10

# Set heuristic for identifying the numerical cutoff for eigenvalues of a density matrix.
# Eigenvalues with magnitude < `abs(most_negative_eigenvalue) * DEFAULT_ETOL_SCALE` get set to 0.
DEFAULT_ETOL_SCALE = 10  # heuristic for identifying


ReturnType = TypeVar("ReturnType")


def _with_default_boson_dim(spin_func: Callable[..., ReturnType]) -> Callable[..., ReturnType]:
    """Infer a default boson dimension."""

    def get_state(num_spins: int, *args: Any, boson_dim: Optional[int] = None) -> ReturnType:
        if boson_dim is None:
            boson_dim = num_spins + 1
        return spin_func(num_spins, *args, boson_dim=boson_dim)

    return get_state


################################################################################
# spin operator definitions


def act_on(op: qutip.Qobj, target_index: int, num_spins: int) -> qutip.Qobj:
    ops = [qutip.qeye(2)] * num_spins
    ops[target_index] = op
    return qutip.tensor(*ops)


def collective_qubit_op(qubit_op: qutip.Qobj, num_spins: int) -> qutip.Qobj:
    return sum(act_on(qubit_op, ss, num_spins) for ss in range(num_spins))


def qubit_lower(num_spins: int, target_index: int) -> qutip.Qobj:
    return act_on(qutip.sigmam(), target_index, num_spins)


def collective_lower(num_spins: int) -> qutip.Qobj:
    return collective_qubit_op(qutip.sigmam(), num_spins)


def collective_raise(num_spins: int) -> qutip.Qobj:
    return collective_lower(num_spins).dag()


def collective_Sz(num_spins: int) -> qutip.Qobj:
    return collective_qubit_op(qutip.sigmaz(), num_spins) / 2


################################################################################
# state definitions


def get_spin_vacuum_state(num_spins: int) -> qutip.Qobj:
    states = [qutip.fock(2, 1)] * num_spins
    return qutip.tensor(*states)


@_with_default_boson_dim
def get_vacuum_state(num_spins: int, boson_dim: int) -> qutip.Qobj:
    return qutip.tensor(get_spin_vacuum_state(num_spins), qutip.fock(boson_dim, 0))


@_with_default_boson_dim
def get_dicke_state(num_spins: int, num_excitations: int, boson_dim: int) -> qutip.Qobj:
    spin_state = get_spin_vacuum_state(num_spins)
    collective_Sp = collective_raise(num_spins)
    for _ in range(num_excitations):
        spin_state = collective_Sp * spin_state
    spin_state = spin_state / np.linalg.norm(spin_state)
    return qutip.tensor(spin_state, qutip.fock(boson_dim, 0))


@_with_default_boson_dim
def get_ghz_state(num_spins: int, boson_dim: int) -> qutip.Qobj:
    return qutip.tensor(qutip.ghz_state(num_spins), qutip.fock(boson_dim, 0))


################################################################################
# generic simulation methods


@functools.cache
@_with_default_boson_dim
def get_hamiltonian(
    num_spins: int, splitting: float, coupling: float, boson_dim: int
) -> qutip.Qobj:
    spin_term = qutip.tensor(collective_Sz(num_spins), qutip.qeye(boson_dim))
    resonator_term = qutip.tensor(*[qutip.qeye(2)] * num_spins, qutip.num(boson_dim))
    coupling_op = qutip.tensor(collective_raise(num_spins), qutip.destroy(boson_dim))
    coupling_term = coupling_op + coupling_op.dag()
    return splitting * (spin_term + resonator_term) + coupling * coupling_term


@functools.cache
@_with_default_boson_dim
def get_jump_ops(
    num_spins: int, decay_res: float, decay_spin: float, boson_dim: int
) -> list[qutip.Qobj]:
    qubit_ops = [
        np.sqrt(decay_spin) * qutip.tensor(qubit_lower(num_spins, ss), qutip.qeye(boson_dim))
        for ss in range(num_spins)
    ]
    lower_res = qutip.tensor(*[qutip.qeye(2)] * num_spins, qutip.destroy(boson_dim))
    return qubit_ops + [np.sqrt(decay_res) * lower_res]


@functools.cache
def get_identity_matrix(dim: int) -> scipy.sparse.spmatrix:
    return scipy.sparse.identity(dim)


def to_adjoint_rep(matrix: scipy.sparse.spmatrix) -> scipy.sparse.spmatrix:
    """Construct the adjoint representation of a matrix (in the Lie algebra sense)."""
    iden = get_identity_matrix(matrix.shape[0])
    return scipy.sparse.kron(matrix, iden) - scipy.sparse.kron(iden, matrix.T)


def to_dissipation_generator(jump_op: scipy.sparse.spmatrix) -> scipy.sparse.spmatrix:
    """
    Convert a jump operator 'J' into a generator of time evolution for a density matrix 'rho', such
    that the generator corresponding to 'J' acts on the vectorized version of 'rho' as
        J_generator @ rho_vec ~= J rho J^dag - 1/2 [J^dag J, rho]_+,
    where '[A, B]_+ = A B + B A', and '~=' denotes equality up to reshaping an array.
    """
    identity = get_identity_matrix(jump_op.shape[0])
    direct_term = scipy.sparse.kron(jump_op, jump_op.conj())
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


def get_states(
    times: Sequence[float] | np.ndarray,
    initial_state: qutip.Qobj,
    hamiltonian: qutip.Qobj,
    jump_ops: Sequence[qutip.Qobj] = (),
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
        lambda time, state: generator @ state,
        (times[0], times[-1]),
        initial_state_matrix_vec,
        t_eval=times,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    return solution.y.T.reshape(final_shape)


################################################################################
# Fisher info calculation


def get_QFI(
    state: np.ndarray, state_diff: np.ndarray, etol_scale: float = DEFAULT_ETOL_SCALE
) -> float:
    vals, vecs = np.linalg.eigh(state)

    # identify numerical cutoff for small eigenvalues, below which they get set to 0
    etol = etol_scale * abs(min(vals))
    vals[abs(vals) < etol] = 0

    # numerators and denominators
    nums = abs(vecs.conj().T @ state_diff @ vecs) ** 2
    dens = vals[:, np.newaxis] + vals[np.newaxis, :]  # matrix M[i, j] = w[i] + w[j]

    # matrix of booleans (True/False) to ignore zero eigenvalues
    include = ~np.isclose(dens, 0)  # matrix of booleans (True/False)

    return 2 * (nums[include] / dens[include]).sum()


def get_QFI_vals(
    times: Sequence[float] | np.ndarray,
    num_spins: int,
    splitting: float,
    coupling: float,
    decay_res: float,
    decay_spin: float,
    initial_state: qutip.Qobj,
    method: str = DEFAULT_INTEGRATION_METHOD,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    etol_scale: float = DEFAULT_ETOL_SCALE,
    diff_step: float = DEFAULT_DIFF_STEP,
) -> tuple[np.ndarray, np.ndarray]:
    boson_dim = initial_state.shape[0] // 2**num_spins
    hamiltonian_p = get_hamiltonian(
        num_spins, splitting, coupling + diff_step / 2, boson_dim=boson_dim
    )
    hamiltonian_m = get_hamiltonian(
        num_spins, splitting, coupling - diff_step / 2, boson_dim=boson_dim
    )
    jump_ops = get_jump_ops(num_spins, decay_res, decay_spin, boson_dim=boson_dim)

    def _get_states(hamiltonian: qutip.Qobj) -> np.ndarray:
        return get_states(times, initial_state, hamiltonian, jump_ops, method, rtol, atol)

    states_p = _get_states(hamiltonian_p)
    states_m = _get_states(hamiltonian_m)

    # compute the QFI
    vals_QFI = np.zeros(len(times))
    vals_QFI_SA = np.zeros(len(times))
    vacuum_state = (1,) * num_spins + (0,)
    dims = (2,) * num_spins + (boson_dim,)
    vacuum_index = (np.ravel_multi_index(vacuum_state, dims),) * 2
    for tt, (state_p, state_m) in enumerate(zip(states_p, states_m)):
        state_avg = (state_p + state_m) / 2
        state_diff = (state_p - state_m) / diff_step

        vals_QFI[tt] = get_QFI(state_avg, state_diff, etol_scale)
        vals_QFI_SA[tt] = vals_QFI[tt] * (1 - state_avg[vacuum_index].real)

    return vals_QFI, vals_QFI_SA
