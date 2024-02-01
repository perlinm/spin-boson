"""Methods for simulating a spin-boson system using a QuTiP backend."""
import functools
from typing import Any, Callable, Optional, Sequence, TypeVar

import numpy as np
import qutip
import scipy

DEFAULT_INTEGRATION_METHOD = "qutip"
DEFAULT_RTOL = 1e-10  # relative/absolute error tolerance for numerical intgeration
DEFAULT_ATOL = 1e-10
DEFAULT_DIFF_STEP = 1e-3  # step size for finite-difference derivative

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
    """Act with the given operator on the given spin (specified by index)."""
    ops = [qutip.qeye(2)] * num_spins
    ops[target_index] = op
    return qutip.tensor(*ops)


def collective_qubit_op(qubit_op: qutip.Qobj, num_spins: int) -> qutip.Qobj:
    """Convert a single-qubit operator into a collective operator.

    If `O` is a single-qubit operator, this method returns sum_j O_j."""
    return sum(act_on(qubit_op, ss, num_spins) for ss in range(num_spins))


def qubit_lower(num_spins: int, target_index: int) -> qutip.Qobj:
    """Construct the lowering operator for single spin."""
    return act_on(qutip.sigmam(), target_index, num_spins)


def collective_lower(num_spins: int) -> qutip.Qobj:
    """Construct the a collective spin-lowering operator."""
    return collective_qubit_op(qutip.sigmam(), num_spins)


def collective_raise(num_spins: int) -> qutip.Qobj:
    """Construct the a collective spin-raising operator."""
    return collective_lower(num_spins).dag()


def qubit_sz(num_spins: int, target_index: int) -> qutip.Qobj:
    """Construct the lowering operator for single spin."""
    return act_on(qutip.sigmaz(), target_index, num_spins) / 2


def collective_Sz(num_spins: int) -> qutip.Qobj:
    """Construct the a collective spin-z operator."""
    return collective_qubit_op(qutip.sigmaz(), num_spins) / 2


################################################################################
# state definitions


def get_spin_vacuum_state(num_spins: int) -> qutip.Qobj:
    """Construct the vacuum (all-spin-down) state for a collection of spins."""
    return qutip.ket("d" * num_spins)


@_with_default_boson_dim
def get_vacuum_state(num_spins: int, boson_dim: int) -> qutip.Qobj:
    """Construct the vacuum state of a spin-boson system."""
    return qutip.tensor(get_spin_vacuum_state(num_spins), qutip.fock(boson_dim, 0))


@_with_default_boson_dim
def get_dicke_state(num_spins: int, num_excitations: int, boson_dim: int) -> qutip.Qobj:
    """Construct a Dicke state with a boson vacuum."""
    spin_state = get_spin_vacuum_state(num_spins)
    collective_Sp = collective_raise(num_spins)
    for _ in range(num_excitations):
        spin_state = collective_Sp * spin_state
    spin_state = spin_state / np.linalg.norm(spin_state)
    return qutip.tensor(spin_state, qutip.fock(boson_dim, 0))


@_with_default_boson_dim
def get_ghz_state(num_spins: int, boson_dim: int) -> qutip.Qobj:
    """Construct a GHZ state with a boson vacuum."""
    return qutip.tensor(qutip.ghz_state(num_spins), qutip.fock(boson_dim, 0))


@_with_default_boson_dim
def get_state_X(num_spins: int, boson_dim: int) -> qutip.Qobj:
    """Construct an X-polarized spin state with a boson vacuum."""
    array = np.ones(2**num_spins) / np.sqrt(2**num_spins)
    state = qutip.Qobj(array, dims=[[2] * num_spins, [1] * num_spins])
    return qutip.tensor(state, qutip.fock(boson_dim, 0))


################################################################################
# generic simulation methods


@functools.cache
@_with_default_boson_dim
def get_hamiltonian(
    num_spins: int,
    spin_splitting: float,
    boson_splitting: float,
    coupling: float,
    *,
    boson_dim: int,
) -> qutip.Qobj:
    """
    Construct the Hamiltonian
    `spin_splitting * Sz + boson_splitting * N + coupling * (Sp a + Sm a^dag)`,
    where:
    - `Sz` is a spin-z operator for the spins
    - `Sm` and `Sp` are collective spin-lowering and spin-raising operators
    - `N` is the number operator for the bosonic mode
    - `a` and `a^dag` are lowering and raising operators for the bosonic mode
    - `spin_splitting`, `boson_splitting`, and `coupling` are scalars
    """
    spin_term = qutip.tensor(collective_Sz(num_spins), qutip.qeye(boson_dim))
    boson_term = qutip.tensor(*[qutip.qeye(2)] * num_spins, qutip.num(boson_dim))
    coupling_op = qutip.tensor(collective_raise(num_spins), qutip.destroy(boson_dim))
    coupling_term = coupling_op + coupling_op.dag()
    return spin_splitting * spin_term + boson_splitting * boson_term + coupling * coupling_term


@functools.cache
@_with_default_boson_dim
def get_jump_ops(
    num_spins: int, decay_res: float, decay_spin: float, dephasing: bool, boson_dim: int
) -> list[qutip.Qobj]:
    """Construct a list of jump operators corresponding to single-spin decay and boson decay."""
    if dephasing:
        # resonator and spin dephasing
        qubit_op_source = qubit_sz
        boson_op = qutip.num(boson_dim)
    else:
        # resonator and spin decay
        qubit_op_source = qubit_lower
        boson_op = qutip.destroy(boson_dim)

    qubit_ops = [
        np.sqrt(decay_spin) * qutip.tensor(qubit_op_source(num_spins, ss), qutip.qeye(boson_dim))
        for ss in range(num_spins)
    ]
    resonaor_op = qutip.tensor(*[qutip.qeye(2)] * num_spins, boson_op)
    return qubit_ops + [np.sqrt(decay_res) * resonaor_op]


@functools.cache
def get_identity_matrix(dim: int) -> scipy.sparse.spmatrix:
    """Construct an identity matrix."""
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
    num_spins: int,
    spin_splitting: float,
    boson_splitting: float,
    coupling: float,
) -> scipy.sparse.spmatrix:
    """Get the (Lie-algebraic) adjoint representation of a Hamiltonian."""
    return to_adjoint_rep(
        get_hamiltonian(num_spins, spin_splitting, boson_splitting, coupling).data
    )


@functools.cache
def get_jump_superop(num_spins: int, decay_res: float, decay_spin: float) -> scipy.sparse.spmatrix:
    """Get the superoperators that generate spin and boson decay."""
    return sum(
        to_dissipation_generator(jump_op.data)
        for jump_op in get_jump_ops(num_spins, decay_res, decay_spin)
    )


def get_states(
    times: np.ndarray,
    initial_state: qutip.Qobj,
    hamiltonian: qutip.Qobj,
    jump_ops: Sequence[qutip.Qobj] = (),
    method: str = DEFAULT_INTEGRATION_METHOD,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> np.ndarray:
    """For a given initial state, Hamiltonian, and jump operators, return states at later times."""
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
    """Compute the QFI from a state and its derivative w.r.t. the parameter to be estimated."""
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
    times: np.ndarray,
    num_spins: int,
    spin_splitting: float,
    boson_splitting: float,
    coupling: float,
    decay_res: float,
    decay_spin: float,
    dephasing: bool,
    initial_state: qutip.Qobj,
    method: str = DEFAULT_INTEGRATION_METHOD,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    etol_scale: float = DEFAULT_ETOL_SCALE,
    diff_step: float = DEFAULT_DIFF_STEP,
) -> np.ndarray:
    """Get the QFI over time for a spin-boson system defined by the provided arguments."""
    boson_dim = initial_state.shape[0] // 2**num_spins
    hamiltonian_p = get_hamiltonian(
        num_spins, spin_splitting, boson_splitting, coupling + diff_step / 2, boson_dim=boson_dim
    )
    hamiltonian_m = get_hamiltonian(
        num_spins, spin_splitting, boson_splitting, coupling - diff_step / 2, boson_dim=boson_dim
    )
    jump_ops = get_jump_ops(num_spins, decay_res, decay_spin, dephasing, boson_dim=boson_dim)

    def _get_states(hamiltonian: qutip.Qobj) -> np.ndarray:
        return get_states(times, initial_state, hamiltonian, jump_ops, method, rtol, atol)

    states_p = _get_states(hamiltonian_p)
    states_m = _get_states(hamiltonian_m)

    # compute the QFI
    vals_QFI = np.zeros(len(times))
    for tt, (state_p, state_m) in enumerate(zip(states_p, states_m)):
        state_avg = (state_p + state_m) / 2
        state_diff = (state_p - state_m) / diff_step
        vals_QFI[tt] = get_QFI(state_avg, state_diff, etol_scale)

    return vals_QFI


################################################################################
# Fisher info bound calculation
# WARNING: CURRENTLY BROKEN


def get_QFI_bound_vals(
    times: np.ndarray,
    num_spins: int,
    spin_splitting: float,
    boson_splitting: float,
    coupling: float,
    decay_res: float,
    decay_spin: float,
    initial_state: qutip.Qobj,
    method: str = DEFAULT_INTEGRATION_METHOD,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    etol_scale: float = DEFAULT_ETOL_SCALE,
    diff_step: float = DEFAULT_DIFF_STEP,
) -> np.ndarray:
    """Get the QFI over time for a spin-boson system defined by the provided arguments."""
    boson_dim = initial_state.shape[0] // 2**num_spins
    state_dims = [initial_state.dims[0]] * 2

    # compute time-evolved states
    hamiltonian = get_hamiltonian(
        num_spins, spin_splitting, boson_splitting, coupling, boson_dim=boson_dim
    )
    jump_ops = get_jump_ops(num_spins, decay_res, decay_spin, boson_dim=boson_dim)
    states = get_states(times, initial_state, hamiltonian, jump_ops, method, rtol, atol)

    # compute the generators of time evolution
    jump_ops_p = get_jump_ops(num_spins, decay_res, decay_spin + diff_step / 2, boson_dim=boson_dim)
    jump_ops_m = get_jump_ops(num_spins, decay_res, decay_spin - diff_step / 2, boson_dim=boson_dim)
    generator_p = qutip.liouvillian(hamiltonian, jump_ops_p)
    generator_m = qutip.liouvillian(hamiltonian, jump_ops_m)

    # compute kraus operators
    kraus_ops_p = [qutip.to_kraus((time * generator_p).expm()) for time in times]
    kraus_ops_m = [qutip.to_kraus((time * generator_m).expm()) for time in times]

    # compute bound at each time
    vals_bound = np.zeros(len(times))
    for tt, state in enumerate(states):
        op_A = op_B = 0
        for kraus_op_p, kraus_op_m in zip(kraus_ops_p[tt], kraus_ops_m[tt]):
            kraus_op = (kraus_op_p + kraus_op_m) / 2
            kraus_op_deriv = (kraus_op_p - kraus_op_m) / diff_step
            kraus_op_deriv_dag = kraus_op_deriv.dag()

            op_A += kraus_op_deriv_dag * kraus_op_deriv
            op_B += 1j * kraus_op_deriv_dag * kraus_op

        qutip_state = qutip.Qobj(state, dims=state_dims)
        term_A = qutip.expect(op_A, qutip_state)
        term_B = qutip.expect(op_B, qutip_state)
        vals_bound[tt] = (term_A - term_B**2).real

    return 4 * vals_bound
