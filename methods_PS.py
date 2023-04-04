from typing import Any, Callable, Optional, Sequence, TypeVar

import numpy as np
import scipy

import spin_ops

DEFAULT_INTEGRATION_METHOD = "DOP853"
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
# operator definitions


def get_boson_num_op(dim: int) -> scipy.sparse.spmatrix:
    shape = (dim, dim)
    vals = range(dim)
    return scipy.sparse.dia_matrix(([vals], [0]), shape)


def get_boson_lower_op(dim: int) -> scipy.sparse.spmatrix:
    shape = (dim, dim)
    vals = [np.sqrt(val) for val in range(dim)]
    return scipy.sparse.dia_matrix(([vals], [1]), shape)


def get_boson_state(dim: int, index: int = 0) -> np.ndarray:
    state = np.zeros(dim)
    state[index] = 1
    return state


@_with_default_boson_dim
def get_hamiltonian_generator(
    num_spins: int,
    splitting: float,
    coupling: float,
    *,
    boson_dim: int,
) -> scipy.sparse.spmatrix:
    spin_op_dim = spin_ops.get_spin_op_dim(num_spins)
    spin_iden = scipy.sparse.identity(spin_op_dim)
    boson_iden = scipy.sparse.identity(boson_dim)

    spin_term_L = spin_ops.get_Sz_L(num_spins)
    spin_term = scipy.sparse.kron(
        spin_term_L - spin_ops.get_dual(spin_term_L),
        scipy.sparse.kron(boson_iden, boson_iden),
    )

    op_num = get_boson_num_op(boson_dim)
    cavity_term = scipy.sparse.kron(
        spin_iden, scipy.sparse.kron(op_num, boson_iden) - scipy.sparse.kron(boson_iden, op_num)
    )

    op_Sp_L = spin_ops.get_Sp_L(num_spins)
    op_lower = get_boson_lower_op(boson_dim)
    coupling_op_L = scipy.sparse.kron(
        op_Sp_L,
        scipy.sparse.kron(op_lower, boson_iden),
    )
    coupling_op_R = scipy.sparse.kron(
        spin_ops.get_dual(op_Sp_L),
        scipy.sparse.kron(boson_iden, op_lower.T),
    )
    coupling_op = coupling_op_L - coupling_op_R
    coupling_term = coupling_op + coupling_op.T

    hamiltonian_bracket = splitting * (spin_term + cavity_term) + coupling * coupling_term
    return -1j * hamiltonian_bracket


@_with_default_boson_dim
def get_dissipator(
    num_spins: int,
    decay_res: float,
    decay_spin: float,
    *,
    boson_dim: int,
) -> scipy.sparse.spmatrix:
    spin_op_dim = spin_ops.get_spin_op_dim(num_spins)
    dissipator_res = scipy.sparse.kron(
        scipy.sparse.identity(spin_op_dim),
        to_dissipation_generator(get_boson_lower_op(boson_dim)),
    )
    dissipator_spin = scipy.sparse.kron(
        spin_ops.get_local_dissipator(num_spins, "-"),
        scipy.sparse.identity(boson_dim**2),
    )
    return decay_res * dissipator_res + decay_spin * dissipator_spin


def to_dissipation_generator(jump_op: scipy.sparse.spmatrix) -> scipy.sparse.spmatrix:
    """
    Convert a jump operator 'J' into a generator of time evolution for a density matrix 'rho', such
    that the generator corresponding to 'J' acts on the vectorized version of 'rho' as
        J_generator @ rho_vec ~= J rho J^dag - 1/2 [J^dag J, rho]_+,
    where '[A, B]_+ = A B + B A', and '~=' denotes equality up to reshaping an array.
    """
    identity = scipy.sparse.identity(jump_op.shape[0])
    direct_term = scipy.sparse.kron(jump_op, jump_op.conj())
    op_JJ = jump_op.conj().T @ jump_op
    recycling_term = scipy.sparse.kron(op_JJ, identity) + scipy.sparse.kron(identity, op_JJ.T)
    return direct_term - recycling_term / 2


################################################################################
# state definitions


def _with_boson_vacuum(get_spin_state: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    """Turn spin-state constructors into spin-boson-state constructors."""

    @_with_default_boson_dim
    def get_state(num_spins: int, *args: Any, boson_dim: int):
        boson_vacuum = np.zeros(boson_dim**2)
        boson_vacuum[0] = 1
        return np.kron(get_spin_state(num_spins, *args), boson_vacuum)

    return get_state


get_vacuum_state = _with_boson_vacuum(spin_ops.get_vacuum_state)
get_dicke_state = _with_boson_vacuum(spin_ops.get_dicke_state)
get_ghz_state = _with_boson_vacuum(spin_ops.get_ghz_state)


################################################################################
# Fisher info calculation


def get_states(
    times: Sequence[float] | np.ndarray,
    initial_state: np.ndarray,
    generator: scipy.sparse.spmatrix,
    method: str = DEFAULT_INTEGRATION_METHOD,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> np.ndarray:
    solution = scipy.integrate.solve_ivp(
        lambda time, state: generator @ state,
        (times[0], times[-1]),
        initial_state.ravel().astype(complex),
        t_eval=times,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    return solution.y.T


def get_QFI(
    state: np.ndarray, state_diff: np.ndarray, etol_scale: float = DEFAULT_ETOL_SCALE
) -> float:
    """Compute the QFI from a state (density matrix) and its derivative."""
    # collect data from each block of the density matrix
    block_nums = []  # numerators in contributions to the QFI
    block_vals = []  # eigenvalues of the state

    for block, block_diff in zip(
        spin_ops.get_spin_blocks(state),
        spin_ops.get_spin_blocks(state_diff),
    ):
        vals, vecs = np.linalg.eigh(block)
        block_vals.append(vals)
        block_nums.append(abs(vecs.conj().T @ block_diff @ vecs) ** 2)

    # identify numerical cutoff for small eigenvalues, below which they get set to 0
    etol = etol_scale * abs(min(val for vals in block_vals for val in vals))

    val_QFI = 0
    for vals, nums in zip(block_vals, block_nums):
        vals[abs(vals) < etol] = 0
        dens = vals[:, np.newaxis] + vals[np.newaxis, :]  # matrix M[i, j] = w[i] + w[j]

        # matrix of booleans (True/False) to ignore zero eigenvalues
        include = ~np.isclose(dens, 0)  # matrix of booleans (True/False)

        # add contribution to the QFI from this block of the density matrix
        val_QFI += (nums[include] / dens[include]).sum()

    return 2 * val_QFI


def get_QFI_vals(
    times: Sequence[float] | np.ndarray,
    num_spins: int,
    splitting: float,
    coupling: float,
    decay_res: float,
    decay_spin: float,
    initial_state: np.ndarray,
    method: str = DEFAULT_INTEGRATION_METHOD,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    etol_scale: float = DEFAULT_ETOL_SCALE,
    diff_step: float = DEFAULT_DIFF_STEP,
) -> tuple[np.ndarray, np.ndarray]:
    spin_op_dim = spin_ops.get_spin_op_dim(num_spins)
    boson_dim = int(np.round(np.sqrt(initial_state.size // spin_op_dim)))

    hamiltonian_p = get_hamiltonian_generator(
        num_spins, splitting, coupling + diff_step / 2, boson_dim=boson_dim
    )
    hamiltonian_m = get_hamiltonian_generator(
        num_spins, splitting, coupling - diff_step / 2, boson_dim=boson_dim
    )
    dissipator = get_dissipator(num_spins, decay_res, decay_spin, boson_dim=boson_dim)

    def _get_states(hamiltonian: scipy.sparse.spmatrix) -> np.ndarray:
        generator = hamiltonian + dissipator
        return get_states(times, initial_state, generator, method, rtol, atol)

    shape = (len(times), spin_op_dim, boson_dim, boson_dim)
    states_p = _get_states(hamiltonian_p).reshape(shape)
    states_m = _get_states(hamiltonian_m).reshape(shape)

    # compute the QFI
    vals_QFI = np.zeros(len(times))
    vals_QFI_SA = np.zeros(len(times))
    vacuum_index = (-((num_spins + 1) ** 2), 0, 0)
    for tt, (state_p, state_m) in enumerate(zip(states_p, states_m)):
        state_avg = (state_p + state_m) / 2
        state_diff = (state_p - state_m) / diff_step

        vals_QFI[tt] = get_QFI(state_avg, state_diff, etol_scale)
        vals_QFI_SA[tt] = vals_QFI[tt] * (1 - state_avg[vacuum_index].real)

    return vals_QFI, vals_QFI_SA
