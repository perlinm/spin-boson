"""Methods for simulating a spin-boson system with permutational symmetry."""
from typing import Any, Callable, Iterator, Optional, TypeVar

import numpy as np
import scipy

import spin_ops

DEFAULT_INTEGRATION_METHOD = "DOP853"
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
# operator definitions


def get_boson_num_op(dim: int) -> scipy.sparse.spmatrix:
    """Construct the number operator for a bosonic mode."""
    shape = (dim, dim)
    vals = range(dim)
    return scipy.sparse.dia_matrix(([vals], [0]), shape)


def get_boson_lower_op(dim: int) -> scipy.sparse.spmatrix:
    """Construct the lowering operator for a bosonic mode."""
    shape = (dim, dim)
    vals = [np.sqrt(val) for val in range(dim)]
    return scipy.sparse.dia_matrix(([vals], [1]), shape)


def get_boson_state(dim: int, index: int = 0) -> np.ndarray:
    """Construct the n-occupation state |n> for a bosonic mode."""
    state = np.zeros(dim)
    state[index] = 1
    return state


@_with_default_boson_dim
def get_hamiltonian_generator(
    num_spins: int,
    spin_splitting: float,
    boson_splitting: float,
    coupling: float,
    *,
    boson_dim: int,
) -> scipy.sparse.spmatrix:
    """Construct the generator of coherent time evolution for a vectorized density matrix.

    The corresponding Hamiltonian is
    `spin_splitting * Sz + boson_splitting * N + coupling * (Sp a + Sm a^dag)`,
    where:
    - `Sz` is a spin-z operator for the spins
    - `Sm` and `Sp` are collective spin-lowering and spin-raising operators
    - `N` is the number operator for the bosonic mode
    - `a` and `a^dag` are lowering and raising operators for the bosonic mode
    - `spin_splitting`, `boson_splitting`, and `coupling` are scalars
    """
    spin_op_dim = spin_ops.get_spin_op_dim(num_spins)
    spin_iden = scipy.sparse.identity(spin_op_dim)
    boson_iden = scipy.sparse.identity(boson_dim)

    spin_term_L = spin_ops.get_Sz_L(num_spins)
    spin_term = scipy.sparse.kron(
        spin_term_L - spin_ops.get_dual(spin_term_L),
        scipy.sparse.kron(boson_iden, boson_iden),
    )

    op_num = get_boson_num_op(boson_dim)
    boson_term = scipy.sparse.kron(
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

    hamiltonian_bracket = (
        spin_splitting * spin_term + boson_splitting * boson_term + coupling * coupling_term
    )
    return -1j * hamiltonian_bracket


@_with_default_boson_dim
def get_dissipator(
    num_spins: int,
    decay_res: float,
    decay_spin: float,
    dephasing: bool,
    *,
    boson_dim: int,
) -> scipy.sparse.spmatrix:
    """Construct a dissipator that generates spin and boson decay.

    The dissipator is represented by a super-operator that acts on a vectorized density matrix.
    """
    if dephasing:
        # resonator and spin dephasing
        spin_op = spin_ops.get_local_dissipator(num_spins, "z")
        boson_op = get_boson_num_op(boson_dim)
    else:
        # resonator and spin decay
        spin_op = spin_ops.get_local_dissipator(num_spins, "-")
        boson_op = get_boson_lower_op(boson_dim)

    spin_op_dim = spin_ops.get_spin_op_dim(num_spins)
    dissipator_res = scipy.sparse.kron(
        scipy.sparse.identity(spin_op_dim),
        to_dissipation_generator(boson_op),
    )
    dissipator_spin = scipy.sparse.kron(
        spin_op,
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
get_state_X = _with_boson_vacuum(spin_ops.get_state_X)


################################################################################
# Fisher info calculation


def get_states(
    times: np.ndarray,
    initial_state: np.ndarray,
    generator: scipy.sparse.spmatrix,
    method: str = DEFAULT_INTEGRATION_METHOD,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> np.ndarray:
    """For a given initial state and generator of time evolution, return states at later times."""
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
    state: np.ndarray,
    state_deriv: np.ndarray,
    etol_scale: float = DEFAULT_ETOL_SCALE,
) -> float:
    """Compute the QFI from a state and its derivative w.r.t. the parameter to be estimated."""
    # collect data from each block of the density matrix
    block_nums = []  # numerators in contributions to the QFI
    block_vals = []  # eigenvalues of the state

    for block, block_deriv in zip(
        spin_ops.get_spin_blocks(state),
        spin_ops.get_spin_blocks(state_deriv),
    ):
        vals, vecs = np.linalg.eigh(block)
        block_vals.append(vals)
        block_nums.append(abs(vecs.conj().T @ block_deriv @ vecs) ** 2)

    # identify numerical cutoff for small eigenvalues, below which they get set to 0
    etol = abs(etol_scale * min(val for vals in block_vals for val in vals))

    val_QFI = 0
    for vals, nums in zip(block_vals, block_nums):
        vals[vals < etol] = 0
        dens = vals[:, np.newaxis] + vals[np.newaxis, :]  # matrix M[i, j] = w[i] + w[j]

        # matrix of booleans (True/False) to ignore zero eigenvalues
        include = ~np.isclose(dens, 0)  # matrix of booleans (True/False)

        # add contribution to the QFI from this block of the density matrix
        val_QFI += (nums[include] / dens[include]).sum()

    return 2 * val_QFI


def get_QFI_vals(
    times: np.ndarray,
    num_spins: int,
    spin_splitting: float,
    boson_splitting: float,
    coupling: float,
    decay_res: float,
    decay_spin: float,
    dephasing: bool,
    initial_state: np.ndarray,
    method: str = DEFAULT_INTEGRATION_METHOD,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    etol_scale: float = DEFAULT_ETOL_SCALE,
    diff_step: float = DEFAULT_DIFF_STEP,
    terminate_early: bool = True,
) -> np.ndarray:
    """Get the QFI over time for a spin-boson system defined by the provided arguments."""
    spin_op_dim = spin_ops.get_spin_op_dim(num_spins)
    boson_dim = int(np.round(np.sqrt(initial_state.size // spin_op_dim)))

    # compute the generators of time evolution
    dissipator = get_dissipator(num_spins, decay_res, decay_spin, dephasing, boson_dim=boson_dim)
    hamiltonian_p = get_hamiltonian_generator(
        num_spins, spin_splitting, boson_splitting, coupling + diff_step / 2, boson_dim=boson_dim
    )
    hamiltonian_m = get_hamiltonian_generator(
        num_spins, spin_splitting, boson_splitting, coupling - diff_step / 2, boson_dim=boson_dim
    )
    generator_p = hamiltonian_p + dissipator
    generator_m = hamiltonian_m + dissipator

    # reshape the initial state
    op_shape = (spin_op_dim, boson_dim, boson_dim)
    initial_state = initial_state.reshape(op_shape)
    vacuum_index = (-((num_spins + 1) ** 2), 0, 0)

    def _get_states(
        times: np.ndarray, initial_state: np.ndarray, generator: scipy.sparse.spmatrix
    ) -> np.ndarray:
        shape = (len(times),) + op_shape
        return get_states(times, initial_state, generator, method, rtol, atol).reshape(shape)

    if not terminate_early:
        states_p = _get_states(times, initial_state, generator_p)
        states_m = _get_states(times, initial_state, generator_m)
        vals_QFI = _get_QFI_vals(states_p, states_m, vacuum_index, diff_step, etol_scale)

    else:
        max_QFI = 0.0
        vals_QFI = [0.0]
        initial_state_p = initial_state
        initial_state_m = initial_state

        for time_section in _get_time_sections(times):
            states_p = _get_states(time_section, initial_state_p, generator_p)
            states_m = _get_states(time_section, initial_state_m, generator_m)
            section_vals_QFI = _get_QFI_vals(
                states_p[1:], states_m[1:], vacuum_index, diff_step, etol_scale
            )
            vals_QFI.extend(section_vals_QFI)

            max_QFI = max(max_QFI, max(section_vals_QFI))
            if vals_QFI[-1] < max_QFI / 2:
                break

            initial_state_p = states_p[-1]
            initial_state_m = states_m[-1]

    return np.array(vals_QFI)


def _get_QFI_vals(
    states_p: np.ndarray,
    states_m: np.ndarray,
    vacuum_index: tuple[int, int, int],
    diff_step: float,
    etol_scale: float,
) -> list[float]:
    states_avg = (states_p + states_m) / 2
    states_deriv = (states_p - states_m) / diff_step
    return [
        get_QFI(state_avg, state_deriv, etol_scale)
        for state_avg, state_deriv in zip(states_avg, states_deriv)
    ]


def _get_time_sections(times: np.ndarray, section_size: float = 1) -> Iterator[np.ndarray]:
    for time in np.arange(times[0], times[-1], section_size):
        start = int(np.argmax(times >= time))
        end = int(np.argmax(times >= time + section_size))
        yield times[start : end + 1]


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
    initial_state: np.ndarray,
    method: str = DEFAULT_INTEGRATION_METHOD,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    etol_scale: float = DEFAULT_ETOL_SCALE,
    diff_step: float = DEFAULT_DIFF_STEP,
) -> np.ndarray:
    """Get the QFI over time for a spin-boson system defined by the provided arguments."""
    spin_op_dim = spin_ops.get_spin_op_dim(num_spins)
    boson_dim = int(np.round(np.sqrt(initial_state.size // spin_op_dim)))

    # compute time-evolved states
    hamiltonian = get_hamiltonian_generator(
        num_spins, spin_splitting, boson_splitting, coupling, boson_dim=boson_dim
    )
    dissipator = get_dissipator(num_spins, decay_res, decay_spin, boson_dim=boson_dim)
    generator = hamiltonian + dissipator
    op_shape = (spin_op_dim, boson_dim, boson_dim)
    shape = (len(times),) + op_shape
    states = get_states(times, initial_state, generator, method, rtol, atol).reshape(shape)

    # compute the generators of time evolution
    dissipator_p = get_dissipator(
        num_spins, decay_res, decay_spin + diff_step / 2, boson_dim=boson_dim
    )
    dissipator_m = get_dissipator(
        num_spins, decay_res, decay_spin - diff_step / 2, boson_dim=boson_dim
    )
    generator_p = hamiltonian + dissipator_p
    generator_m = hamiltonian + dissipator_m

    # compute kraus operators
    kraus_ops_p = [_log_channel_to_kraus_ops(time * generator_p, op_shape) for time in times]
    kraus_ops_m = [_log_channel_to_kraus_ops(time * generator_m, op_shape) for time in times]

    # compute bound at each time
    vals_bound = np.zeros(len(times))
    for tt, state in enumerate(states):
        op_A = op_B = np.zeros_like(states[0])
        for kraus_vec_p, kraus_vec_m in zip(kraus_ops_p[tt], kraus_ops_m[tt]):
            kraus_op_p = kraus_vec_p.reshape(op_shape)
            kraus_op_m = kraus_vec_m.reshape(op_shape)
            kraus_op = (kraus_op_p + kraus_op_m) / 2
            kraus_op_deriv = (kraus_op_p - kraus_op_m) / diff_step
            kraus_op_deriv_dag = spin_ops.adjoint(kraus_op_deriv)

            op_A += spin_ops.matmul(kraus_op_deriv_dag, kraus_op_deriv)
            op_B += 1j * spin_ops.matmul(kraus_op_deriv_dag, kraus_op)

        term_A = state.conj().ravel() @ op_A.ravel()
        term_B = state.conj().ravel() @ op_B.ravel()
        vals_bound[tt] = (term_A - term_B**2).real

    return 4 * vals_bound


def _log_channel_to_kraus_ops(
    log_channel: scipy.sparse.spmatrix, op_shape: tuple[int, ...]
) -> list[np.ndarray]:
    """Get the Kraus operators of a quantum channel specified by its natural logarithm."""
    channel = np.array(scipy.sparse.linalg.expm(log_channel.tocsc()).todense())
    vals, vecs = np.linalg.eigh(channel)
    return [
        np.sqrt(abs(val)) * vec.reshape(op_shape)
        for val, vec in zip(vals, vecs.T)
        if not np.isclose(val, 0)
    ]
