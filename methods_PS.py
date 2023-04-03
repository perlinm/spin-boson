import functools
from typing import Optional, Sequence

import numpy as np
import scipy

import spin_ops

DEFAULT_INTEGRATION_METHOD = "DOP853"
DEFAULT_DIFF_STEP = 1e-4  # step size for finite-difference derivative
DEFAULT_RTOL = 1e-12  # relative/absolute error tolerance for numerical intgeration
DEFAULT_ATOL = 1e-12
DEFAULT_ETOL = np.sqrt(DEFAULT_DIFF_STEP * DEFAULT_ATOL)  # eigenvalue cutoff

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


def get_hamiltonian_generator(
    num_spins: int,
    splitting: float,
    coupling: float,
    *,
    boson_dim: Optional[int] = None,
) -> scipy.sparse.spmatrix:
    if boson_dim is None:
        boson_dim = num_spins + 1

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


def get_dissipator(
    num_spins: int,
    decay_res: float,
    decay_spin: float,
    *,
    boson_dim: Optional[int] = None,
) -> scipy.sparse.spmatrix:
    if not boson_dim:
        boson_dim = num_spins + 1
    spin_op_dim = spin_ops.get_spin_op_dim(num_spins)
    dissipator_res = scipy.sparse.kron(
        scipy.sparse.identity(spin_op_dim),
        to_dissipation_generator(get_boson_lower_op(num_spins + 1)),
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


def _get_block_QFI(state: np.ndarray, state_diff: np.ndarray, etol: float = DEFAULT_ETOL):
    vals, vecs = np.linalg.eigh(state)

    # numerators and denominators
    nums = 2 * abs(vecs.conj().T @ state_diff @ vecs) ** 2
    dens = vals[:, np.newaxis] + vals[np.newaxis, :]  # matrix M[i, j] = w[i] + w[j]

    include = ~np.isclose(dens, 0, atol=etol)  # matrix of booleans (True/False)
    return (nums[include] / dens[include]).sum()


def get_QFI(state: np.ndarray, state_diff: np.ndarray, etol: float = DEFAULT_ETOL) -> float:
    return sum(
        _get_block_QFI(block, block_diff, etol)
        for block, block_diff in zip(
            spin_ops.get_spin_blocks(state),
            spin_ops.get_spin_blocks(state_diff),
        )
    )


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
    etol: float = DEFAULT_ETOL,
    diff_step: float = DEFAULT_DIFF_STEP,
) -> tuple[np.ndarray, np.ndarray]:
    spin_op_dim = spin_ops.get_spin_op_dim(num_spins)
    boson_dim = num_spins + 1

    hamiltonian_p = get_hamiltonian_generator(num_spins, splitting, coupling + diff_step / 2)
    hamiltonian_m = get_hamiltonian_generator(num_spins, splitting, coupling - diff_step / 2)
    dissipator = get_dissipator(num_spins, decay_res, decay_spin)

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

        vals_QFI[tt] = get_QFI(state_avg, state_diff, etol)
        vals_QFI_SA[tt] = np.real(1 - state_avg[vacuum_index]) * vals_QFI[tt]

    return vals_QFI, vals_QFI_SA
