"""Methods for constructing operators for a spin system with permutational symmetry."""
import functools
from typing import Iterator, Literal, Optional

import numpy as np
import scipy


def op_tensor_to_matrix(tensor: np.ndarray) -> np.ndarray:
    """Convert a tensor op on a factorized Hilbert space (HS) into a matrix on the entire HS."""
    matrix_dim = int(np.prod(tensor.shape[::2]))
    matrix_shape = (matrix_dim, matrix_dim)
    return np.moveaxis(
        tensor,
        range(0, tensor.ndim, 2),
        range(tensor.ndim // 2),
    ).reshape(matrix_shape)


def op_matrix_to_tensor(
    matrix: np.ndarray,
    dims: tuple[int, ...],
    aux_dims: bool = False,
) -> np.ndarray:
    """Convert a matrix op into a tensor op on a factorized Hilbert space."""
    if aux_dims:
        first_dim = matrix.shape[0] // int(np.prod(dims))
        dims = (first_dim,) + dims
    return np.moveaxis(
        matrix.reshape(dims + dims),
        range(len(dims)),
        range(0, 2 * len(dims), 2),
    )


################################################################################
# spin operator format metadata


@functools.cache
def get_spin_basis_index(shell_dim: int, up_out: int, up_inp: int) -> int:
    """Compute the basis index for the spin operator `|S><S|⊗|M_f><M_i| --> |S M_f M_i)`.

    Since `S, M_f, M_i` can generally be half-integer-valued and the `M`s run from `-S` to `S`, we
    instead index by:
        - `shell_dim = 2 * S + 1`,
        - `up_out = S + M_f`,
        - `up_inp = S + M_i`,
    where `shell_dim` is a positive integer, and `up_out, up_inp` are integers in `[0, shell_dim)`.
    """
    assert 0 <= up_out < shell_dim
    assert 0 <= up_inp < shell_dim
    net_index = up_out * shell_dim + up_inp
    if shell_dim > 2:
        inner_dim = shell_dim - 2
        net_index += get_spin_basis_index(inner_dim, inner_dim - 1, inner_dim - 1) + 1
    return net_index


@functools.cache
def get_spin_basis_vals(basis_index: int, num_spins_mod_2: int) -> tuple[int, int, int]:
    """Inverse of get_spin_basis_index."""
    shell_dim = num_spins_mod_2 % 2 + 1
    while (shell_op_dim := shell_dim**2) <= basis_index:
        basis_index -= shell_op_dim
        shell_dim += 2
    up_out = basis_index // shell_dim
    up_inp = basis_index % shell_dim
    return shell_dim, up_out, up_inp


def spin_op_basis_elements(num_spins: int) -> Iterator[tuple[int, int, int, int]]:
    """Iterate over the basis elements of operators acting on a given number of spins."""
    index = 0
    for shell_dim in range(num_spins % 2 + 1, num_spins + 2, 2):
        for up_out in range(shell_dim):
            for up_inp in range(shell_dim):
                yield index, shell_dim, up_out, up_inp
                index += 1


@functools.cache
def get_spin_op_dim(num_spins: int) -> int:
    """Get the dimension of operators addressing a given number of spins."""
    if num_spins < 0:
        return 0
    return get_spin_basis_index(num_spins + 1, num_spins, num_spins) + 1


@functools.cache
def get_num_spins(spin_op_dim: int) -> int:
    """Get the total number of spins addressed by an operator of a given dimension."""
    for num_spins in range(spin_op_dim):
        if get_spin_op_dim(num_spins) == spin_op_dim:
            return num_spins
    raise ValueError(f"Invalid operator dimension: {spin_op_dim}")


def get_transpose_basis_index(basis_index: int, num_spins_mod_2: int) -> int:
    """Given the basis index for `|S M_f M_i)`, return the basis index for `|S M_i M_f)`."""
    shell_dim, up_out, up_inp = get_spin_basis_vals(basis_index, num_spins_mod_2 % 2)
    return get_spin_basis_index(shell_dim, up_inp, up_out)


def get_dual(spin_op: scipy.sparse.spmatrix) -> scipy.sparse.spmatrix:
    """Compute the right-acting "dual" of a left-acting operator.

    For a given (super)operator `O` that acts on a density matrix `rho` as `O(rho) = M @ rho`,
    compute the dual `Q` that acts on `rho` as `Q(rho) = rho @ M`.

    The dual is determined by `rho @ M = (M^T @ rho^T)^T`.
    """
    initial_format = spin_op.getformat()
    spin_op = spin_op.tocsr()
    num_spins = get_num_spins(spin_op.shape[0])
    new_op = scipy.sparse.dok_array(spin_op.shape, dtype=spin_op.dtype)
    for idx_out, idx_inp in zip(*spin_op.nonzero()):
        new_idx_out = get_transpose_basis_index(idx_inp, num_spins % 2)
        new_idx_inp = get_transpose_basis_index(idx_out, num_spins % 2)
        new_op[new_idx_out, new_idx_inp] = spin_op[idx_out, idx_inp]
    return new_op.asformat(initial_format)


################################################################################
# methods to construct spin operators


def get_Sz_L(num_spins: int) -> scipy.sparse.spmatrix:
    """Compute the superoporator `O` that acts on a density matrix `rho` as `O(rho) = S_z rho`."""
    vals = [
        up_out - (shell_dim - 1) / 2
        for _, shell_dim, up_out, _ in spin_op_basis_elements(num_spins)
    ]
    shape = (len(vals),) * 2
    return scipy.sparse.dia_array(([vals], [0]), shape)


def get_Sp_L(num_spins: int) -> scipy.sparse.spmatrix:
    """Compute the superoporator `O` that acts on density matrix `rho` as `O(rho) = S_p rho`."""
    dim = get_spin_op_dim(num_spins)
    mat = scipy.sparse.dok_array((dim, dim), dtype=float)
    for idx_inp, shell_dim, up_out, up_inp in spin_op_basis_elements(num_spins):
        new_up_out = up_out + 1
        if not 0 <= new_up_out < shell_dim:
            continue
        idx_out = get_spin_basis_index(shell_dim, new_up_out, up_inp)
        spin_val = (shell_dim - 1) / 2
        proj_val = up_out - spin_val
        mat_val = np.sqrt(spin_val * (spin_val + 1) - proj_val * (proj_val + 1))
        mat[idx_out, idx_inp] = mat_val
    return mat


def get_Sm_L(num_spins: int) -> scipy.sparse.spmatrix:
    """Compute the superoporator `O` that acts on density matrix `rho` as `O(rho) = S_m rho`."""
    return get_Sp_L(num_spins).T


def get_S(num_spins: int) -> scipy.sparse.spmatrix:
    """Compute the superoporator `O` that acts on density matrix `rho` as `O(rho) = S rho`."""
    vals = [(shell_dim - 1) / 2 for _, shell_dim, _, _ in spin_op_basis_elements(num_spins)]
    shape = (len(vals),) * 2
    return scipy.sparse.dia_array(([vals], [0]), shape)


def get_SS(num_spins: int) -> scipy.sparse.spmatrix:
    """Compute the superoporator `O` that acts on density matrix `rho` as `O(rho) = S (S+1) rho`."""
    op_S = get_S(num_spins)
    op_I = scipy.sparse.identity(op_S.shape[0])
    return op_S @ (op_S + op_I)


def get_local_dissipator(
    num_spins: int,
    local_op: Literal["z", "+", "-"],
) -> scipy.sparse.spmatrix:
    """Return a dissipator that generates spontaneous local spin dephasing, excitation, or decay."""
    dim = get_spin_op_dim(num_spins)
    identity_op = num_spins * scipy.sparse.identity(dim, dtype=float)
    if local_op == "z":
        recycling_op = identity_op / 4
    elif local_op == "+":
        recycling_op = (identity_op - 2 * get_Sz_L(num_spins)) / 2
    elif local_op == "-":
        recycling_op = (identity_op + 2 * get_Sz_L(num_spins)) / 2
    recycling_term = recycling_op + get_dual(recycling_op)
    return _local_op_conjugator(num_spins, local_op) - recycling_term / 2


def _local_op_conjugator(
    num_spins: int,
    op_lft: Literal["z", "+", "-"],
    op_rht: Optional[Literal["z", "+", "-"]] = None,
) -> scipy.sparse.spmatrix:
    """For a collective spin state `rho`, return sum_{spin j} O_lft_j rho O_rht_j^dag,
    where O_lft and O_rht are single-spin dephasing, excitation, or decay operators, respectively
    |1><1| - |0><0|, |1><0|, |0><1|.
    """
    if op_rht is None:
        op_rht = op_lft

    total_dim = get_spin_op_dim(num_spins)
    mat = scipy.sparse.dok_array((total_dim, total_dim), dtype=float)
    for idx_inp, shell_dim, up_out, up_inp in spin_op_basis_elements(num_spins):
        spin_val = (shell_dim - 1) / 2
        proj_out = up_out - spin_val
        proj_inp = up_inp - spin_val

        for spin_shift in [0, -1, +1]:
            dim_out = shell_dim + 2 * spin_shift
            shifted_up_out = up_out + _proj_shift(op_lft) + spin_shift
            shifted_up_inp = up_inp + _proj_shift(op_rht) + spin_shift
            if (
                0 < dim_out <= num_spins + 1
                and 0 <= shifted_up_out < dim_out
                and 0 <= shifted_up_inp < dim_out
            ):
                idx_out = get_spin_basis_index(dim_out, shifted_up_out, shifted_up_inp)
                coef_S = _get_coef_S(num_spins, spin_val, spin_shift)
                coef_CG_lft = _get_coef_CG(op_lft, spin_val, proj_out, spin_shift)
                coef_CG_rht = _get_coef_CG(op_rht, spin_val, proj_inp, spin_shift)
                mat[idx_out, idx_inp] = coef_S * coef_CG_lft * coef_CG_rht

    mat = mat.tocsr()
    mat.eliminate_zeros()
    return mat


def _proj_shift(shift: Literal["z", "+", "-"]) -> float:
    if shift == "+":
        return 1
    if shift == "-":
        return -1
    return 0


def _get_coef_S(num_spins: int, spin_val: float, spin_shift: int) -> float:
    if spin_val == 0 and spin_shift != 1:
        return 0
    if spin_shift == 0:
        ratio = _coef_alpha(num_spins, spin_val + 1) / _coef_dim(num_spins, spin_val)
        return (1 + ratio * (2 * spin_val + 1) / (spin_val + 1)) / (2 * spin_val)
    if spin_shift == -1:
        ratio = _coef_alpha(num_spins, spin_val) / _coef_dim(num_spins, spin_val)
        return ratio / (2 * spin_val)
    assert spin_shift == 1
    ratio = _coef_alpha(num_spins, spin_val + 1) / _coef_dim(num_spins, spin_val)
    return ratio / (2 * (spin_val + 1))


def _coef_alpha(num_spins: int, spin_val: float) -> int:
    return scipy.special.comb(num_spins, num_spins / 2 + spin_val, exact=True)


def _coef_dim(num_spins: int, spin_val: float) -> int:
    binom_factor = scipy.special.comb(num_spins, num_spins / 2 + spin_val, exact=True)
    return binom_factor * int(2 * spin_val + 1) / int(num_spins / 2 + spin_val + 1)


def _get_coef_CG(
    op: Literal["z", "+", "-"],
    spin_val: float,
    spin_proj: float,
    spin_shift: int,
) -> float:
    if spin_shift == 0:
        return _get_coef_A(op, spin_val, spin_proj)
    if spin_shift == -1:
        return _get_coef_B(op, spin_val, spin_proj)
    assert spin_shift == 1
    return _get_coef_D(op, spin_val, spin_proj)


def _get_coef_A(op: Literal["z", "+", "-"], spin_val: float, spin_proj: float) -> float:
    if op == "+":
        return np.sqrt((spin_val - spin_proj) * (spin_val + spin_proj + 1))
    if op == "-":
        return np.sqrt((spin_val + spin_proj) * (spin_val - spin_proj + 1))
    assert op == "z"
    return spin_proj


def _get_coef_B(op: Literal["z", "+", "-"], spin_val: float, spin_proj: float) -> float:
    if op == "+":
        return np.sqrt((spin_val - spin_proj) * (spin_val - spin_proj - 1))
    if op == "-":
        return -np.sqrt((spin_val + spin_proj) * (spin_val + spin_proj - 1))
    assert op == "z"
    return np.sqrt((spin_val + spin_proj) * (spin_val - spin_proj))


def _get_coef_D(op: Literal["z", "+", "-"], spin_val: float, spin_proj: float) -> float:
    if op == "+":
        return -np.sqrt((spin_val + spin_proj + 1) * (spin_val + spin_proj + 2))
    if op == "-":
        return np.sqrt((spin_val - spin_proj + 1) * (spin_val - spin_proj + 2))
    assert op == "z"
    return np.sqrt((spin_val + spin_proj + 1) * (spin_val - spin_proj + 1))


################################################################################
# methods to construct spin states


def get_vacuum_state(num_spins: int) -> np.ndarray:
    """Construct the vacuum (all-spin-down) state for a collection of spins."""
    return get_dicke_state(num_spins, 0)


def get_state_X(num_spins: int) -> np.ndarray:
    """Construct an X-polarized spin state."""
    if num_spins == 0:
        return np.ones(1)

    vals = [
        np.sqrt(scipy.special.binom(num_spins, spins_up) / 2**num_spins)
        for spins_up in range(num_spins + 1)
    ]
    state_vec = np.outer(vals, vals).ravel()
    state = np.zeros(get_spin_op_dim(num_spins))
    state[-state_vec.size :] = state_vec
    return state


def get_dicke_state(num_spins: int, num_excitations: int) -> np.ndarray:
    """Prepare a Dicke state of the given number of spins."""
    if num_spins == 0:
        return np.ones(1)
    state = np.zeros(get_spin_op_dim(num_spins))
    if 0 <= num_excitations <= num_spins:
        shell_start = -((num_spins + 1) ** 2)
        shell_index = num_excitations * (num_spins + 2)
        state[shell_start + shell_index] = 1
    return state


def get_ghz_state(num_spins: int) -> np.ndarray:
    """Prepare a GHZ state of the given number of spins."""
    if num_spins == 0:
        return np.ones(1)
    shell_dim = num_spins + 1
    state = np.zeros(get_spin_op_dim(num_spins))
    # set matrix elements to 0.5 within the S = N/2 manifold:
    #   |0><0|, |0><N|, |N><0|, |N><N|,
    # where |M> is the Dicke state with M excitations
    shell_start = -(shell_dim**2)
    state[shell_start] = 0.5
    state[shell_start + num_spins] = 0.5
    state[-shell_dim] = 0.5
    state[-1] = 0.5
    return state


def get_spin_blocks(state: np.ndarray) -> Iterator[np.ndarray]:
    """Iterate over the fixed-S blocks of a density matrix."""
    spin_op_dim = state.shape[0]
    num_spins = get_num_spins(spin_op_dim)
    shell_start = 0
    shell_dim = num_spins % 2 + 1
    while shell_start < spin_op_dim:
        block_size = shell_dim**2
        block_slice = slice(shell_start, shell_start + block_size)
        block_data = state[block_slice]

        tensor_shape = (shell_dim, shell_dim) + state.shape[1:]
        yield op_tensor_to_matrix(block_data.reshape(tensor_shape))

        shell_start += block_size
        shell_dim += 2


def get_spin_trace(op: np.ndarray) -> np.ndarray:
    """Preform a partial trace over an operator's spin degrees of freedom."""
    num_spins = get_num_spins(op.shape[0])
    shell_dims = range(num_spins % 2 + 1, num_spins + 2, 2)
    return sum(
        op[get_spin_basis_index(shell_dim, num, num)]
        for shell_dim in shell_dims
        for num in range(shell_dim)
    )


################################################################################
# methods to manipulate operators on the spin Hilbert space


def adjoint(op: np.ndarray) -> np.ndarray:
    """Compute the conjugate-transpose of a tensor spin operator."""
    op_shape = op.shape
    boson_dims = op_shape[1::2]
    block_shape = (-1,) + op_shape[1:]
    return np.vstack(
        [
            op_matrix_to_tensor(block.conj().T, boson_dims, aux_dims=True).reshape(block_shape)
            for block in get_spin_blocks(op)
        ]
    )


def matmul(op_a: np.ndarray, op_b: np.ndarray) -> np.ndarray:
    """Matrix-multiply two spin tensor operators."""
    assert op_a.shape == op_b.shape
    op_shape = op_a.shape
    boson_dims = op_shape[1::2]
    block_shape = (-1,) + op_shape[1:]
    return np.vstack(
        [
            op_matrix_to_tensor(block_a @ block_b, boson_dims, aux_dims=True).reshape(block_shape)
            for block_a, block_b in zip(get_spin_blocks(op_a), get_spin_blocks(op_b))
        ]
    )
