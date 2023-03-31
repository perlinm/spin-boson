"""
CONTENTS: methods for constructing operators for a spin system with permutational symmetry.
"""
import functools
from typing import Iterator, Literal, Optional

import numpy as np
import scipy


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
    num_spins = get_num_spins(spin_op.shape[0])
    new_op = scipy.sparse.dok_matrix(spin_op.shape, dtype=spin_op.dtype)
    for idx_out, idx_inp in zip(*spin_op.nonzero()):
        new_idx_out = get_transpose_basis_index(idx_inp, num_spins % 2)
        new_idx_inp = get_transpose_basis_index(idx_out, num_spins % 2)
        new_op[new_idx_out, new_idx_inp] = spin_op[idx_out, idx_inp]
    return new_op.asformat(spin_op.getformat())


def get_Sz_L(num_spins: int) -> scipy.sparse.spmatrix:
    """Compute the superoporator `O` that acts on a density matrix `rho` as `O(rho) = S_z @ rho`."""
    vals = [
        up_out - (shell_dim - 1) / 2
        for _, shell_dim, up_out, _ in spin_op_basis_elements(num_spins)
    ]
    shape = (len(vals),) * 2
    return scipy.sparse.dia_matrix(([vals], [0]), shape)


def get_Sp_L(num_spins: int) -> scipy.sparse.spmatrix:
    """Compute the superoporator `O` that acts on density matrix `rho` as `O(rho) = S_p @ rho`."""
    dim = get_spin_op_dim(num_spins)
    mat = scipy.sparse.dok_matrix((dim, dim), dtype=float)
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
    """Compute the superoporator `O` that acts on density matrix `rho` as `O(rho) = S_m @ rho`."""
    return get_Sp_L(num_spins).T


def get_S(num_spins: int) -> scipy.sparse.spmatrix:
    """Compute the superoporator `O` that acts on density matrix `rho` as `O(rho) = S rho`."""
    vals = [(shell_dim - 1) / 2 for _, shell_dim, _, _ in spin_op_basis_elements(num_spins)]
    shape = (len(vals),) * 2
    return scipy.sparse.dia_matrix(([vals], [0]), shape)


def get_SS(num_spins: int) -> scipy.sparse.spmatrix:
    """Compute the superoporator `O` that acts on density matrix `rho` as `O(rho) = S (S+1) rho`."""
    op_S = get_S(num_spins)
    op_I = scipy.sparse.identity(op_S.shape[0])
    return op_S @ (op_S + op_I)


def get_local_dissipator(
    num_spins: int,
    local_op: Literal["+", "-", "z"],
) -> scipy.sparse.spmatrix:
    """Return a dissipator that generates local spin dissipation."""
    dim = get_spin_op_dim(num_spins)
    identity_op = num_spins * scipy.sparse.identity((dim, dim), dtype=float)
    if local_op == "+":
        recycling_op = (identity_op - get_Sz_L(num_spins)) / 2
    elif local_op == "-":
        recycling_op = (identity_op + get_Sz_L(num_spins)) / 2
    elif local_op == "z":
        recycling_op = identity_op
    recycling_term = -(recycling_op + get_dual(recycling_op)) / 2
    return _local_op_conjugator(num_spins, local_op) + recycling_term


def _local_op_conjugator(
    num_spins: int,
    op_lft: Literal["+", "-", "z"],
    op_rht: Optional[Literal["+", "-", "z"]] = None,
) -> scipy.sparse.spmatrix:
    if op_rht is None:
        op_rht = op_lft

    dim = get_spin_op_dim(num_spins)
    mat = scipy.sparse.dok_matrix((dim, dim), dtype=float)
    for idx_inp, shell_dim, up_out, up_inp in spin_op_basis_elements(num_spins):
        spin_val = (shell_dim - 1) / 2
        shifted_up_out = _shifted_proj(up_out, op_lft)
        shifted_up_inp = _shifted_proj(up_inp, op_rht)

        idx_out = get_spin_basis_index(shell_dim, shifted_up_out, shifted_up_inp)
        coef_0 = (num_spins + 2) / (spin_val * (spin_val + 1))
        _A_lft = _coef_A(op_lft, spin_val, up_out)
        _A_rht = _coef_A(op_rht, spin_val, up_inp)
        mat[idx_out, idx_inp] = coef_0 * _A_lft * _A_rht / 4

        idx_out = get_spin_basis_index(shell_dim - 1, shifted_up_out, shifted_up_inp)
        coef_m = (num_spins + 2 * spin_val + 2) / (spin_val * (2 * spin_val + 1))
        _B_lft = _coef_B(op_lft, spin_val, up_out)
        _B_rht = _coef_B(op_rht, spin_val, up_inp)
        mat[idx_out, idx_inp] = coef_m * _B_lft * _B_rht

        idx_out = get_spin_basis_index(shell_dim + 1, shifted_up_out, shifted_up_inp)
        coef_p = (num_spins - 2 * spin_val) / (2 * spin_val**2 + 3 * spin_val + 1)
        _D_lft = _coef_D(op_lft, spin_val, up_out)
        _D_rht = _coef_D(op_rht, spin_val, up_inp)
        mat[idx_out, idx_inp] = coef_p * _D_lft * _D_rht / 4

    return mat


def _shifted_proj(spin_proj: float, shift: Literal["+", "-", "z"]) -> float:
    if shift == "+":
        return spin_proj + 1
    if shift == "-":
        return spin_proj - 1
    return spin_proj


def _coef_A(op: Literal["+", "-", "z"], spin_val: float, spin_proj: float) -> float:
    if op == "+":
        return np.sqrt((spin_val - spin_proj) * (spin_val + spin_proj + 1))
    if op == "-":
        return np.sqrt((spin_val + spin_proj) * (spin_val - spin_proj + 1))
    return spin_proj


def _coef_B(op: Literal["+", "-", "z"], spin_val: float, spin_proj: float) -> float:
    if op == "+":
        return np.sqrt((spin_val - spin_proj) * (spin_val - spin_proj - 1))
    if op == "-":
        return -np.sqrt((spin_val + spin_proj) * (spin_val + spin_proj - 1))
    return np.sqrt((spin_val + spin_proj) * (spin_val - spin_proj))


def _coef_D(op: Literal["+", "-", "z"], spin_val: float, spin_proj: float) -> float:
    if op == "+":
        return -np.sqrt((spin_val + spin_proj + 1) * (spin_val + spin_proj + 2))
    if op == "-":
        return np.sqrt((spin_val - spin_proj + 1) * (spin_val - spin_proj + 2))
    return np.sqrt((spin_val + spin_proj + 1) * (spin_val - spin_proj + 1))


def get_dicke_state(num_spins: int, num_excitations: int) -> np.ndarray:
    """Prepare a Dicke state of the given number of spins."""
    state = np.zeros(get_spin_op_dim(num_spins))
    shell_start = -(num_spins + 1)**2
    shell_index = num_excitations * (num_spins + 2)
    state[shell_start + shell_index] = 1
    return state


def get_ghz_state(num_spins: int) -> np.ndarray:
    """Prepare a GHZ state of the given number of spins."""
    state = np.zeros(get_spin_op_dim(num_spins))
    # set matrix elements to 0.5 within the S = N/2 manifold:
    #   |0><0|, |0><N|, |N><0|, |N><N|,
    # where |M> is the Dicke state with M excitations
    shell_start = -(num_spins + 1)**2
    state[shell_start] = 0.5
    state[shell_start + num_spins] = 0.5
    state[-1 - num_spins] = 0.5
    state[-1] = 0.5
    return state
