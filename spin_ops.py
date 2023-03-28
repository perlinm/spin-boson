"""
CONTENTS: methods for constructing operators for a spin system with permutational symmetry.
"""
import functools
from typing import Iterator

import numpy as np
import scipy


@functools.cache
def get_spin_basis_index(shell_dim: int, up_out: int, up_inp: int) -> int:
    """Compute the basis index for the spin operator |SM><SM'|."""
    assert 0 <= up_out < shell_dim
    assert 0 <= up_inp < shell_dim
    net_index = up_out * shell_dim + up_inp
    if shell_dim > 2:
        inner_dim = shell_dim - 2
        net_index += get_spin_basis_index(inner_dim, inner_dim - 1, inner_dim - 1) + 1
    return net_index


@functools.cache
def get_spin_basis_vals(basis_index: int, spin_num_mod_2: int) -> tuple[int, int, int]:
    """Inverse of get_spin_basis_index."""
    shell_dim = spin_num_mod_2 % 2 + 1
    while (shell_op_dim := shell_dim**2) <= basis_index:
        basis_index -= shell_op_dim
        shell_dim += 2
    up_out = basis_index // shell_dim
    up_inp = basis_index % shell_dim
    return shell_dim, up_out, up_inp


def spin_op_basis_elements(spin_num: int) -> Iterator[tuple[int, int, int, int]]:
    """Iterate over the basis elements of operators acting on a given number of spins."""
    index = 0
    for shell_dim in range(spin_num % 2 + 1, spin_num + 2, 2):
        for up_out in range(shell_dim):
            for up_inp in range(shell_dim):
                yield index, shell_dim, up_out, up_inp
                index += 1


@functools.cache
def get_spin_op_dim(spin_num: int) -> int:
    """Get the dimension of operators addressing a given number of spins."""
    return get_spin_basis_index(spin_num + 1, spin_num, spin_num) + 1


@functools.cache
def get_spin_num(spin_op_dim: int) -> int:
    """Get the total number of spins addressed by an operator of a given dimension."""
    for spin_num in range(spin_op_dim):
        if get_spin_op_dim(spin_num) == spin_op_dim:
            return spin_num
    raise ValueError(f"Invalid operator dimension: {spin_op_dim}")


def get_transpose_basis_index(basis_index: int, spin_num_mod_2: int) -> int:
    """Given the basis index for |SM><SM'|, return the basis index for |SM'><SM|."""
    shell_dim, up_out, up_inp = get_spin_basis_vals(basis_index, spin_num_mod_2 % 2)
    return get_spin_basis_index(shell_dim, up_inp, up_out)


def get_adjoint(spin_op: scipy.sparse.spmatrix) -> scipy.sparse.spmatrix:
    """Compute an adjoint of a left-acting operator.

    For a given superoperator `O` that acts on a density matrix `rho` as `O(rho) = M rho`, compute
    the adjoint `Q` that acts on `p` as `Q(rho) = rho M^dag`.
    """
    spin_num = get_spin_num(spin_op.shape[0])
    new_op = scipy.sparse.dok_matrix(spin_op.shape, dtype=spin_op.dtype)
    for idx_out, idx_inp in zip(*spin_op.nonzero()):
        idx_out_T = get_transpose_basis_index(idx_out, spin_num % 2)
        idx_inp_T = get_transpose_basis_index(idx_inp, spin_num % 2)
        new_op[idx_out_T, idx_inp_T] = spin_op[idx_out, idx_inp].conj()
    return new_op.asformat(spin_op.getformat())


def get_Sz_L(spin_num: int) -> scipy.sparse.spmatrix:
    """Compute the superoporator `O` that acts on a density matrix `rho` as `O(rho) = S_z @ rho`."""
    vals = [
        up_out - (shell_dim - 1) / 2 for _, shell_dim, up_out, _ in spin_op_basis_elements(spin_num)
    ]
    shape = (len(vals),) * 2
    return scipy.sparse.dia_matrix(([vals], [0]), shape)


def get_Sp_L(spin_num: int, _invert: bool = False) -> scipy.sparse.spmatrix:
    """Compute the superoporator `O` that acts on density matrix `rho` as `O(rho) = S_p rho`."""
    sign = 1 if not _invert else -1
    dim = get_spin_op_dim(spin_num)
    mat = scipy.sparse.dok_matrix((dim, dim), dtype=float)
    for idx_inp, shell_dim, up_out, up_inp in spin_op_basis_elements(spin_num):
        new_up_out = up_out + sign
        if not 0 <= new_up_out < shell_dim:
            continue
        idx_out = get_spin_basis_index(shell_dim, new_up_out, up_inp)
        spin_val = (shell_dim - 1) / 2
        proj_val = up_out - spin_val
        mat_val = np.sqrt(spin_val * (spin_val + 1) - proj_val * (proj_val + sign))
        mat[idx_out, idx_inp] = mat_val
    return mat


def get_Sm_L(spin_num: int) -> scipy.sparse.spmatrix:
    """Compute the superoporator `O` that acts on density matrix `rho` as `O(rho) = S_m rho`."""
    return get_Sp_L(spin_num, _invert=True)


def get_S(spin_num: int) -> scipy.sparse.spmatrix:
    """Compute the superoporator `O` that acts on density matrix `rho` as `O(rho) = S rho`."""
    vals = [(shell_dim - 1) / 2 for _, shell_dim, _, _ in spin_op_basis_elements(spin_num)]
    shape = (len(vals),) * 2
    return scipy.sparse.dia_matrix(([vals], [0]), shape)


def get_SS(spin_num: int) -> scipy.sparse.spmatrix:
    """Compute the superoporator `O` that acts on density matrix `rho` as `O(rho) = S (S+1) rho`."""
    op_S = get_S(spin_num)
    op_I = scipy.sparse.identity(op_S.shape[0])
    return op_S @ (op_S + op_I)
