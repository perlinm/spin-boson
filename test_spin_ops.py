import itertools

import numpy as np
import scipy

import spin_ops


def test_spin_indexing() -> None:
    for spin_num in range(5):
        for shell_dim in range(spin_num % 2 + 1, spin_num + 2, 2):
            for up_out, up_inp in itertools.product(range(shell_dim), repeat=2):
                basis_index = spin_ops.get_spin_basis_index(shell_dim, up_out, up_inp)
                basis_vals = spin_ops.get_spin_basis_vals(basis_index, spin_num % 2)
                assert (shell_dim, up_out, up_inp) == basis_vals

        dim = spin_ops.get_spin_op_dim(spin_num)
        assert spin_num == spin_ops.get_spin_num(dim)


def test_collective_ops() -> None:
    for spin_num in range(5):
        dim = spin_ops.get_spin_op_dim(spin_num)
        op_Sz_L = spin_ops.get_Sz_L(spin_num).todok()
        op_Sz_R = spin_ops.get_adjoint(op_Sz_L)

        # check correctness of left- and right-acting S_z operators
        for index in range(dim):
            shell_dim, up_out, up_inp = spin_ops.get_spin_basis_vals(index, spin_num % 2)
            assert op_Sz_L[index, index] == up_out - (shell_dim - 1) / 2
            assert op_Sz_R[index, index] == up_inp - (shell_dim - 1) / 2

        # check that S_+ @ S_- = S(S+1) - S_z^2 + S_z

        op_Sp_L = spin_ops.get_Sp_L(spin_num)
        op_Sm_L = spin_ops.get_Sm_L(spin_num)
        op_SS = spin_ops.get_SS(spin_num)

        op_Sp_Sm_L_1 = op_Sp_L @ op_Sm_L
        op_Sp_Sm_L_2 = op_SS - op_Sz_L @ op_Sz_L + op_Sz_L
        assert np.allclose(op_Sp_Sm_L_1.diagonal(), op_Sp_Sm_L_2.diagonal())

        op_Sp_R = spin_ops.get_adjoint(op_Sp_L)
        op_Sm_R = spin_ops.get_adjoint(op_Sm_L)

        op_Sp_Sm_R_1 = op_Sp_R @ op_Sm_R
        op_Sp_Sm_R_2 = op_SS - op_Sz_R @ op_Sz_R + op_Sz_R
        assert np.allclose(op_Sp_Sm_R_1.diagonal(), op_Sp_Sm_R_2.diagonal())
