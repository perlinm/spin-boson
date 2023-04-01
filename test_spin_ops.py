import itertools

import numpy as np

import spin_ops

MAX_NUM_SPINS = 4


def test_spin_indexing() -> None:
    for num_spins in range(MAX_NUM_SPINS + 1):
        for shell_dim in range(num_spins % 2 + 1, num_spins + 2, 2):
            for up_out, up_inp in itertools.product(range(shell_dim), repeat=2):
                basis_index = spin_ops.get_spin_basis_index(shell_dim, up_out, up_inp)
                basis_vals = spin_ops.get_spin_basis_vals(basis_index, num_spins % 2)
                assert (shell_dim, up_out, up_inp) == basis_vals

        dim = spin_ops.get_spin_op_dim(num_spins)
        assert num_spins == spin_ops.get_num_spins(dim)


def test_collective_ops() -> None:
    for num_spins in range(MAX_NUM_SPINS + 1):
        dim = spin_ops.get_spin_op_dim(num_spins)
        op_Sz_L = spin_ops.get_Sz_L(num_spins).todok()
        op_Sz_R = spin_ops.get_dual(op_Sz_L)

        # check correctness of left- and right-acting S_z operators
        for index in range(dim):
            shell_dim, up_out, up_inp = spin_ops.get_spin_basis_vals(index, num_spins % 2)
            assert op_Sz_L[index, index] == up_out - (shell_dim - 1) / 2
            assert op_Sz_R[index, index] == up_inp - (shell_dim - 1) / 2

        # check that S_+ @ S_- = S(S+1) - S_z^2 + S_z

        op_Sp_L = spin_ops.get_Sp_L(num_spins)
        op_Sm_L = spin_ops.get_Sm_L(num_spins)
        op_SS = spin_ops.get_SS(num_spins)

        op_Sp_Sm_L_1 = op_Sp_L @ op_Sm_L
        op_Sp_Sm_L_2 = op_SS - op_Sz_L @ op_Sz_L + op_Sz_L
        assert np.allclose(op_Sp_Sm_L_1.diagonal(), op_Sp_Sm_L_2.diagonal())

        op_Sp_R = spin_ops.get_dual(op_Sp_L)
        op_Sm_R = spin_ops.get_dual(op_Sm_L)

        op_Sp_Sm_R_1 = op_Sm_R @ op_Sp_R
        op_Sp_Sm_R_2 = op_SS - op_Sz_R @ op_Sz_R + op_Sz_R
        assert np.allclose(op_Sp_Sm_R_1.diagonal(), op_Sp_Sm_R_2.diagonal())


def test_spin_states() -> None:
    for num_spins in range(MAX_NUM_SPINS + 1):

        # construct left- and rigt-acting S_z
        op_Sz_L = spin_ops.get_Sz_L(num_spins)
        op_Sz_R = spin_ops.get_dual(op_Sz_L)

        # construct an operator that acts with S_+ on the left and S_- on the right
        op_Sp_L = spin_ops.get_Sp_L(num_spins)
        op_Sp = spin_ops.get_dual(op_Sp_L.T) @ op_Sp_L

        for num_excitations in range(num_spins + 1):
            spin_val = num_spins / 2
            proj_val = num_excitations - num_spins / 2

            # test that the Dicke states transform appropriately under S_z
            state = spin_ops.get_dicke_state(num_spins, num_excitations)
            assert np.allclose(op_Sz_L @ state, proj_val * state)
            assert np.allclose(op_Sz_R @ state, proj_val * state)

            # test that the Dicke states transform appropriately under S_+
            excited_state = op_Sp @ state
            exact_excited_state = spin_ops.get_dicke_state(num_spins, num_excitations + 1)
            scalar = spin_val * (spin_val + 1) - proj_val * (proj_val + 1)
            assert np.allclose(excited_state, scalar * exact_excited_state)

        # test that the GHZ state has <S_z> = 0 and <S_z^2> = S^2
        state = spin_ops.get_ghz_state(num_spins)
        assert np.isclose(state @ op_Sz_L @ state, 0)
        assert np.isclose(state @ op_Sz_L @ op_Sz_L @ state, (num_spins / 2) ** 2)
