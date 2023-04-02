from typing import Literal, Sequence

import numpy as np
import qutip
import scipy

import methods
import methods_PS
import spin_ops

MAX_NUM_SPINS = 4


def get_collective_ops_qutip(num_spins: int) -> tuple[qutip.Qobj, qutip.Qobj, qutip.Qobj]:
    return (
        methods.collective_qubit_op(qutip.sigmaz(), num_spins) / 2,
        methods.collective_qubit_op(qutip.sigmax(), num_spins) / 2,
        methods.collective_qubit_op(qutip.sigmay(), num_spins) / 2,
    )


def get_collective_ops_PS(
    num_spins: int,
) -> tuple[scipy.sparse.spmatrix, scipy.sparse.spmatrix, scipy.sparse.spmatrix]:
    op_Sz = spin_ops.get_Sz_L(num_spins)
    op_Sp = spin_ops.get_Sp_L(num_spins)
    op_Sm = spin_ops.get_Sm_L(num_spins)
    op_Sx = (op_Sp + op_Sm) / 2
    op_Sy = (op_Sp - op_Sm) / 2j
    return op_Sz, op_Sx, op_Sy


def get_jump_ops_qutip(
    num_spins: int, kraus_vec: tuple[float, float, float]
) -> tuple[qutip.Qobj, ...]:
    ops = [qutip.sigmaz() / 2, qutip.sigmap(), qutip.sigmam()]
    return tuple(
        np.sqrt(coef) * methods.act_on(op, spin, num_spins)
        for spin in range(num_spins)
        for coef, op in zip(kraus_vec, ops)
    )


def get_dissipator_PS(
    num_spins: int, kraus_vec: tuple[float, float, float]
) -> scipy.sparse.spmatrix:
    ops: Sequence[Literal["z", "+", "-"]] = ("z", "+", "-")
    return sum(
        coef * spin_ops.get_local_dissipator(num_spins, op) for coef, op in zip(kraus_vec, ops)
    )


def test_spin_evolution() -> None:
    times = np.linspace(0, 5, 20)
    ham_vec = np.random.random(3)
    kraus_vec = tuple(np.random.random(3))

    ham_vec = (0, 1, 0)
    kraus_vec = (0, 1, 0)

    # for num_spins in range(1, MAX_NUM_SPINS):
    for num_spins in range(2, 3):

        # simulate with qutip
        initial_state = qutip.ket2dm(methods.get_all_down_state(num_spins))
        collective_ops = get_collective_ops_qutip(num_spins)
        hamiltonian = sum(coef * op for coef, op in zip(ham_vec, collective_ops))
        jump_ops = get_jump_ops_qutip(num_spins, kraus_vec)
        states = methods.get_states(times, initial_state, hamiltonian, jump_ops)
        vals = [
            (collective_op.data.conj().toarray().ravel() @ state.ravel()).real
            for collective_op in collective_ops
            for state in states
        ]

        # simulate with PS methods
        initial_state = spin_ops.get_spin_vacuum(num_spins)
        collective_ops = get_collective_ops_PS(num_spins)
        hamiltonian = sum(coef * op for coef, op in zip(ham_vec, collective_ops))
        dissipator = get_dissipator_PS(num_spins, kraus_vec)
        generator = -1j * (hamiltonian - spin_ops.get_dual(hamiltonian)) + dissipator
        states = methods_PS.get_states(times, initial_state, generator)
        vals_PS = [
            spin_ops.get_spin_trace(collective_op @ state).real
            for collective_op in collective_ops
            for state in states
        ]

        import itertools

        np.set_printoptions(linewidth=200)
        spin_op_dim = initial_state.size
        print()
        print()
        print("dissipator")
        print()
        for idx_inp, idx_out in itertools.product(range(spin_op_dim), repeat=2):
            if abs(dissipator[idx_out, idx_inp]) > 1e-3:
                vals_out = spin_ops.get_spin_basis_vals(idx_out, num_spins % 2)
                vals_inp = spin_ops.get_spin_basis_vals(idx_inp, num_spins % 2)
                print(vals_out, vals_inp, dissipator[idx_out, idx_inp])


        print()
        print()
        print("actual dissipator")
        print()

        def spin_state(shell_dim: int, num_up: int) -> np.ndarray:
            ket_up = qutip.fock(2, 0)
            ket_dn = qutip.fock(2, 1)
            if num_spins == 1:
                return ket_up if num_up == 1 else ket_dn
            if num_spins == 2:
                if shell_dim == 1:
                    return (np.kron(ket_up, ket_dn) - np.kron(ket_dn, ket_up)) / np.sqrt(2)
                if num_up == 0:
                    return (np.kron(ket_up, ket_dn) + np.kron(ket_dn, ket_up)) / np.sqrt(2)
                if num_up == 0:
                    return np.kron(ket_dn, ket_dn)
                return np.kron(ket_up, ket_up)

        def density_op(shell_dim: int, up_out: int, up_inp: int) -> np.ndarray:
            return np.outer(spin_state(shell_dim, up_out), spin_state(shell_dim, up_inp))

        def act_on(op: qutip.Qobj, target_index: int, num_spins: int) -> qutip.Qobj:
            ops = [qutip.qeye(2)] * num_spins
            ops[target_index] = op
            return qutip.tensor(*ops)

        ops = [qutip.sigmaz() / 2, qutip.sigmap(), qutip.sigmam()]
        jump_ops = tuple(
            np.sqrt(coef) * act_on(op, spin, num_spins)
            for spin in range(num_spins)
            for coef, op in zip(kraus_vec, ops)
        )
        dissipator = sum(
            methods_PS.to_dissipation_generator(jump_op.data) for jump_op in jump_ops
        )
        for idx_inp in range(spin_op_dim):
            vals_inp = spin_ops.get_spin_basis_vals(idx_inp, num_spins % 2)
            rho_inp = density_op(*vals_inp).ravel()
            rho_out = dissipator @ rho_inp
            for idx_out in range(spin_op_dim):
                vals_out = spin_ops.get_spin_basis_vals(idx_out, num_spins % 2)
                val = density_op(*vals_out).ravel() @ rho_out
                if np.isclose(val.imag, 0):
                    val = val.real
                if abs(val) > 1e-3:
                    print(vals_out, vals_inp, val)

        break

        final_state = states[-1]
        print()
        print()
        print()
        print("final state")
        print()
        for index in range(spin_op_dim):
            print(spin_ops.get_spin_basis_vals(index, num_spins % 2), final_state[index])
        print()

        import matplotlib.pyplot as plt

        plt.plot(times, vals[: len(times)])
        plt.plot(times, vals_PS[: len(times)], "--")
        plt.tight_layout()
        plt.show()

        if not np.allclose(vals, vals_PS):
            print()
            print()
            print("FAIL!")
            print("num_spins:", num_spins)
            break

        # assert np.allclose(vals, vals_PS)


def test_spin_boson_evolution() -> None:
    ...
