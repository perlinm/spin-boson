from typing import Sequence

import numpy as np
import qutip
import scipy

import methods
import methods_PS
import spin_ops

MAX_NUM_SPINS = 4


def get_collective_ops_qutip(num_spins: int) -> tuple[qutip.Qobj, qutip.Qobj, qutip.Qobj]:
    return (
        methods.collective_qubit_op(qutip.sigmax(), num_spins) / 2,
        methods.collective_qubit_op(qutip.sigmay(), num_spins) / 2,
        methods.collective_qubit_op(qutip.sigmaz(), num_spins) / 2,
    )


def get_collective_ops_PS(
    num_spins: int,
) -> tuple[scipy.sparse.spmatrix, scipy.sparse.spmatrix, scipy.sparse.spmatrix]:
    op_Sz = spin_ops.get_Sz_L(num_spins)
    op_Sp = spin_ops.get_Sp_L(num_spins)
    op_Sm = spin_ops.get_Sm_L(num_spins)
    op_Sx = (op_Sp + op_Sm) / 2
    op_Sy = (op_Sp - op_Sm) / 2j
    return op_Sx, op_Sy, op_Sz


def get_jump_ops_qutip(num_spins: int, kraus_vec: tuple[float, float, float]) -> tuple[qutip.Qobj, ...]:
    return [
        np.sqrt(coef) * methods.act_on(op, spin, num_spins)
        for spin in range(num_spins)
        for coef, op in zip(kraus_vec, [qutip.sigmaz(), qutip.sigmap(), qutip.sigmam()])
    ]


def get_dissipator_PS(num_spins: int, kraus_vec: tuple[float, float, float]) -> scipy.sparse.spmatrix:
    return sum(
        coef * spin_ops.get_local_dissipator(num_spins, op)
        for coef, op in zip(kraus_vec, ["z", "+", "-"])
    )


def test_spin_evolution() -> None:
    times = np.linspace(0, 5, 20)
    ham_vec = np.random.random(3)
    kraus_vec = tuple(np.random.random(3))
    kraus_vec = (0, 0, 0)

    for num_spins in range(1, MAX_NUM_SPINS):
        # simulate with qutip
        initial_state = qutip.ket2dm(methods.get_all_down_state(num_spins))
        collective_ops = get_collective_ops_qutip(num_spins)
        hamiltonian = sum(coef * op for coef, op in zip(ham_vec, collective_ops))
        jump_ops = get_jump_ops_qutip(num_spins, kraus_vec)
        states = methods.get_states(times, initial_state, jump_ops, hamiltonian)
        vals = [
            state.conj().ravel() @ collective_op.data.toarray().ravel()
            for state in states
            for collective_op in collective_ops
        ]

        # simulate with PS methods
        initial_state = spin_ops.get_spin_vacuum(num_spins)
        collective_ops = get_collective_ops_PS(num_spins)
        hamiltonian = sum(coef * op for coef, op in zip(ham_vec, collective_ops))
        dissipator = get_dissipator_PS(num_spins, kraus_vec)
        generator = -1j * (hamiltonian - spin_ops.get_dual(hamiltonian)) + dissipator
        states = methods_PS.get_states(times, initial_state, generator)
        vals_PS = [
            spin_ops.get_spin_trace(collective_op @ state)
            for state in states
            for collective_op in collective_ops
        ]

        assert np.allclose(vals, vals_PS)


def test_spin_boson_evolution() -> None:
    ...
