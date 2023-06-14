"""
Contents: Test file for methods.py and methods_PS.py.
Author: Michael A. Perlin (2023)
"""
from typing import Literal, Optional, Sequence

import numpy as np
import qutip
import scipy

import methods
import methods_PS
import spin_ops

MAX_NUM_SPINS = 5


def get_collective_ops_qutip(
    num_spins: int, *, boson_dim: Optional[int] = None
) -> tuple[qutip.Qobj, ...]:
    ops = (
        methods.collective_qubit_op(qutip.sigmaz(), num_spins) / 2,
        methods.collective_qubit_op(qutip.sigmax(), num_spins) / 2,
        methods.collective_qubit_op(qutip.sigmay(), num_spins) / 2,
    )
    if boson_dim is None:
        return ops
    return tuple(qutip.tensor(op, qutip.qeye(boson_dim)) for op in ops)


def get_collective_ops_PS(
    num_spins: int, *, boson_dim: Optional[int] = None
) -> tuple[scipy.sparse.spmatrix, ...]:
    op_Sz = spin_ops.get_Sz_L(num_spins)
    op_Sp = spin_ops.get_Sp_L(num_spins)
    op_Sm = spin_ops.get_Sm_L(num_spins)
    op_Sx = (op_Sp + op_Sm) / 2
    op_Sy = (op_Sp - op_Sm) / 2j
    ops = (op_Sz, op_Sx, op_Sy)
    if boson_dim is None:
        return ops
    return tuple(scipy.sparse.kron(op, scipy.sparse.identity(boson_dim**2)) for op in ops)


def get_jump_ops_qutip(
    num_spins: int, kraus_vec: Sequence[float] | np.ndarray
) -> tuple[qutip.Qobj, ...]:
    ops = [qutip.sigmaz() / 2, qutip.sigmap(), qutip.sigmam()]
    return tuple(
        np.sqrt(coef) * methods.act_on(op, spin, num_spins)
        for spin in range(num_spins)
        for coef, op in zip(kraus_vec, ops)
    )


def get_dissipator_PS(
    num_spins: int, kraus_vec: Sequence[float] | np.ndarray
) -> scipy.sparse.spmatrix:
    ops: Sequence[Literal["z", "+", "-"]] = ("z", "+", "-")
    return sum(
        coef * spin_ops.get_local_dissipator(num_spins, op) for coef, op in zip(kraus_vec, ops)
    )


def test_spin_evolution() -> None:
    times = np.linspace(0, 5, 20)
    ham_vec = np.random.random(3)
    kraus_vec = np.random.random(3)

    for num_spins in range(1, MAX_NUM_SPINS):

        # simulate with qutip
        collective_ops = get_collective_ops_qutip(num_spins)
        initial_state = qutip.ket2dm(methods.get_spin_vacuum_state(num_spins))
        hamiltonian = sum(coef * op for coef, op in zip(ham_vec, collective_ops))
        jump_ops = get_jump_ops_qutip(num_spins, kraus_vec)
        states = methods.get_states(times, initial_state, hamiltonian, jump_ops)

        vals = [
            (collective_op.data.conj().toarray().ravel() @ state.ravel()).real
            for collective_op in collective_ops
            for state in states
        ]

        # simulate with PS methods
        collective_ops = get_collective_ops_PS(num_spins)
        initial_state = spin_ops.get_dicke_state(num_spins, 0)
        hamiltonian = sum(coef * op for coef, op in zip(ham_vec, collective_ops))
        dissipator = get_dissipator_PS(num_spins, kraus_vec)
        generator = -1j * (hamiltonian - spin_ops.get_dual(hamiltonian)) + dissipator
        states = methods_PS.get_states(times, initial_state, generator)

        vals_PS = [
            spin_ops.get_spin_trace(collective_op @ state).real
            for collective_op in collective_ops
            for state in states
        ]

        assert np.allclose(vals, vals_PS)


def test_spin_boson_evolution() -> None:
    times = np.linspace(0, 5, 20)
    splitting = np.random.random()
    coupling = np.random.random()
    decay_res = np.random.random()
    decay_spin = np.random.random()
    args = (splitting, coupling, decay_res, decay_spin)

    for num_spins in range(1, MAX_NUM_SPINS):
        boson_dim = num_spins + 1

        # simulate with qutip
        collective_ops = get_collective_ops_qutip(num_spins, boson_dim=boson_dim)
        initial_state = methods.get_ghz_state(num_spins)
        hamiltonian = methods.get_hamiltonian(num_spins, splitting, coupling)
        jump_ops = methods.get_jump_ops(num_spins, decay_res, decay_spin)
        states = methods.get_states(times, initial_state, hamiltonian, jump_ops)

        vals = [
            (collective_op.data.conj().toarray().ravel() @ state.ravel()).real
            for collective_op in collective_ops
            for state in states
        ]
        vals_QFI, vals_QFI_SA = methods.get_QFI_vals(times, num_spins, *args, initial_state)

        # simulate with PS methods
        collective_ops_PS = get_collective_ops_PS(num_spins, boson_dim=boson_dim)
        boson_vacuum = np.zeros(boson_dim**2)
        boson_vacuum[0] = 1
        initial_state = np.kron(spin_ops.get_ghz_state(num_spins), boson_vacuum)
        hamiltonian_generator = methods_PS.get_hamiltonian_generator(num_spins, splitting, coupling)
        dissipator = methods_PS.get_dissipator(num_spins, decay_res, decay_spin)
        generator = hamiltonian_generator + dissipator
        states = methods_PS.get_states(times, initial_state, generator)

        shape = (-1, boson_dim, boson_dim)
        vals_PS = [
            spin_ops.get_spin_trace((collective_op @ state).reshape(shape)).trace()
            for collective_op in collective_ops_PS
            for state in states
        ]
        vals_QFI_PS, vals_QFI_SA_PS = methods_PS.get_QFI_vals(
            times, num_spins, *args, initial_state
        )

        assert np.allclose(vals, vals_PS)
        assert np.allclose(vals_QFI, vals_QFI_PS, atol=1e-3)
        assert np.allclose(vals_QFI_SA, vals_QFI_SA_PS, atol=1e-3)
