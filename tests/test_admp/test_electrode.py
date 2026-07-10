"""Tests for the polarizable electrode (constant potential / constant charge).

Reference data are LAMMPS ``fix electrode`` single-point calculations
(units metal: eV, angstrom) in tests/data/electrode. Naming convention:

- 2D: boundary p p f, slab correction on, no finite field
- 3D: boundary p p p, no slab correction, finite field for conp
"""

import functools

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
from ase import io

from dmff.admp.electrode import (
    LAMMPSElectrodeConstraint,
    PolarizableElectrode,
    infer,
    setup_from_lammps,
)
from dmff.api import DMFFTopology, Hamiltonian
from dmff.common.constants import EV2KJ
from dmff.common.nblist import NeighborListFreud
from dmff.utils import pair_buffer_scales

RCUT = 5.0  # A
KAPPA = 0.5  # A^-1
ETA_LMP = 1.6  # A^-1
ETHRESH = 1e-6
SLAB_FACTOR = 3.0

BOTTOM = np.arange(108)
TOP = np.arange(108, 216)
ALL_ELEC = np.arange(216)

# dataset -> (slab_corr, symm, tolerance, constraints)
CASES = {
    "lmp_conp_slab_2d": (
        True, True, 5e-5,
        [
            LAMMPSElectrodeConstraint(BOTTOM, "conp", 20.0, ETA_LMP),
            LAMMPSElectrodeConstraint(TOP, "conp", 0.0, ETA_LMP),
        ],
    ),
    "lmp_conp_slab_3d": (
        False, True, 5e-5,
        [
            LAMMPSElectrodeConstraint(BOTTOM, "conp", 20.0, ETA_LMP, ffield=True),
            LAMMPSElectrodeConstraint(TOP, "conp", 0.0, ETA_LMP, ffield=True),
        ],
    ),
    "lmp_conp_interface_2d_pzc": (
        True, True, 5e-4,
        [
            LAMMPSElectrodeConstraint(BOTTOM, "conp", 0.0, ETA_LMP),
            LAMMPSElectrodeConstraint(TOP, "conp", 0.0, ETA_LMP),
        ],
    ),
    "lmp_conp_interface_2d_bias": (
        True, True, 5e-4,
        [
            LAMMPSElectrodeConstraint(BOTTOM, "conp", 20.0, ETA_LMP),
            LAMMPSElectrodeConstraint(TOP, "conp", 0.0, ETA_LMP),
        ],
    ),
    "lmp_conp_interface_3d_pzc": (
        False, True, 5e-4,
        [
            LAMMPSElectrodeConstraint(BOTTOM, "conp", 0.0, ETA_LMP, ffield=True),
            LAMMPSElectrodeConstraint(TOP, "conp", 0.0, ETA_LMP, ffield=True),
        ],
    ),
    "lmp_conp_interface_3d_bias": (
        False, True, 5e-4,
        [
            LAMMPSElectrodeConstraint(BOTTOM, "conp", 20.0, ETA_LMP, ffield=True),
            LAMMPSElectrodeConstraint(TOP, "conp", 0.0, ETA_LMP, ffield=True),
        ],
    ),
    "lmp_conq_interface_2d_pzc": (
        True, False, 5e-4,
        [LAMMPSElectrodeConstraint(ALL_ELEC, "conq", 0.0, ETA_LMP)],
    ),
    "lmp_conq_interface_2d_edl": (
        True, False, 5e-4,
        [LAMMPSElectrodeConstraint(ALL_ELEC, "conq", -10.0, ETA_LMP)],
    ),
    "lmp_conq_interface_2d_bias": (
        True, False, 5e-4,
        [
            LAMMPSElectrodeConstraint(BOTTOM, "conq", -10.0, ETA_LMP),
            LAMMPSElectrodeConstraint(TOP, "conq", 10.0, ETA_LMP),
        ],
    ),
    "lmp_conq_interface_3d_pzc": (
        False, False, 5e-4,
        [LAMMPSElectrodeConstraint(ALL_ELEC, "conq", 0.0, ETA_LMP)],
    ),
    "lmp_conq_interface_3d_edl": (
        False, False, 5e-4,
        [LAMMPSElectrodeConstraint(ALL_ELEC, "conq", -10.0, ETA_LMP)],
    ),
}


@functools.lru_cache(maxsize=None)
def load_case(name, slab_corr):
    """Load a LAMMPS reference frame and build the pair list (angstrom)."""
    atoms = io.read(f"tests/data/electrode/{name}/dump.lammpstrj")
    cell = atoms.cell.array.copy()
    if slab_corr:
        cell[2, 2] *= SLAB_FACTOR
    # dumps store unwrapped coordinates; wrap into [0, L) like LAMMPS.
    # Absolute coordinates matter for the slab correction / finite field.
    pos = atoms.get_positions()
    frac = pos @ np.linalg.inv(cell)
    positions = (frac - np.floor(frac)) @ cell

    n = len(atoms)
    nblist = NeighborListFreud(cell, RCUT, jnp.zeros((n, n), dtype=int))
    nblist.allocate(positions)
    pairs = jnp.array(nblist.pairs)
    return (
        jnp.array(positions),
        jnp.array(cell),
        pairs,
        jnp.array(atoms.get_initial_charges()),
        atoms.get_forces(),
    )


@pytest.mark.parametrize("name", list(CASES))
def test_electrode_vs_lammps(name):
    """Optimized charges and forces must match LAMMPS fix electrode."""
    slab_corr, symm, tol, constraints = CASES[name]
    positions, box, pairs, ref_q, ref_forces = load_case(name, slab_corr)
    n = positions.shape[0]

    setup = setup_from_lammps(n, constraints, symm)
    calc = PolarizableElectrode(
        rcut=RCUT, box=np.array(box), ethresh=ETHRESH, kappa=KAPPA,
        slab_corr=slab_corr, eps=1e-6,
    )
    # start the electrode charges from scratch
    charges = ref_q.at[setup.elec_idx].set(0.0)

    for method in ["matinv", "lbfgs"]:
        energy, forces, q_opt = infer(
            calc, positions, box, charges, pairs, setup, method=method
        )
        # forces [eV/A] against LAMMPS
        np.testing.assert_allclose(
            np.array(forces) / EV2KJ, ref_forces, atol=tol, rtol=tol
        )
        # optimized charges [e] against LAMMPS
        np.testing.assert_allclose(
            np.array(q_opt), np.array(ref_q), atol=1e-3, rtol=1e-3
        )


@pytest.mark.parametrize(
    "name", ["lmp_conp_slab_2d", "lmp_conp_interface_2d_pzc"]
)
def test_fixed_charge_coulomb(name):
    """Coulomb core check: forces at the LAMMPS-converged charges."""
    slab_corr, symm, tol, constraints = CASES[name]
    positions, box, pairs, ref_q, ref_forces = load_case(name, slab_corr)
    n = positions.shape[0]

    setup = setup_from_lammps(n, constraints, symm)
    calc = PolarizableElectrode(
        rcut=RCUT, box=np.array(box), ethresh=ETHRESH, kappa=KAPPA,
        slab_corr=slab_corr,
    )
    buffer_scales = pair_buffer_scales(pairs[:, :2])
    energy, forces = calc.coulomb_calculator(
        positions, box, ref_q, setup.eta, pairs, buffer_scales
    )
    np.testing.assert_allclose(
        np.array(forces) / EV2KJ, ref_forces, atol=tol, rtol=tol
    )


def test_setup_errors():
    with pytest.raises(AttributeError, match="symm should be False for conq"):
        setup_from_lammps(
            10, [LAMMPSElectrodeConstraint(np.arange(5), "conq", 0.0, ETA_LMP)], True
        )
    with pytest.raises(AttributeError, match="ffield with conq"):
        setup_from_lammps(
            10,
            [LAMMPSElectrodeConstraint(np.arange(5), "conq", 0.0, ETA_LMP, ffield=True)],
        )
    with pytest.raises(AttributeError, match="number of ffield group"):
        setup_from_lammps(
            10,
            [LAMMPSElectrodeConstraint(np.arange(5), "conp", 0.0, ETA_LMP, ffield=True)],
        )


def test_ffield_slab_corr_error():
    slab_corr, symm, tol, constraints = CASES["lmp_conp_slab_3d"]
    positions, box, pairs, ref_q, _ = load_case("lmp_conp_slab_3d", slab_corr)
    setup = setup_from_lammps(positions.shape[0], constraints, symm)
    calc = PolarizableElectrode(
        rcut=RCUT, box=np.array(box), ethresh=ETHRESH, kappa=KAPPA, slab_corr=True
    )
    with pytest.raises(ValueError, match="Slab correction and finite field"):
        infer(calc, positions, box, ref_q, pairs, setup)


def test_frontend_generator():
    """The XML frontend must reproduce the backend result for conp_slab_2d."""
    name = "lmp_conp_slab_2d"
    slab_corr, symm, tol, constraints = CASES[name]
    positions, box, pairs, ref_q, ref_forces = load_case(name, slab_corr)
    n = positions.shape[0]

    top = DMFFTopology()
    chain = top.addChain()
    res = top.addResidue("ELE", chain)
    for i in range(n):
        top.addAtom(f"Pt{i}", app.element.platinum, res)
    for a in top.atoms():
        a.meta["charge"] = 0.0
        a.meta["type"] = "Pt"
    top.setPeriodicBoxVectors(np.array(box) * 0.1)  # nm

    hamilt = Hamiltonian("tests/data/electrode/electrode.xml")
    pot = hamilt.createPotential(
        top,
        nonbondedMethod=app.PME,
        nonbondedCutoff=RCUT * 0.1 * unit.nanometer,
        electrode_constraints=constraints,
        symm=symm,
        slab_corr=slab_corr,
        kappa=KAPPA,
        ethresh=ETHRESH,
        has_aux=True,
    )
    efunc = pot.getPotentialFunc()

    pos_nm = positions * 0.1
    box_nm = box * 0.1
    energy, aux = efunc(pos_nm, box_nm, pairs, hamilt.paramset.parameters, aux={})

    # optimized charges against LAMMPS
    np.testing.assert_allclose(np.array(aux["q"]), np.array(ref_q), atol=1e-3)

    # forces via jax.grad against LAMMPS (kJ/mol/nm -> eV/A)
    grad_fn = jax.grad(efunc, argnums=0, has_aux=True)
    gradient, _ = grad_fn(pos_nm, box_nm, pairs, hamilt.paramset.parameters, aux={})
    forces_ev = -np.array(gradient) / EV2KJ / 10.0
    np.testing.assert_allclose(forces_ev, ref_forces, atol=tol, rtol=tol)
