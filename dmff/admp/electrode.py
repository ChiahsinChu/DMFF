"""
Polarizable electrode models for constant potential (CONP) and constant
charge (CONQ) simulations.

This module implements the LAMMPS ``fix electrode`` style polarizable
electrode method: electrode charges are equilibrated under conp/conq
constraints (optionally with finite-field boundary conditions), while
electrolyte charges stay fixed. It is a JAX port of the reference
implementation in torch-admp (torch_admp/electrode.py).

Internal units: length in angstrom, energy in kJ/mol, charge in e.
Potentials/electronegativities given in V (and hardness in V/e) are
converted with EV2KJ at setup time.
"""

from typing import List, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from ..common.constants import DIELECTRIC, EV2KJ
from ..utils import pair_buffer_scales
from .pme import energy_pme
from .qeq import E_sr3, E_site3
from .recip import Ck_1, generate_pme_recip

try:
    import jaxopt
except ImportError:
    jaxopt = None


class LAMMPSElectrodeConstraint:
    """
    Electrode constraint defined in the style of LAMMPS fix electrode

    Parameters
    ----------
    indices : Union[List[int], np.ndarray]
        indices of the atoms in constraint
    mode : str
        conp or conq
    value : float
        value of the constraint (potential [V] for conp, total charge [e] for conq)
    eta : float
        eta as used in LAMMPS (in length^-1)
    chi: float
        electronegativity [V], default 0.0 (single element)
    hardness: float
        atomic hardness [V/e], default 0.0
    ffield: bool
        if used as ffield group
    """

    def __init__(
        self,
        indices: Union[List[int], np.ndarray],
        mode: str,
        value: float,
        eta: float,
        chi: float = 0.0,
        hardness: float = 0.0,
        ffield: bool = False,
    ) -> None:
        self.indices = np.array(indices, dtype=int)
        assert self.indices.ndim == 1

        self.mode = mode
        assert mode in ["conp", "conq"], f"mode {mode} not supported"

        self.value = value
        self.eta = eta
        self.hardness = hardness
        self.chi = chi
        self.ffield = ffield


class ElectrodeSetup(NamedTuple):
    """Input data for the electrode charge optimization.

    All energetic quantities are in kJ/mol based units, lengths in angstrom.
    ``elec_idx`` is a static numpy array so that scatters keep static shapes
    under jit.
    """

    mask: jnp.ndarray  # (n_atoms,) bool
    elec_idx: np.ndarray  # (n_electrode,) int, static
    eta: jnp.ndarray  # (n_atoms,) Gaussian width [A], E_sr3 convention
    chi: jnp.ndarray  # (n_atoms,) electronegativity [kJ/mol/e]
    hardness: jnp.ndarray  # (n_atoms,) [kJ/mol/e^2]
    constraint_matrix: jnp.ndarray  # (n_const, n_electrode)
    constraint_vals: jnp.ndarray  # (n_const,) [e]
    ffield_electrode_mask: Optional[jnp.ndarray]  # (2, n_atoms) bool
    ffield_potential: Optional[jnp.ndarray]  # (2,) [kJ/mol/e]


def setup_from_lammps(
    n_atoms: int,
    constraint_list: List[LAMMPSElectrodeConstraint],
    symm: bool = False,
) -> ElectrodeSetup:
    """
    Generate input data based on lammps-like constraint definitions

    Note the eta convention: LAMMPS defines the Gaussian as
    rho(r) ~ exp(-eta_lmp^2 r^2), while E_sr3 uses erfc(r / sqrt(eta_i^2 +
    eta_j^2)), so eta here is 1 / eta_lmp (in angstrom).
    """
    mask = np.zeros(n_atoms, dtype=bool)

    eta = np.zeros(n_atoms)
    chi = np.zeros(n_atoms)
    hardness = np.zeros(n_atoms)

    constraint_matrix = []
    constraint_vals = []
    ffield_electrode_mask = []
    ffield_potential = []

    for constraint in constraint_list:
        mask[constraint.indices] = True
        # eta = 0 means "defined elsewhere" (e.g. the XML per-type value
        # in the frontend generator)
        eta[constraint.indices] = 1.0 / constraint.eta if constraint.eta > 0 else 0.0
        chi[constraint.indices] = constraint.chi * EV2KJ
        hardness[constraint.indices] = constraint.hardness * EV2KJ
        if constraint.mode == "conq":
            if symm:
                raise AttributeError(
                    "symm should be False for conq, user can implement symm by conq"
                )
            if constraint.ffield:
                raise AttributeError("ffield with conq has not been implemented yet")
            constraint_matrix.append(np.zeros((1, n_atoms)))
            constraint_matrix[-1][0, constraint.indices] = 1.0
            constraint_vals.append(constraint.value)
        if constraint.mode == "conp":
            chi[constraint.indices] -= constraint.value * EV2KJ
        if constraint.ffield:
            ffield_electrode_mask.append(np.zeros((1, n_atoms)))
            ffield_electrode_mask[-1][0, constraint.indices] = 1.0
            ffield_potential.append(constraint.value * EV2KJ)

    if len(ffield_electrode_mask) == 0:
        ffield_electrode_mask = None
        ffield_potential = None
    elif len(ffield_electrode_mask) == 2:
        ffield_electrode_mask = jnp.array(
            np.concatenate(ffield_electrode_mask, axis=0), dtype=bool
        )
        ffield_potential = jnp.array(np.array(ffield_potential))
    else:
        raise AttributeError("number of ffield group should be 0 or 2")

    if symm:
        constraint_matrix.append(np.ones((1, n_atoms)))
        constraint_vals.append(0.0)

    if len(constraint_matrix) > 0:
        constraint_matrix = jnp.array(
            np.concatenate(constraint_matrix, axis=0)[:, mask]
        )
        constraint_vals = jnp.array(np.array(constraint_vals))
    else:
        n_electrode = int(mask.sum())
        constraint_matrix = jnp.zeros((0, n_electrode))
        constraint_vals = jnp.zeros(0)

    return ElectrodeSetup(
        mask=jnp.array(mask),
        elec_idx=np.where(mask)[0],
        eta=jnp.array(eta),
        chi=jnp.array(chi),
        hardness=jnp.array(hardness),
        constraint_matrix=constraint_matrix,
        constraint_vals=constraint_vals,
        ffield_electrode_mask=ffield_electrode_mask,
        ffield_potential=ffield_potential,
    )


def finite_field_add_chi(
    positions: jnp.ndarray,
    box: jnp.ndarray,
    ffield_electrode_mask: jnp.ndarray,
    ffield_potential: jnp.ndarray,
    slab_axis: int = 2,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the correction term for the finite field

    potential need to be same in the electrode_mask
    potential drop is potential[0] - potential[1]

    Returns the potential correction for all atoms (only electrode entries
    are meaningful) and the electric field strength along slab_axis.
    """
    assert positions.ndim == 2
    assert box.ndim == 2
    assert ffield_potential.shape[0] == 2
    assert ffield_electrode_mask.shape[0] == 2
    assert ffield_electrode_mask.shape[1] == positions.shape[0]

    first_electrode_mask = ffield_electrode_mask[0]
    second_electrode_mask = ffield_electrode_mask[1]

    potential_drop = ffield_potential[0] - ffield_potential[1]

    # find max position in slab_axis for each electrode
    max_pos_first = jnp.max(
        jnp.where(first_electrode_mask, positions[:, slab_axis], -jnp.inf)
    )
    max_pos_second = jnp.max(
        jnp.where(second_electrode_mask, positions[:, slab_axis], -jnp.inf)
    )
    # only valid for orthogonality cell
    lz = box[slab_axis][slab_axis]
    normalized_positions = positions[:, slab_axis] / lz
    # lammps fix electrode implementation
    # cos180(-1) or cos0(1) for E(delta_psi/(r1-r2)) and r
    sign = jnp.where(max_pos_first > max_pos_second, 1.0, -1.0)
    potential = potential_drop * sign * normalized_positions
    efield = -sign * potential_drop / lz
    return potential, efield


class PolarizableElectrode:
    """Polarizable electrode calculator

    Energy model (in kJ/mol, angstrom): point-charge PME (real + reciprocal +
    self) + non-neutral background correction + optional Yeh-Berkowitz slab
    correction + Gaussian charge-smearing short-range correction (E_sr3).

    Parameters
    ----------
    rcut : float
        cutoff radius for short-range interactions [A]
    box : np.ndarray
        (3, 3) simulation box [A], used to determine the static k-mesh;
        pass the slab-extended box when slab_corr is used
    ethresh : float, optional
        energy threshold controlling Ewald accuracy, by default 1e-6
    kappa : float, optional
        Ewald splitting parameter [A^-1]; sqrt(-log(2 ethresh))/rcut if None
    kmesh : tuple, optional
        (K1, K2, K3) reciprocal mesh; derived from kappa/ethresh if None
    slab_corr : bool, optional
        enable slab correction, by default False
    slab_axis : int, optional
        axis of the slab, by default 2
    damping : bool, optional
        enable Gaussian smearing correction, by default True
    eps : float, optional
        convergence criterion for iterative solvers, in eV per atom to match
        the torch-admp convention, by default 1e-4
    max_iter : int, optional
        max iterations of the iterative solver, by default 100
    """

    def __init__(
        self,
        rcut: float,
        box: np.ndarray,
        ethresh: float = 1e-6,
        kappa: Optional[float] = None,
        kmesh: Optional[Tuple[int, int, int]] = None,
        slab_corr: bool = False,
        slab_axis: int = 2,
        damping: bool = True,
        eps: float = 1e-4,
        max_iter: int = 100,
    ) -> None:
        self.rcut = rcut
        self.ethresh = ethresh
        if kappa is None:
            kappa = np.sqrt(-np.log(2 * ethresh)) / rcut
        self.kappa = kappa
        if kmesh is None:
            box_diag = np.diagonal(np.array(box))
            kmesh = np.ceil(2 * kappa * box_diag / (3.0 * ethresh ** (1.0 / 5.0)))
            kmesh = tuple(int(k) for k in kmesh)
        self.kmesh = kmesh
        self.slab_corr = slab_corr
        self.slab_axis = slab_axis
        self.damping = damping
        self.eps = eps
        self.max_iter = max_iter

        self.pme_recip_fn = generate_pme_recip(
            Ck_fn=Ck_1,
            kappa=kappa,
            gamma=False,
            pme_order=6,
            K1=kmesh[0],
            K2=kmesh[1],
            K3=kmesh[2],
            lmax=0,
        )
        # no covalent exclusions between electrode/electrolyte atoms
        self.mscales = jnp.ones(6)

    def energy_fn(
        self,
        positions: jnp.ndarray,
        box: jnp.ndarray,
        pairs: jnp.ndarray,
        q: jnp.ndarray,
        eta: jnp.ndarray,
        chi: jnp.ndarray,
        hardness: jnp.ndarray,
        buffer_scales: jnp.ndarray,
    ) -> jnp.ndarray:
        """Total energy of the system [kJ/mol]; positions/box in angstrom."""
        e = E_site3(chi, hardness, q)
        e += energy_pme(
            positions,
            box,
            pairs,
            q.reshape(-1, 1),
            None,
            None,
            None,
            self.mscales,
            None,
            None,
            None,
            self.pme_recip_fn,
            self.kappa,
            self.kmesh[0],
            self.kmesh[1],
            self.kmesh[2],
            0,
            False,
        )
        e += e_non_neutral(q, box, self.kappa)
        if self.slab_corr:
            e += e_slab_corr(q, positions, box, self.slab_axis)
        if self.damping:
            e += E_sr3(positions, box, pairs, q, eta, buffer_scales, True)
        return e

    def calc_coulomb_potential(
        self,
        electrode_mask: Optional[jnp.ndarray],
        positions: jnp.ndarray,
        box: jnp.ndarray,
        eta: jnp.ndarray,
        charges: jnp.ndarray,
        pairs: jnp.ndarray,
        buffer_scales: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Per-atom electrostatic potential [kJ/mol/e] generated by the fixed
        (electrolyte) charges; electrode charges are zeroed out.
        """
        if electrode_mask is None:
            modified_charges = charges
        else:
            modified_charges = jnp.where(electrode_mask, 0.0, charges)
        zeros = jnp.zeros_like(charges)
        return jax.grad(self.energy_fn, argnums=3)(
            positions, box, pairs, modified_charges, eta, zeros, zeros, buffer_scales
        )

    def coulomb_calculator(
        self,
        positions: jnp.ndarray,
        box: jnp.ndarray,
        charges: jnp.ndarray,
        eta: jnp.ndarray,
        pairs: jnp.ndarray,
        buffer_scales: jnp.ndarray,
        efield: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Coulomb energy [kJ/mol] and forces [kJ/mol/A] for fixed charges,
        optionally in a uniform electric field along slab_axis.
        """
        zeros = jnp.zeros_like(charges)
        energy, pos_grads = jax.value_and_grad(self.energy_fn, argnums=0)(
            positions, box, pairs, charges, eta, zeros, zeros, buffer_scales
        )
        forces = -pos_grads

        if efield is not None:
            efield_vec = jnp.zeros(3).at[self.slab_axis].set(efield)
            forces = forces + charges[:, None] * efield_vec
            energy = energy + jnp.sum(
                efield_vec.reshape(1, 3) * charges[:, None] * positions
            )
        return energy, forces


def e_non_neutral(q: jnp.ndarray, box: jnp.ndarray, kappa: float) -> jnp.ndarray:
    """Background correction for non-neutral systems [kJ/mol]."""
    volume = jnp.linalg.det(box)
    q_tot = jnp.sum(q)
    return -jnp.pi / (2 * volume * kappa**2) * DIELECTRIC * q_tot**2


def e_slab_corr(
    q: jnp.ndarray, positions: jnp.ndarray, box: jnp.ndarray, slab_axis: int = 2
) -> jnp.ndarray:
    """Slab correction energy [kJ/mol] (ref: 10.1063/1.3216473)."""
    volume = jnp.linalg.det(box)
    z = positions[:, slab_axis]
    mz = jnp.sum(q * z)
    q_tot = jnp.sum(q)
    lz = jnp.linalg.norm(box[slab_axis])
    pre_corr = 2 * jnp.pi / volume * DIELECTRIC
    return pre_corr * (mz**2 - q_tot * jnp.sum(q * z**2) - q_tot**2 * lz**2 / 12)


def vector_projection(
    x: jnp.ndarray, constraint_matrix: jnp.ndarray, constraint_vals: jnp.ndarray
) -> jnp.ndarray:
    """Project x onto the affine subspace {x: A x = b}."""
    a_mat = constraint_matrix
    gram_inv = jnp.linalg.inv(a_mat @ a_mat.T)
    return x + a_mat.T @ (gram_inv @ (constraint_vals - a_mat @ x))


def charge_optimization(
    calculator: PolarizableElectrode,
    positions: jnp.ndarray,
    box: jnp.ndarray,
    charges: jnp.ndarray,
    pairs: jnp.ndarray,
    buffer_scales: jnp.ndarray,
    setup: ElectrodeSetup,
    method: str = "matinv",
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Optimize the electrode charges under the given constraints.

    Returns the optimized electrode charges (in elec_idx order) and the
    finite-field electric field strength (None unless ffield is used).
    """
    elec_idx = setup.elec_idx
    n_electrode = len(elec_idx)
    if n_electrode == 0:
        return charges[setup.mask], None
    if setup.ffield_electrode_mask is not None and calculator.slab_corr:
        raise ValueError("Slab correction and finite field cannot be used together.")

    # potential on electrode sites from the fixed electrolyte charges
    chi_elec = calculator.calc_coulomb_potential(
        setup.mask, positions, box, setup.eta, charges, pairs, buffer_scales
    )
    chi = (setup.chi + chi_elec)[elec_idx]

    if setup.ffield_electrode_mask is not None:
        chi_ffield, efield = finite_field_add_chi(
            positions,
            box,
            setup.ffield_electrode_mask,
            setup.ffield_potential,
            calculator.slab_axis,
        )
        chi = chi + chi_ffield[elec_idx]
    else:
        efield = None

    hardness = setup.hardness[elec_idx]
    zeros = jnp.zeros_like(charges)

    def energy_electrode(q_e: jnp.ndarray) -> jnp.ndarray:
        # electrode-only subsystem energy: all electrolyte charges are zero,
        # so evaluating the full-system quadratic energy is equivalent
        q_full = jnp.zeros_like(charges).at[elec_idx].set(q_e)
        e_coul = calculator.energy_fn(
            positions, box, pairs, q_full, setup.eta, zeros, zeros, buffer_scales
        )
        return e_coul + jnp.sum(chi * q_e) + jnp.sum(hardness * q_e**2)

    if method == "matinv":
        q_opt = matinv_optimize(
            energy_electrode,
            chi,
            setup.constraint_matrix,
            setup.constraint_vals,
        )
    else:
        q0 = vector_projection(
            charges[elec_idx], setup.constraint_matrix, setup.constraint_vals
        )
        q_opt = pgrad_optimize(
            energy_electrode,
            q0,
            setup.constraint_matrix,
            eps=calculator.eps,
            max_iter=calculator.max_iter,
        )

    return q_opt, efield


def calc_hessian(energy_fn, n: int, chunk_size: int = 8) -> jnp.ndarray:
    """
    Hessian of a quadratic energy function w.r.t. its n charges.

    The energy is exactly quadratic in the charges, so the Hessian columns
    are H e_j = grad(E)(e_j) - grad(E)(0), evaluated with reverse-mode AD
    only (forward-mode is unreliable through the piecewise constructs in
    E_sr3). Gradients are batched in chunks to bound the memory footprint
    of the vmapped PME evaluations.
    """
    grad_fn = jax.grad(energy_fn)
    g0 = grad_fn(jnp.zeros(n))

    eye = jnp.eye(n)
    rows = []
    for i in range(0, n, chunk_size):
        rows.append(jax.vmap(grad_fn)(eye[i : i + chunk_size]) - g0)
    return jnp.concatenate(rows, axis=0)


def matinv_optimize(
    energy_fn,
    chi: jnp.ndarray,
    constraint_matrix: jnp.ndarray,
    constraint_vals: jnp.ndarray,
) -> jnp.ndarray:
    """
    Solve the constrained quadratic minimization by matrix inversion of the
    KKT system [[H, A^T], [A, 0]] [q, lambda] = [-chi, b].
    """
    n = len(chi)
    hessian = calc_hessian(energy_fn, n)
    n_const = constraint_matrix.shape[0]
    if n_const == 0:
        return jnp.linalg.solve(hessian, -chi)
    coeff_matrix = jnp.block(
        [
            [hessian, constraint_matrix.T],
            [constraint_matrix, jnp.zeros((n_const, n_const))],
        ]
    )
    vector = jnp.concatenate([-chi, constraint_vals])
    solution = jnp.linalg.solve(coeff_matrix, vector)
    return solution[:n]


def pgrad_optimize(
    energy_fn,
    q0: jnp.ndarray,
    constraint_matrix: jnp.ndarray,
    eps: float = 1e-4,
    max_iter: int = 100,
) -> jnp.ndarray:
    """
    Minimize the energy under linear constraints with LBFGS on the projected
    gradient. q0 must already satisfy the constraints.
    """
    if jaxopt is None:
        raise ImportError("jaxopt is required for the iterative solver")

    n = len(q0)
    n_const = constraint_matrix.shape[0]
    if n_const > 0:
        a_mat = constraint_matrix
        # A^T (A A^T)^-1, maps constraint-space residuals back to q-space
        proj_coeff = a_mat.T @ jnp.linalg.inv(a_mat @ a_mat.T)
    else:
        proj_coeff = None

    def value_and_proj_grad(q):
        value, grads = jax.value_and_grad(energy_fn)(q)
        if proj_coeff is not None:
            grads = grads - proj_coeff @ (a_mat @ grads)
        return value, grads

    # torch-admp converges on |proj grad| / n <= eps with eps in eV;
    # jaxopt's tol is on the l2 norm of the (projected) gradient in kJ/mol
    solver = jaxopt.LBFGS(
        fun=value_and_proj_grad,
        value_and_grad=True,
        tol=eps * EV2KJ * n,
        maxiter=max_iter * 20,
    )
    res = solver.run(q0)
    return res.params


def infer(
    calculator: PolarizableElectrode,
    positions: jnp.ndarray,
    box: jnp.ndarray,
    charges: jnp.ndarray,
    pairs: jnp.ndarray,
    setup: ElectrodeSetup,
    method: str = "matinv",
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Optimize electrode charges, then compute energy and forces.

    Parameters
    ----------
    calculator : PolarizableElectrode
        the electrode calculator
    positions : jnp.ndarray
        (n_atoms, 3) positions [A]
    box : jnp.ndarray
        (3, 3) box [A]
    charges : jnp.ndarray
        (n_atoms,) initial charges [e]; electrolyte entries are kept fixed
    pairs : jnp.ndarray
        (n_pairs, 3) DMFF-style pair list (i, j, covalent order), may be
        padded with (n_atoms, n_atoms, 0) rows
    setup : ElectrodeSetup
        constraint data from setup_from_lammps
    method : str, optional
        "matinv" or an iterative method ("lbfgs"), by default "matinv"

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        energy [kJ/mol], forces [kJ/mol/A], optimized charges [e]
    """
    buffer_scales = pair_buffer_scales(pairs[:, :2])

    q_elec, efield = charge_optimization(
        calculator, positions, box, charges, pairs, buffer_scales, setup, method
    )

    q_opt = charges.at[setup.elec_idx].set(jax.lax.stop_gradient(q_elec))

    energy, forces = calculator.coulomb_calculator(
        positions, box, q_opt, setup.eta, pairs, buffer_scales, efield=efield
    )
    return energy, forces, q_opt
