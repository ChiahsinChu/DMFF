import openmm.app as app
import openmm.unit as unit
import numpy as np
import jax.numpy as jnp
from ..api.topology import DMFFTopology
from ..api.paramset import ParamSet
from ..api.hamiltonian import _DMFFGenerators
from ..utils import DMFFException
from ..common.constants import EV2KJ
from ..admp.electrode import (
    LAMMPSElectrodeConstraint,
    PolarizableElectrode,
    infer,
    setup_from_lammps,
)


class ADMPPolarizableElectrodeGenerator:
    """
    Frontend generator for the polarizable electrode (constant potential /
    constant charge) force.

    XML schema (per-atom parameters, differentiable through ParamSet):

    .. code-block:: xml

        <ADMPPolarizableElectrodeForce>
            <Atom type="Pt" chi="0.0" J="0.0" eta="0.625"/>
            <Atom type="O" chi="0.0" J="0.0" eta="0.0"/>
        </ADMPPolarizableElectrodeForce>

    with chi the electronegativity [V], J the atomic hardness [V/e] and eta
    the Gaussian charge width [A] in the E_sr3 convention (eta = 1 / eta_lammps;
    eta = 0 means point charge).

    Electrode groups are system-specific and passed to createPotential via
    the ``electrode_constraints`` kwarg: a list of LAMMPSElectrodeConstraint
    objects or dicts with keys indices/mode/value and optional
    eta/chi/hardness/ffield. Values given in a constraint override the XML
    per-type parameters for those atoms (eta in the LAMMPS convention,
    [A^-1]); omitted ones fall back to the XML values.
    """

    def __init__(self, ffinfo: dict, paramset: ParamSet):
        self.name = "ADMPPolarizableElectrodeForce"
        self.ffinfo = ffinfo
        paramset.addField(self.name)

        self.key_type = None
        keys, params = [], []
        mask = []
        for node in self.ffinfo["Forces"][self.name]["node"]:
            attribs = node["attrib"]

            if self.key_type is None and "type" in attribs:
                self.key_type = "type"
            elif self.key_type is None and "class" in attribs:
                self.key_type = "class"
            elif self.key_type is not None and f"{self.key_type}" not in attribs:
                raise ValueError("Keyword 'class' or 'type' cannot be used together.")
            elif self.key_type is not None and f"{self.key_type}" in attribs:
                pass
            else:
                raise ValueError(
                    "Cannot find key type for ADMPPolarizableElectrodeForce."
                )
            keys.append(attribs[self.key_type])

            chi0 = float(attribs.get("chi", 0.0))
            J0 = float(attribs.get("J", 0.0))
            eta0 = float(attribs.get("eta", 0.0))

            if "mask" in attribs and attribs["mask"].upper() == "TRUE":
                mask.append(0.0)
            else:
                mask.append(1.0)

            params.append([chi0, J0, eta0])

        self.atom_keys = keys
        mask = jnp.array(mask)
        paramset.addParameter(
            jnp.array([i[0] for i in params]), "chi", field=self.name, mask=mask
        )
        paramset.addParameter(
            jnp.array([i[1] for i in params]), "J", field=self.name, mask=mask
        )
        paramset.addParameter(
            jnp.array([i[2] for i in params]), "eta", field=self.name, mask=mask
        )
        self._jaxPotential = None

    def getName(self) -> str:
        return self.name

    def overwrite(self, paramset: ParamSet) -> None:
        node_indices = [
            i
            for i in range(len(self.ffinfo["Forces"][self.name]["node"]))
            if self.ffinfo["Forces"][self.name]["node"][i]["name"] == "Atom"
        ]
        chi = paramset[self.name]["chi"]
        J = paramset[self.name]["J"]
        eta = paramset[self.name]["eta"]
        atom_mask = paramset.mask[self.name]["chi"]
        for nidx, idx in enumerate(node_indices):
            attrib = self.ffinfo["Forces"][self.name]["node"][idx]["attrib"]
            attrib["chi"] = chi[nidx]
            attrib["J"] = J[nidx]
            attrib["eta"] = eta[nidx]
            if atom_mask[nidx] < 0.999:
                attrib["mask"] = "true"

    def _find_atype_key_index(self, atype: str):
        for n, i in enumerate(self.atom_keys):
            if i == atype:
                return n
        return None

    @staticmethod
    def _parse_constraint(c) -> LAMMPSElectrodeConstraint:
        if isinstance(c, LAMMPSElectrodeConstraint):
            return c
        return LAMMPSElectrodeConstraint(
            indices=c["indices"],
            mode=c["mode"],
            value=c["value"],
            eta=c.get("eta", 0.0),
            chi=c.get("chi", 0.0),
            hardness=c.get("hardness", 0.0),
            ffield=c.get("ffield", False),
        )

    def createPotential(
        self, topdata: DMFFTopology, nonbondedMethod, nonbondedCutoff, **kwargs
    ):
        if nonbondedMethod not in [app.PME]:
            raise DMFFException(
                "Only PME is supported for ADMPPolarizableElectrodeForce"
            )

        if unit.is_quantity(nonbondedCutoff):
            r_cut = nonbondedCutoff.value_in_unit(unit.nanometer)
        else:
            r_cut = nonbondedCutoff
        r_cut_ang = r_cut * 10.0

        constraints = [
            self._parse_constraint(c) for c in kwargs.get("electrode_constraints", [])
        ]
        symm = kwargs.get("symm", False)
        slab_corr = kwargs.get("slab_corr", False)
        slab_axis = kwargs.get("slab_axis", 2)
        method = kwargs.get("method", "matinv")
        ethresh = kwargs.get("ethresh", 1e-6)
        kappa = kwargs.get("kappa", None)  # [A^-1]
        kmesh = kwargs.get("kmesh", None)
        damping = kwargs.get("damping", True)
        eps = kwargs.get("eps", 1e-4)
        max_iter = kwargs.get("max_iter", 100)
        has_aux = kwargs.get("has_aux", False)

        # topology info
        n_atoms = topdata.getNumAtoms()
        atoms = [a for a in topdata.atoms()]
        init_q = jnp.array(np.array([a.meta["charge"] for a in atoms]))
        map_idx = []
        for atom in atoms:
            atype = atom.meta[self.key_type]
            idx = self._find_atype_key_index(atype)
            if idx is None:
                raise DMFFException(
                    f"Atom type {atype} not found in ADMPPolarizableElectrodeForce"
                )
            map_idx.append(idx)
        map_idx = jnp.array(map_idx)

        # static constraint structure (mask, elec_idx, constraint matrix,
        # ffield groups); per-atom parameter arrays are rebuilt from the
        # ParamSet in every potential_fn call to stay differentiable
        setup = setup_from_lammps(n_atoms, constraints, symm)

        # static per-constraint parameter overrides
        overrides = []
        for c in constraints:
            overrides.append(
                (
                    c.indices,
                    None if c.eta == 0.0 else 1.0 / c.eta,
                    c.chi,
                    c.hardness,
                    c.value if c.mode == "conp" else None,
                )
            )

        # static kmesh from the topology box (in angstrom)
        cell = np.array(topdata.getPeriodicBoxVectors()) * 10.0
        calculator = PolarizableElectrode(
            rcut=r_cut_ang,
            box=cell,
            ethresh=ethresh,
            kappa=kappa,
            kmesh=kmesh,
            slab_corr=slab_corr,
            slab_axis=slab_axis,
            damping=damping,
            eps=eps,
            max_iter=max_iter,
        )
        self.calculator = calculator
        self.setup = setup

        def potential_fn(
            positions: jnp.ndarray,
            box: jnp.ndarray,
            pairs: jnp.ndarray,
            params: ParamSet,
            aux: dict = None,
        ):
            # nm -> angstrom
            positions_ang = positions * 10.0
            box_ang = box * 10.0

            eta = params[self.name]["eta"][map_idx]
            chi = params[self.name]["chi"][map_idx] * EV2KJ
            hardness = params[self.name]["J"][map_idx] * EV2KJ
            for indices, eta_c, chi_c, hard_c, conp_value in overrides:
                if eta_c is not None:
                    eta = eta.at[indices].set(eta_c)
                if chi_c != 0.0:
                    chi = chi.at[indices].set(chi_c * EV2KJ)
                if hard_c != 0.0:
                    hardness = hardness.at[indices].set(hard_c * EV2KJ)
                if conp_value is not None:
                    chi = chi.at[indices].add(-conp_value * EV2KJ)

            run_setup = setup._replace(eta=eta, chi=chi, hardness=hardness)
            charges = init_q
            if aux is not None and "q" in aux:
                charges = aux["q"]

            energy, _, q_opt = infer(
                calculator,
                positions_ang,
                box_ang,
                charges,
                pairs,
                run_setup,
                method=method,
            )
            if has_aux:
                aux = {} if aux is None else dict(aux)
                aux["q"] = q_opt
                return energy, aux
            return energy

        self._jaxPotential = potential_fn
        return potential_fn


_DMFFGenerators["ADMPPolarizableElectrodeForce"] = ADMPPolarizableElectrodeGenerator
