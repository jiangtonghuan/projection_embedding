#!/usr/bin/env python3

import numpy as np
from pyscf import gto, scf  # , mcscf, fci
from pyscf.qmmm import mm_charge
# from pyscf import lo
from pyscf.tools import molden
# from pyscf.lib import logger
# from pyscf import symm
import os
import sys
from pyscf.lib import logger
from pyscf.lo import orth
from pyscf.pbc.scf.addons import smearing_

sys.path.append("..")

from orth_csv import save_csv, c_meta_lowdin_ao, dm_meta_lowdin_ao
from functools import reduce

# assign project to the filename without .py
pyfile = __file__
project = pyfile[:pyfile.rfind(".")]

mol = gto.Mole()
mol.max_memory = 110*1024
mol.verbose = 4
mol.output = project+".log"
mol.build(
    atom="""
O1     -0.000000000     0.000000000     0.092050000
Cu1     0.006363961     1.904945669     0.000000000
Cu1    -0.006363961    -1.904945669     0.000000000
La1     1.878187334    -0.020394374    -1.814700000
La1    -1.878187334     0.020394374    -1.814700000
La1     1.931704003     0.033122296     1.814700000
La1    -1.931704003    -0.033122296     1.814700000
O2      1.898581707    -1.898581707     0.092050000
O2     -1.898581707     1.898581707     0.092050000
O2      1.911309630     1.911309630    -0.092050000
O2     -1.911309630    -1.911309630    -0.092050000
O2      0.112137236    -1.786444471    -2.459050000
O2     -0.112137236     1.786444471    -2.459050000
O2     -0.124865158    -2.023446866     2.459050000
O2      0.124865158     2.023446866     2.459050000
O2      3.809891337     0.012727922    -0.092050000
O2      0.012727922     3.809891337    -0.092050000
O2     -3.809891337    -0.012727922    -0.092050000
O2     -0.012727922    -3.809891337    -0.092050000
Cu2    -3.803527376     1.892217746     0.000000000
Cu2     3.803527376    -1.892217746     0.000000000
Cu2    -3.816255298    -1.917673591     0.000000000
Cu2     3.816255298     1.917673591     0.000000000
La2     1.890915256     3.789496963     1.814700000
La2    -1.890915256    -3.789496963     1.814700000
La2    -1.918976081     3.776769041    -1.814700000
La2     1.918976081    -3.776769041    -1.814700000
La2     1.865459412    -3.830285711     1.814700000
La2    -1.865459412     3.830285711     1.814700000
La2    -1.944431925    -3.843013633    -1.814700000
La2     1.944431925     3.843013633    -1.814700000
La2     0.020394374    -1.878187334     4.760300000
La2    -0.020394374     1.878187334     4.760300000
La2     0.033122296     1.931704003    -4.760300000
La2    -0.033122296    -1.931704003    -4.760300000
O3      3.797163415    -3.797163415     0.092050000
O3     -3.797163415     3.797163415     0.092050000
O3      3.822619259     3.822619259     0.092050000
O3     -3.822619259    -3.822619259     0.092050000
Cu2     0.019091883     5.714837006     0.000000000
Cu2    -0.019091883    -5.714837006     0.000000000
O3      1.885853785    -5.708473045    -0.092050000
O3     -1.885853785     5.708473045    -0.092050000
O3      5.708473045    -1.885853785    -0.092050000
O3     -5.708473045     1.885853785    -0.092050000
O3     -5.721200967    -1.924037552     0.092050000
O3     -1.924037552    -5.721200967     0.092050000
O3      1.924037552     5.721200967     0.092050000
O3      5.721200967     1.924037552     0.092050000
O3      0.025455844     7.619782674     0.092050000
O3     -0.025455844    -7.619782674     0.092050000
""",
    ecp={"Cu2": "crenbs",
         "La": gto.basis.parse_ecp("""
La nelec 46
La ul
2      1.000000000            0.0000000
La S
2      3.30990000            91.9321770
2      1.65500000            -3.7887640
La P
2      2.83680000            63.7594860
2      1.41840000            -0.6479580
La D
2      2.02130000            36.1191730
2      1.01070000             0.2191140
La F
2      4.02860000           -36.0100160
           """),
         "La2": "crenbs"
         },
    basis={"Cu1": "ccpvdz",
           "Cu2": "crenbs",
           "O1": "ccpvdz",
           "O2": "ccpvdz",
           "O3": "ccpvdz@3s2p",
           "La1":  gto.parse("""
#BASIS SET: (4s4p3d) -> [2s2p1d] Basis  for PP46MWB
La   S
      5.087399            -0.417243             0.000000
      4.270978             0.886010             0.000000
      1.915458            -1.419752             0.000000
      0.525596             0.000000             1.000000
La   P
      3.025161             0.538196             0.000000
      2.382095            -0.981640             0.000000
      0.584426             1.239590             0.000000
      0.260360             0.000000             1.000000
La   D
      1.576824            -0.096944
      0.592390             0.407466
      0.249500             0.704363
END
"""),
           "La2": "crenbs"
           },
    # symmetry=False,
    symmetry=True,
    # symmetry="D2h",
    charge=10,
)

chg_fname = "La214_Grande.evjen.lat"
chg_data = np.loadtxt(chg_fname, skiprows=2)
coords = chg_data[:, 0:3]
assert (coords == np.loadtxt(chg_fname, skiprows=2)[:, 0:3]).all()
charges = chg_data[:, 3]

def dump_flags(qmmm_obj, verbose=None):
    method_class = qmmm_obj.__class__.__bases__[1]
    method_class.dump_flags(qmmm_obj, verbose)
    logger.info(qmmm_obj, "** Add background charges for %s", method_class)
    if qmmm_obj.verbose >= logger.DEBUG:
        logger.debug(qmmm_obj, 'Charge      Location')
        coords = qmmm_obj.mm_mol.atom_coords()
        charges = qmmm_obj.mm_mol.atom_charges()
        if len(charges) <= 12:
            for i, z in enumerate(charges):
                logger.debug(qmmm_obj, '%.9g    %s', z, coords[i])
        else:
            for i, z in enumerate(charges[:6]):
                logger.debug(qmmm_obj, '%.9g    %s', z, coords[i])
            logger.debug(qmmm_obj, '...... (%i point charges in total)', len(charges))
            for i, z in enumerate(charges[-6:]):
                logger.debug(qmmm_obj, '%.9g    %s', z, coords[i])
    return qmmm_obj

def write_mulliken(mf, outdir):
    s = mf.get_ovlp()
    orth_coeff = orth.orth_ao(mol, "meta_lowdin", s=s)

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    label = mf.mol.ao_labels()
    save_csv(os.path.join(outdir, "meta_lowdin_orth.csv"), orth_coeff, label=label, label2=label)
    _, mo_coeff_orth = c_meta_lowdin_ao(mf.mol, mf.mo_coeff, s=s, orth_coeff=orth_coeff)
    _, dm_orth = dm_meta_lowdin_ao(mf.mol, mf.make_rdm1(), s=s, orth_coeff=orth_coeff)

    mo_coeff_orth_extended = np.vstack((mf.mo_energy, mf.mo_occ, mo_coeff_orth))
    label_extended = np.hstack((["mo_energy", "mo_occ"], label))

    save_csv(os.path.join(outdir, "mo_coeff.csv"), mo_coeff_orth_extended, label=label_extended)
    save_csv(os.path.join(outdir, "dm.csv"), dm_orth, label=label)

    molden.from_scf(mf, os.path.join(outdir, "scf.molden"))


mf = mm_charge(scf.RHF(mol), coords, charges)
mf.dump_flags = lambda *args, **kwargs: dump_flags(mf, *args, **kwargs)
restart_chk = "0a_rohf-rohf0.chk"
mo_coeff = scf.chkfile.load(restart_chk, "scf/mo_coeff")
mo_energy = scf.chkfile.load(restart_chk, "scf/mo_energy")
mo_occ = scf.chkfile.load(restart_chk, "scf/mo_occ")
dm0 = mf.make_rdm1(mo_coeff, mo_occ)

def mixed_state_(mf, nopen=None):
    # mf should be RHF objects. 
    mf_class = mf.__class__
        
    if nopen % 2 != 0:
        raise ValueError("Mixed RHF's nopen must be even.")
               # In principle is also possible to have odd nelectron and odd nopen, 
               # yet it's not the case in our calculation, and not implemented here. 

    def get_occ(mo_energy=None, mo_coeff=None):
        '''Label the occupancies for each orbital.
        NOTE the occupancies are not assigned based on the orbital energy ordering.
        The first N orbitals are assigned to be occupied orbitals.
    
        Examples:
    
        >>> mol = gto.M(atom='H 0 0 0; O 0 0 1.1', spin=1)
        >>> mf = scf.hf.SCF(mol)
        >>> energy = np.array([-10., -1., 1, -2., 0, -3])
        >>> mf.get_occ(energy)
        array([2, 2, 2, 2, 1, 0])
        '''
    
        if mo_energy is None: mo_energy = mf.mo_energy
        nmo = mo_energy.size
        mo_occ = np.zeros(nmo)
        if nopen == 0:
            logger.warn(mf, "nopen set to 0. Using original RHF get_occ.")
            mo_occ = mf.__class__.get_occ(mf, mo_energy, mo_coeff)
    
        nocc = mf.mol.nelectron // 2 + nopen // 2
        ncore = mf.mol.nelectron // 2 - nopen // 2
        mo_occ = _fill_rohf_occ(mo_energy, ncore, nopen)
    
        if mf.verbose >= logger.INFO and nocc < nmo and ncore > 0:
            ehomo = max(mo_energy[mo_occ> 0])
            elumo = min(mo_energy[mo_occ==0])
            if ehomo+1e-3 > elumo:
                logger.warn(mf, 'HOMO %.15g >= LUMO %.15g', ehomo, elumo)
            else:
                logger.info(mf, '  HOMO = %.15g  LUMO = %.15g', ehomo, elumo)
            if nopen > 0 and mf.verbose >= logger.DEBUG:
                core_idx = mo_occ == 2
                open_idx = mo_occ == 1
                vir_idx = mo_occ == 0
                logger.debug(mf, '  Frontier mo_energy\'s')
                logger.debug(mf, '  Highest 2-occ = %18.15g', max(mo_energy[core_idx]))
                logger.debug(mf, '  Lowest 0-occ =  %18.15g', min(mo_energy[vir_idx]))
                for i in np.where(open_idx)[0]:
                    logger.debug(mf, '  1-occ = %18.15g', mo_energy[i])
    
            if mf.verbose >= logger.DEBUG:
                np.set_printoptions(threshold=nmo)
                logger.debug(mf, '  mo_energy =\n%s', mo_energy)
                np.set_printoptions(threshold=1000)
        return mo_occ
    
    def _fill_rohf_occ(mo_energy, ncore, nopen):
        mo_occ = np.zeros_like(mo_energy)
        open_idx = []
        core_sort = np.argsort(mo_energy)
        core_idx = core_sort[:ncore]
        if nopen > 0:
            open_idx = core_sort[ncore:]
            open_sort = np.argsort(mo_energy[open_idx])
            open_idx = open_idx[open_sort[:nopen]]
        mo_occ[core_idx] = 2
        mo_occ[open_idx] = 1
        return mo_occ
    
    def get_grad(mo_coeff, mo_occ, fock):
        '''ROHF gradients is the off-diagonal block [co + cv + ov], where
        [ cc co cv ]
        [ oc oo ov ]
        [ vc vo vv ]
        '''
        occidxa = mo_occ > 0
        occidxb = mo_occ == 2
        viridxa = ~occidxa
        viridxb = ~occidxb
        uniq_var_a = viridxa.reshape(-1,1) & occidxa
        uniq_var_b = viridxb.reshape(-1,1) & occidxb
    
        fock = reduce(np.dot, (mo_coeff.conj().T, fock, mo_coeff))
    
        g = np.zeros_like(fock)
        g[uniq_var_a]  = fock[uniq_var_a]
        g[uniq_var_b] += fock[uniq_var_b]
        return g[uniq_var_a | uniq_var_b]

    mf.nopen = nopen
    mf._keys = mf._keys.union(['nopen'])
    mf.get_occ = get_occ
    mf.get_grad = get_grad
    # make_rdm1, get_fock, energy_tot need not be revised, 
    # because they are the same as original RHF. 
    return mf

mixed_state_(mf, nopen=8)
# mf = mf.newton()
# mf.__dict__.update(scf.chkfiale.load(project+"-rohf0.chk", "scf"))
mf.chkfile = project+"-rhf_mixed1.chk"
# mf.irrep_nelec = occMolcas
mf.verbose = 5
mf.kernel(dm0)

# Orbital energies, Mulliken population etc.
try:
    mf.analyze()
except:
    print("no analysis for now")
print("E(HF) = %.12f" % mf.e_tot)

molden.from_scf(mf, project+"_rhf_mixed1.molden")
write_mulliken(mf, project+"-rhf_mixed1-mulliken")
