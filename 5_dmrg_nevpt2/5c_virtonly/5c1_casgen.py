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
from pyscf.lo import PM
from pyscf.pbc.scf.addons import smearing_
from pyscf import mcscf
import scipy.linalg
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
O3      1.898581707    -1.898581707     0.092050000
O3     -1.898581707     1.898581707     0.092050000
O3      1.911309630     1.911309630    -0.092050000
O3     -1.911309630    -1.911309630    -0.092050000
O2      0.112137236    -1.786444471    -2.459050000
O2     -0.112137236     1.786444471    -2.459050000
O2     -0.124865158    -2.023446866     2.459050000
O2      0.124865158     2.023446866     2.459050000
O4      3.809891337     0.012727922    -0.092050000
O3      0.012727922     3.809891337    -0.092050000
O4     -3.809891337    -0.012727922    -0.092050000
O3     -0.012727922    -3.809891337    -0.092050000
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
O5      3.797163415    -3.797163415     0.092050000
O5     -3.797163415     3.797163415     0.092050000
O5      3.822619259     3.822619259     0.092050000
O5     -3.822619259    -3.822619259     0.092050000
Cu2     0.019091883     5.714837006     0.000000000
Cu2    -0.019091883    -5.714837006     0.000000000
O5      1.885853785    -5.708473045    -0.092050000
O5     -1.885853785     5.708473045    -0.092050000
O5      5.708473045    -1.885853785    -0.092050000
O5     -5.708473045     1.885853785    -0.092050000
O5     -5.721200967    -1.924037552     0.092050000
O5     -1.924037552    -5.721200967     0.092050000
O5      1.924037552     5.721200967     0.092050000
O5      5.721200967     1.924037552     0.092050000
O5      0.025455844     7.619782674     0.092050000
O5     -0.025455844    -7.619782674     0.092050000
""",
    # O1 bridging oxygen
    # O2 apical oxygen 
    # O3 peripheral oxygen
    # O4 O5 oxygen in buffer
    ecp={"Cu2": "crenbs",
         "La": gto.basis.parse_ecp("""
La nelec 46
La ul
2      1.00000000            0.00000000
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
           "O3": "ccpvdz",
           "O4": "ccpvdz",
           "O5": "ccpvdz@3s2p",
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
    symmetry=False,
    # symmetry=True,
    # symmetry="D2h",
    charge=10,
)

chg_fname = "../../0_hf/La214_Grande.evjen.lat"
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

def add_vext(mf, mol=None, h1e=None, enuc=None, v_ext=None, e_ext=None):
    '''
        A one-step function to add external potential and extra energy (1-body term, mf.get_hcore(); and constant term, mf.energy_nuc()).
        If this function is called on the same mf for many times, the latter call will remove extra h and e and re-add external terms for the original mf.
        To remove extra terms, just call add_vext(mf) with unspecified v_ext and e_ext.
    '''
    if mol is None:
        mol = mf.mol
    if h1e is None:
        h1e = mf.__class__.get_hcore(mf, mol)
    if enuc is None:
        enuc = mf.__class__.energy_nuc(mf)
    if v_ext is not None:
        logger.note(mf, '** Add external potential for %s **' % mf.__class__)
        mf.get_hcore = lambda *arg: h1e + v_ext
    else:
        logger.note(mf, '** No external potential added for %s **' % mf.__class__)
    if e_ext is not None:
        logger.note(mf, '** Add extra energy %.8f for %s **' % (e_ext, mf.__class__))
        mf.energy_nuc = lambda *arg: enuc + e_ext
    else:
        logger.note(mf, '** No external potential added for %s **' % mf.__class__)

from pyscf.lib.chkfile import load, save
emb_chk = "../../2_proj/2a_proj_rhf_mixed.chk"
s0_chk = "../../3_dmrgcas/3b_PM24e26o_s0.chk"
s1_chk = "../../3_dmrgcas/3b_PM24e26o_s1.chk"
# Load projection embedding potential and energy from chk.
h1e = load(emb_chk, "h1e")
vemb = load(emb_chk, "vemb")
eemb = load(emb_chk, "eemb")

mf = mm_charge(scf.RHF(mol), coords, charges)
s = mf.get_ovlp()
mf.dump_flags = lambda *args, **kwargs: dump_flags(mf, *args, **kwargs)
# Load HF result for embedded system.
mf.mo_coeff = load(s0_chk, "mcscf/mo_coeff")
mf.mo_energy = load(s0_chk, "mcscf/mo_energy")
mf.mo_occ = load(s0_chk, "mcscf/mo_occ")
nelec_a = int(np.sum(mf.mo_occ) + 0.1)
logger.note(mf, "num of electron = %i" % nelec_a)
mf.mol.nelectron = nelec_a

# Add embedding potential and energy to the system.
add_vext(mf, h1e=h1e, v_ext=vemb, e_ext=eemb)

ncore = load(s0_chk, "mcscf/ncore")
ncas = load(s0_chk, "mcscf/ncas")
nproj = np.count_nonzero(mf.mo_energy > 1e3)
nvirt = mol.nao - ncore - ncas - nproj

from pyscf.lo.orth import vec_schmidt
import scipy.linalg

def ao_split(mol, mo, ao_label, s=None):
    if s is None:
        s = mol.intor_symmetric("int1e_ovlp")
    orth_coeff = orth.orth_ao(mol, "meta_lowdin", s=s)[:,mol.search_ao_label(ao_label)]
    orth_coeff_proj = reduce(np.dot, (mo, mo.conj().T, s, orth_coeff))
    orth_coeff_proj_orth = vec_schmidt(orth_coeff_proj, s=s)
    orth_coeff_comp = mo - reduce(np.dot, (orth_coeff_proj_orth, orth_coeff_proj_orth.conj().T, s, mo))
    e, u = scipy.linalg.eigh(reduce(np.dot, (s, orth_coeff_comp, orth_coeff_comp.conj().T, s)), s)
    orth_coeff_comp_orth = u[:,e>1e-3]
    return orth_coeff_proj_orth, orth_coeff_comp_orth

# print(mol.ao_labels())
# print(len(mol.search_ao_label(["O[123]", "La1", "Cu1"])))

mo_core, mo_cas, mo_virt, mo_proj = np.hsplit(mf.mo_coeff, (ncore, ncore+ncas, ncore+ncas+nvirt))
# mo_core_proj, mo_core_comp = ao_split(mol, mo_core, ["O1 [12]s", "Cu1 [12]", "Cu1 3[sp]", "La1 5[sp]", "O[23] 1s", "O[23] 2[sp]"], s=s)
# logger.info(mf, "Core-correlated orbitals = %i" % mo_core_proj.shape[1])
mo_virt_proj, mo_virt_comp = ao_split(mol, mo_virt, ["O1 3[sd]", "Cu1 4[spf]", "Cu1 [56]", "La1 5d", "La1 6", "O[23] 3[spd]"], s=s)
logger.info(mf, "Virt-correlated orbitals = %i" % mo_virt_proj.shape[1])
mo_coeff_new_s0 = np.hstack((mo_core, mo_cas, mo_virt_proj, mo_virt_comp, mo_proj))

molden.from_mo(mol, project + "_ptorbs_s0.molden", mo_virt_proj, occ=np.zeros(mo_virt_proj.shape[1]))
molden.from_mo(mol, project + "_mowith_ptorbs_s0.molden", mo_coeff_new_s0, occ=mf.mo_occ)

mf.mo_coeff = load(s1_chk, "mcscf/mo_coeff")

mo_core, mo_cas, mo_virt, mo_proj = np.hsplit(mf.mo_coeff, (ncore, ncore+ncas, ncore+ncas+nvirt))
# mo_core_proj, mo_core_comp = ao_split(mol, mo_core, ["O1 [12]s", "Cu1 [12]", "Cu1 3[sp]", "La1 5[sp]", "O[23] 1s", "O[23] 2[sp]"], s=s)
# logger.info(mf, "Core-correlated orbitals = %i" % mo_core_proj.shape[1])
mo_virt_proj, mo_virt_comp = ao_split(mol, mo_virt, ["O1 3[sd]", "Cu1 4[spf]", "Cu1 [56]", "La1 5d", "La1 6", "O[23] 3[spd]"], s=s)
logger.info(mf, "Virt-correlated orbitals = %i" % mo_virt_proj.shape[1])
mo_coeff_new_s1 = np.hstack((mo_core, mo_cas, mo_virt_proj, mo_virt_comp, mo_proj))

molden.from_mo(mol, project + "_ptorbs_s1.molden", mo_virt_proj, occ=np.zeros(mo_virt_proj.shape[1]))
molden.from_mo(mol, project + "_mowith_ptorbs_s1.molden", mo_coeff_new_s1, occ=mf.mo_occ)

from pyscf.lib.chkfile import save
save(project + ".chk", "s0_coeff", mo_coeff_new_s0)
save(project + ".chk", "s1_coeff", mo_coeff_new_s1)
save(project + ".chk", "nelectron", mf.mol.nelectron)

