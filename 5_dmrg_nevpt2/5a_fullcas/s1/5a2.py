#!/usr/bin/env python3

import numpy as np
from pyscf import gto, scf  # , mcscf, fci
from pyscf.qmmm import mm_charge
from pyscf.tools import molden
import os
import sys
from pyscf.lib import logger
from pyscf.lo import orth
from pyscf.lo import PM
from pyscf.pbc.scf.addons import smearing_
from pyscf import mcscf
import scipy.linalg
from functools import reduce
from pyscf import mrpt

# assign project to the filename without .py
pyfile = __file__
project = pyfile[:pyfile.rfind(".")]

mol = gto.Mole()
mol.max_memory = 110*1024
mol.verbose = 5
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
    symmetry=False,
    # symmetry=True,
    # symmetry="D2h",
    charge=10,
)

chg_fname = "../../../0_hf/La214_Grande.evjen.lat"
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
emb_chk = "../../../2_proj/2a_proj_rhf_mixed.chk"
mo_chk = "../5a1_casgen.chk"

# Load projection embedding potential and energy from chk.
h1e = load(emb_chk, "h1e")
vemb = load(emb_chk, "vemb")
eemb = load(emb_chk, "eemb")

mf_ao = mm_charge(scf.RHF(mol), coords, charges)
s = mf_ao.get_ovlp()
mf_ao.dump_flags = lambda *args, **kwargs: dump_flags(mf, *args, **kwargs)
# Load HF result for embedded system.
mf_ao.mo_coeff = load(mo_chk, "s1_coeff")
mf_ao.mol.nelectron = load(mo_chk, "nelectron")

# Add embedding potential and energy to the system.
add_vext(mf_ao, h1e=h1e, v_ext=vemb, e_ext=eemb)

# Now really do the CASSCF S=0.

from pyscf import dmrgscf
from pyscf import ao2mo
norb, nelec = 292, (99, 97)
mc_ao = mcscf.CASCI(mf_ao, norb, nelec)
logger.info(mc_ao, "Start generating h1e_cas, nuc_cas in MO basis.")
h1e_cas, nuc_cas = mc_ao.get_h1eff()
logger.info(mc_ao, "h1e_cas, nuc_cas generated. ")
logger.info(mc_ao, "Start generating eri_cas in MO basis. ")
eri_cas = ao2mo.restore(8, mc_ao.get_h2eff(), norb)
logger.info(mc_ao, "eri_cas generated, shape = %s" % (eri_cas.shape,))

mol_mo = gto.M()
mol_mo.max_memory = mol.max_memory
mol_mo.output = mol.output
mol_mo.nelectron = sum(nelec) # 196
mol_mo.nao = norb # 292
mol_mo.incore_anyway = True

mf_mo = scf.RHF(mol_mo)
mf_mo.get_hcore = lambda *args: h1e_cas
mf_mo.energy_nuc = lambda *args: nuc_cas
mf_mo.get_ovlp = lambda *args: np.identity(mol_mo.nao)
mf_mo._eri = eri_cas
mf_mo.mo_coeff = np.identity(mol_mo.nao)

norb_mo, nelec_mo = 26, (13, 11)
mc_mo = mcscf.CASCI(mf_mo, norb_mo, nelec_mo)
mc_mo.fcisolver = dmrgscf.DMRGCI(mol_mo, maxM=1000)
# Specify M here, should converge with M in principle
# mc_ao.fcisolver.wfnsym = 'A'
# sing.fcisolver.spin = 1  # ????? probably ignored
# sing.fcisolver.num_thrds=int(os.environ.get('OMP_NUM_THREADS',1))
# sing.fcisolver.memory=mol.max_memory/1024.
mc_mo.fcisolver.runtimeDir = os.path.abspath('.')
mc_mo.fcisolver.scratchDirectory = os.path.join(mc_mo.fcisolver.runtimeDir, "dmrg_scratch")
mc_mo.fcisolver.block_extra_keyword = ["warmup occ", 
                                       "occ " + " ".join(["2"]*11)
                                        + " " + " ".join(["1"]*2)
                                        + " " + " ".join(["0"]*13), 
                                       "cbias 0.2",
                                       "singlet_embedding", 
                                       "gaopt default"]
# mc_mo.fcisolver.threads = int(os.environ.get("OMP_NUM_THREADS", 32))
mc_mo.fcisolver.threads = 64
mc_mo.fcisolver.memory = mol.max_memory // 1024
mc_mo.chkfile = project + '_dmrg_s1.chk'
mc_mo.kernel()

mpt_mo = mrpt.NEVPT(mc_mo).compress_approx()
# In this way, virtual CAS orbitals other than La 5d are frozen. 
# However, core orbitals are not frozen, since h1e_for_cas is involved and I don't know how to do that. 
mpt_mo.scratchDirectory = os.path.join(mc_mo.fcisolver.runtimeDir, "nevpt_scratch")
mpt_mo.chkfile = project + '_nevpt_s1.chk'
mpt_mo.kernel()

