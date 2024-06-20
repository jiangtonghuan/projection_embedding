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

# assign project to the filename without .py
pyfile = __file__
project = pyfile[:pyfile.rfind(".")]

mol = gto.Mole()
mol.max_memory = 120*1024
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
    symmetry=False,
    # symmetry=True,
    # symmetry="D2h",
    charge=10,
)

chg_fname = "../0_hf/La214_Grande.evjen.lat"
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
emb_chk = "../2_proj/2a_proj_rhf_mixed.chk"
# Load projection embedding potential and energy from chk.
h1e = load(emb_chk, "h1e")
vemb = load(emb_chk, "vemb")
eemb = load(emb_chk, "eemb")

mf = mm_charge(scf.RHF(mol), coords, charges)
s = mf.get_ovlp()
mf.dump_flags = lambda *args, **kwargs: dump_flags(mf, *args, **kwargs)
# Load HF result for embedded system.
mf.mo_coeff = load(emb_chk, "scf/mo_coeff")
mf.mo_occ = load(emb_chk, "scf/mo_occ")
mf.mo_energy = load(emb_chk, "scf/mo_energy")
nelec_a = int(np.sum(mf.mo_occ))
logger.note(mf, "num of electron = %i" % nelec_a)
mf.mol.nelectron = nelec_a

# Add embedding potential and energy to the system.
add_vext(mf, h1e=h1e, v_ext=vemb, e_ext=eemb)
mf.chkfile = project + "_rhf_mixed.chk"

def c_meta_lowdin_ao(mol, c, s=None, orth_coeff=None):
    label = mol.ao_labels()
    if s is None:
        s = mol.intor_symmetric("int1e_ovlp")
    if orth_coeff is None:
        orth_coeff = orth.orth_ao(mol, 'meta_lowdin', s=s)
    c_inv = np.dot(orth_coeff.conj().T, s)
    if isinstance(c, np.ndarray) and c.ndim == 2:
        c = np.dot(c_inv, c)
    else: # ROHF
        c = np.dot(c_inv, c[0]+c[1])
    return label, c  

from pyscf.lib.chkfile import load
pm_chkfile = "../1_pm/1a_PM.chk"
double_loc_a = load(pm_chkfile, "mo_da")
double_loc_b = load(pm_chkfile, "mo_db")
single_loc_a = load(pm_chkfile, "mo_sa")
single_loc_b = load(pm_chkfile, "mo_sb")
empty_loc = load(pm_chkfile, "mo_u")

def get_mo_with_aolabel(mol, mo, ao_label, mo_orth=None):
    if mo_orth is None:
        s = mol.intor_symmetric("int1e_ovlp")
        orth_coeff = orth.orth_ao(mol, "meta_lowdin", s=s)
        label_list, mo_orth = c_meta_lowdin_ao(mol, mo, s=s, orth_coeff=orth_coeff)
    else:
        label_list = mol.ao_labels()
    ao_ids = mol.search_ao_label(ao_label)
    # print(mo_orth.shape)
    mo_orth_on_ao_ids = mo_orth[ao_ids]
    mo_argmax_on_ao_ids = np.argmax(np.abs(mo_orth_on_ao_ids), axis=1)
    for ao_id, mo_max in zip(ao_ids, mo_argmax_on_ao_ids):
        logger.info(mol, "AO #%i: %s, mainly represented by MO #%i, amplitude %.5f" %
                    (ao_id, label_list[ao_id], mo_max, mo_orth[ao_id, mo_max]))
    return mo_argmax_on_ao_ids

logger.info(mf, "Find CAS in double_loc")
double_loc_casid = get_mo_with_aolabel(mol, double_loc_a, ["O1 2p", "Cu1 3dz", "Cu1 3dxy", "Cu1 3dxz", "Cu1 3dyz"])
logger.info(mf, "Find CAS in single_loc")
single_loc_casid = get_mo_with_aolabel(mol, single_loc_a, "Cu1 3dx2-y2")
logger.info(mf, "Find CAS in empty_loc")
empty_loc_casid  = get_mo_with_aolabel(mol, empty_loc, ["O1 3p", "Cu1 4d"])

mo_loc = np.hstack((double_loc_a, single_loc_a, empty_loc, single_loc_b, double_loc_b))
casid = np.hstack((double_loc_casid, 
                   single_loc_casid + double_loc_a.shape[1], 
                   empty_loc_casid + double_loc_a.shape[1] + single_loc_a.shape[1]))
logger.info(mf, "CAS id in localized basis: %s" % casid)

# Now really do the CASSCF S=0.

from pyscf import dmrgscf
norb, nelec = 26, (12, 12)
mc0 = mcscf.CASSCF(mf, norb, nelec)
mc0.frozen = np.where(mf.mo_energy > 1e3)[0]
mo_init = mcscf.sort_mo(mc0, mo_loc, casid+1)
molden.from_mo(mol, project+"_init.molden", mo_init, 
               occ=np.concatenate((np.full(mc0.ncore, 2), 
                                   np.full(mc0.ncas, 1), 
                                   np.full(mol.nao - mc0.ncas - mc0.ncore - len(mc0.frozen), 0), 
                                   np.full(len(mc0.frozen), -1))))
mc0.chk_ci = True
mc0.fix_spin_(ss=0)
mc0.fcisolver = dmrgscf.DMRGCI(mol, maxM=1000)  # Specify M here, should converge with M in principle
# mc0.fcisolver.wfnsym = 'A'
# sing.fcisolver.spin = 1  # ????? probably ignored
# sing.fcisolver.num_thrds=int(os.environ.get('OMP_NUM_THREADS',1))
# sing.fcisolver.memory=mol.max_memory/1024.
# sing.fcisolver.scratchDirectory = scratchdir
# sing.fcisolver.runtimeDir = project+'_'+str(os.getpid())
mc0.fcisolver.block_extra_keyword = ["warmup occ", 
                                     "occ " + " ".join(["2"]*11)
                                      + " " + " ".join(["1"]*2)
                                      + " " + " ".join(["0"]*13), 
                                     "cbias 0.2",
                                     "singlet_embedding", 
                                     "gaopt default"]
# mc0.fcisolver.threads = int(os.environ.get("OMP_NUM_THREADS", 32))
mc0.fcisolver.threads = 32
mc0.fcisolver.memory = mol.max_memory // 1024
mc0.chkfile = project + '_s0.chk'
mc0.kernel(mo_init)

# from pyscf.symm.addons import label_orb_symm
# mc_symm = label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mc.mo_coeff, s=s)
mc_occ = np.diag(reduce(np.dot, (mc0.mo_coeff.T, s, mc0.make_rdm1(), s, mc0.mo_coeff)))
molden.from_mo(mol, project+"_s0.molden", mc0.mo_coeff, ene=mc0.mo_energy, occ=mc_occ)

mc0.analyze()

# Now really do the CASSCF S=1.
norb, nelec = 26, (13, 11)
mc1 = mcscf.CASSCF(mf, norb, nelec)
mc1.frozen = np.where(mf.mo_energy > 1e3)[0]
mo_init = mcscf.sort_mo(mc1, mo_loc, casid+1)
molden.from_mo(mol, project+"_init.molden", mo_init, 
               occ=np.concatenate((np.full(mc1.ncore, 2), 
                                   np.full(mc1.ncas, 1), 
                                   np.full(mol.nao - mc1.ncas - mc1.ncore - len(mc1.frozen), 0), 
                                   np.full(len(mc1.frozen), -1))))
mc1.chk_ci = True
mc1.fcisolver = dmrgscf.DMRGCI(mol, maxM=1000)  # Specify M here, should converge with M in principle
mc1.fcisolver.block_extra_keyword = ["warmup occ", 
                                     "occ " + " ".join(["2"]*11)
                                      + " " + " ".join(["1"]*2)
                                      + " " + " ".join(["0"]*13), 
                                     "cbias 0.2",
                                     "singlet_embedding", 
                                     "gaopt default"]
mc1.fcisolver.threads = 32
mc1.fcisolver.memory = mol.max_memory // 1024
mc1.chkfile = project + '_s1.chk'
mc1.kernel(mo_init)

# from pyscf.symm.addons import label_orb_symm
# mc_symm = label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mc.mo_coeff, s=s)
mc_occ = np.diag(reduce(np.dot, (mc1.mo_coeff.T, s, mc1.make_rdm1(), s, mc1.mo_coeff)))
molden.from_mo(mol, project+"_s1.molden", mc1.mo_coeff, ene=mc1.mo_energy, occ=mc_occ)

mc1.analyze()
