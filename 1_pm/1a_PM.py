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

mf = mm_charge(scf.RHF(mol), coords, charges)
mf.dump_flags = lambda *args, **kwargs: dump_flags(mf, *args, **kwargs)
restart_chk = "../0_hf/0b_mixedrhf-rhf_mixed1.chk"
mo_coeff = scf.chkfile.load(restart_chk, "scf/mo_coeff")
mo_energy = scf.chkfile.load(restart_chk, "scf/mo_energy")
mo_occ = scf.chkfile.load(restart_chk, "scf/mo_occ")
# print(mo_occ==2)

double_occ = mo_coeff[:,mo_occ==2]
single_occ = mo_coeff[:,mo_occ==1]
empty_occ = mo_coeff[:,mo_occ==0]

double_loc = PM(mol, double_occ).kernel()
single_loc = PM(mol, single_occ).kernel()
empty_loc = PM(mol, empty_occ).kernel()

def atom_belong(mol, *mos):
    r_ao = mol.intor_symmetric("int1e_r", comp=3)
    atom_list, atom_pos = tuple(map(np.array, zip(*mol._atom)))
    result = list()
    for i, mo in enumerate(mos):
        r_mean = np.einsum("ui,xuv,vi->ix", mo, r_ao, mo)
        orb_rela_pos = np.linalg.norm(r_mean[:,None,:] - atom_pos, axis=2)
        atom_closest = np.argmin(orb_rela_pos, axis=1)
        print(atom_closest)
        result.append(atom_closest)
    return tuple(result)

atom_list, atom_pos = tuple(map(np.array, zip(*mol._atom)))
d_at, s_at, u_at = atom_belong(mol, double_loc, single_loc, empty_loc)
for i, atm_name in enumerate(atom_list):
    print("Atom #%i, Label %s, Occurence in d %i, s %i, u %i" % 
            (i, atm_name, np.count_nonzero(d_at == i), np.count_nonzero(s_at == i), np.count_nonzero(u_at == i)))
    
atomlist_a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18], dtype=int) # Cu 2 O 11 La 4
atomlist_b = np.setdiff1d(np.arange(mol.natm), atomlist_a)
atomlist_b2 = np.array([35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50], dtype=int)
atomlist_b1 = np.setdiff1d(atomlist_b, atomlist_b2)
double_loc_a = double_loc[:,np.in1d(d_at, atomlist_a)]
double_loc_b = double_loc[:,np.in1d(d_at, atomlist_b)]
single_loc_a = single_loc[:,np.in1d(s_at, atomlist_a)]
single_loc_b = single_loc[:,np.in1d(s_at, atomlist_b)]
# Get the original PM localized orbital. 

# I found a bug in my old localization script.

# If we work in a Cu 2 O 11 La 4 cluster, then the cluster atom list (A atom) should be

# np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18]).

# However, in my old code, A atom is incorrectly written as

# np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18]).

# This leads to a wrong cluster setting, potentially making all the following calculations wrong and SCF and CASSCF convergence extremely hard.

dma = 2 * np.einsum("vi,ui->uv", *((double_loc_a,)*2)) \
        + np.einsum("vi,ui->uv", *((single_loc_a,)*2))
dmb = 2 * np.einsum("vi,ui->uv", *((double_loc_b,)*2)) \
        + np.einsum("vi,ui->uv", *((single_loc_b,)*2))
empty_loc_ab1 = empty_loc[:,np.in1d(u_at, np.concatenate((atomlist_a, atomlist_b1)))]
empty_loc_b2  = empty_loc[:,np.in1d(u_at, atomlist_b2)]

from pyscf.lib.chkfile import save
save(project + ".chk", "mo_da", double_loc_a)
save(project + ".chk", "mo_db", double_loc_b)
save(project + ".chk", "mo_sa", single_loc_a)
save(project + ".chk", "mo_sb", single_loc_b)
save(project + ".chk", "mo_u",  empty_loc)
save(project + ".chk", "mo_uab1",  empty_loc_ab1)
save(project + ".chk", "mo_ub2", empty_loc_b2)
save(project + ".chk", "dm_a",  dma)
save(project + ".chk", "dm_b",  dmb)

molden.from_mo(mol, project+"_double_occ_A.molden", double_loc_a, occ=2*np.ones(double_loc_a.shape[1]), symm=None)
molden.from_mo(mol, project+"_double_occ_B.molden", double_loc_b, occ=2*np.ones(double_loc_b.shape[1]))
molden.from_mo(mol, project+"_single_occ_A.molden", single_loc_a, occ=  np.ones(single_loc_a.shape[1]))
molden.from_mo(mol, project+"_single_occ_B.molden", single_loc_b, occ=  np.ones(single_loc_b.shape[1]))
molden.from_mo(mol, project+"_empty_occ.molden",    empty_loc,    occ=  np.ones(empty_loc.shape[1]))
molden.from_mo(mol, project+"_empty_occ_AB1.molden",empty_loc_ab1,occ=  np.zeros(empty_loc_ab1.shape[1]))
molden.from_mo(mol, project+"_empty_occ_B2.molden", empty_loc_b2, occ=  np.zeros(empty_loc_b2.shape[1]))

import sys
sys.path.append("..")
from orth_csv import save_csv, c_meta_lowdin_ao

def write_mulliken(mol, mo, outfile_mark="", s=None, label_mark=None):
    if s is None:
        s = mol.intor_symmetric("int1e_ovlp")
    orth_coeff = orth.orth_ao(mol, "meta_lowdin", s=s)

    label = mol.ao_labels()
    save_csv(project + "%s_meta_lowdin_orth.csv" % outfile_mark, orth_coeff, label_row=label, label_col=label)
    _, mo_orth = c_meta_lowdin_ao(mol, mo, s=s, orth_coeff=orth_coeff)

    if label_mark is None:
        label2_withmark = label
    else:
        label2_withmark = ["#%i %s" % (i, mk) for i, mk in zip(range(mo.shape[1]), label_mark)]
    save_csv(project + "%s_mo_in_meta_lowdin.csv" % outfile_mark, mo_orth, label_row=label, label_col=label2_withmark)

label_mark = ["da#%i" % i for i in range(double_loc_a.shape[1])] \
           + ["sa#%i" % i for i in range(single_loc_a.shape[1])] \
           + ["uab1#%i" % i for i in range(empty_loc_ab1.shape[1])] \
           + ["ub2#%i" % i for i in range(empty_loc_b2.shape[1])] \
           + ["sb#%i" % i for i in range(single_loc_b.shape[1])] \
           + ["db#%i" % i for i in range(double_loc_b.shape[1])]
for i in range(len(label_mark)):
    label_mark[i] = ("#%i_" % i) + label_mark[i]

write_mulliken(mol, np.hstack((double_loc_a, single_loc_a, empty_loc_ab1, empty_loc_b2, single_loc_b, double_loc_b)),
                    outfile_mark = "local",
                    label_mark = label_mark)
