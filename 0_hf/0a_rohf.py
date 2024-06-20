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

sys.path.append("..")

from orth_csv import save_csv, c_meta_lowdin_ao, dm_meta_lowdin_ao

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
    spin=1*8
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
    save_csv(os.path.join(outdir, "meta_lowdin_orth.csv"), orth_coeff, label_row=label, label_col=label)
    _, mo_coeff_orth = c_meta_lowdin_ao(mf.mol, mf.mo_coeff, s=s, orth_coeff=orth_coeff)
    _, dm_orth = dm_meta_lowdin_ao(mf.mol, mf.make_rdm1(), s=s, orth_coeff=orth_coeff)

    mo_coeff_orth_extended = np.vstack((mf.mo_energy, mf.mo_occ, mo_coeff_orth))
    label_extended = np.hstack((["mo_energy", "mo_occ"], label))

    save_csv(os.path.join(outdir, "mo_coeff.csv"), mo_coeff_orth_extended, label_row=label_extended)
    save_csv(os.path.join(outdir, "dm.csv"), dm_orth, label_row=label, label_col=label)

    molden.from_scf(mf, os.path.join(outdir, "scf.molden"))


# ROHF try

mf = mm_charge(scf.ROHF(mol), coords, charges)
mf.chkfile = project+"-rohf0.chk"
# mf.irrep_nelec = occMolcas
mf.init_guess = "huckel"
# mf.init_guess = "hcore"
# mf.init_guess = "atom"
mf.max_cycle = 15
mf.dump_flags = lambda *args, **kwargs: dump_flags(mf, *args, **kwargs)
mf.kernel()

restart_chk = mf.chkfile
mo_coeff = scf.chkfile.load(restart_chk, "scf/mo_coeff")
mo_occ = scf.chkfile.load(restart_chk, "scf/mo_occ")

mf = mm_charge(scf.ROHF(mol), coords, charges).newton()
# mf.__dict__.update(scf.chkfiale.load(project+"-rohf0.chk", "scf"))
mf.chkfile = project+"-rohf1.chk"
mf.verbose = 5
# mf.irrep_nelec = occMolcas
mf.kernel(mo_coeff, mo_occ)

# Orbital energies, Mulliken population etc.
try:
    mf.analyze()
except:
    print("no analysis for now")
print("E(HF) = %.12f" % mf.e_tot)

molden.from_scf(mf, project+"_rohf.molden")
write_mulliken(mf, project+"-rohf-mulliken")


# noons, natorbs = mcscf.addons.make_natural_orbitals(mf)

# molden.from_mo(mol, project+"-rona.molden", natorbs, occ=noons)

# chkfile = project+"-rona.chk"
# fh5 = h5py.File(chkfile, 'w')
# fh5['mol'] = mf.mol.dumps()
# fh5['mcscf/mo_coeff'] = natorbs
# fh5['mcscf/mo_occ'] = noons
# fh5.close()
