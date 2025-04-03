import numpy as np
from pyscf.lo import orth
from functools import reduce

def dm_meta_lowdin_ao(mol, dm, s=None, orth_coeff=None):
    label = mol.ao_labels()
    if s is None:
        s = mol.intor_symmetric("int1e_ovlp")
    if orth_coeff is None:
        orth_coeff = orth.orth_ao(mol, 'meta_lowdin', s=s)
    c_inv = np.dot(orth_coeff.conj().T, s)
    if isinstance(dm, np.ndarray) and dm.ndim == 2:
        dm = reduce(np.dot, (c_inv, dm, c_inv.T.conj()))
    else:  # ROHF
        dm = reduce(np.dot, (c_inv, dm[0]+dm[1], c_inv.T.conj()))
    return label, dm

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
    
def save_csv(csv_file, c, label_row=None, label_col=None):
    with open(csv_file, "w") as f:
        if label_row is None:
            label_row = ["#%i" % i for i in range(c.shape[0])]
        if label_col is None:
            label_col = ["#%i" % i for i in range(c.shape[1])]
        f.write('label,' + ','.join(label_col) + '\n')
        for lab, row in zip(label_row, c):
            f.write(','.join((lab, *["%.10f" % c_elem for c_elem in row])) +'\n')
        f.write('\n')
