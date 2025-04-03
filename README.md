# projection_embedding

Manby's projection embedding method implemented with PySCF. DMRG/DMRG-NEVPT2/FCIQMC as WFT solver implemented. 

## Dependences: 

* python3

* pyscf (ver 2.1.x) and all its dependencies

* pyscf-dmrgscf (For DMRG runs in 3_dmrgcas/ and 5_dmrg_nevpt2/) 	https://github.com/pyscf/dmrgscf
 
* Block2 (For DMRG runs in 3_dmrgcas/ and 5_dmrg_nevpt2/)  https://block2.readthedocs.io/en/latest/
 
* NECI (For FCIQMC runs in 4_fciqmc/)  https://github.com/ghb24/NECI_STABLE

## Citation:         

For this code: Jiang, T.; Bogodanov, N. A. ; Alavi, A. ; Chen, J.  Individual and synergistic superexchange enhancement in cuprates. Preprint at [https://arxiv.org/abs/2503.17596](https://arxiv.org/abs/2503.17596) (2025). 

For dependences: 

pyscf: 

Q. Sun, X. Zhang, S. Banerjee, P. Bao, M. Barbry, N. S. Blunt, N. A. Bogdanov, G. H. Booth, J. Chen, Z.-H. Cui, J. J. Eriksen, Y. Gao, S. Guo, J. Hermann, M. R. Hermes, K. Koh, P. Koval, S. Lehtola, Z. Li, J. Liu, N. Mardirossian, J. D. McClain, M. Motta, B. Mussard, H. Q. Pham, A. Pulkin, W. Purwanto, P. J. Robinson, E. Ronca, E. R. Sayfutyarova, M. Scheurer, H. F. Schurkus, J. E. T. Smith, C. Sun, S.-N. Sun, S. Upadhyay, L. K. Wagner, X. Wang, A. White, J. Daniel Whitfield, M. J. Williamson, S. Wouters, J. Yang, J. M. Yu, T. Zhu, T. C. Berkelbach, S. Sharma, A. Yu. Sokolov, and G. K.-L. Chan, Recent developments in the PySCF program package, <em>J. Chem. Phys.</em> <strong>153</strong>, 024109 (2020)

block2: 

Zhai, H.; Larsson, H. R.; Lee, S.; Cui, Z.; Zhu, T.; Sun, C.; Peng, L.; Peng, R.; Liao, K.; Tölle, J.; Yang, J.; Li, S.; Chan, G. K.-L. Block2: a comprehensive open source framework to develop and apply state-of-the-art DMRG algorithms in electronic structure and beyond, <em>J. Chem. Phys.</em> <strong>159</strong>, 234801 (2023)

neci: 

Guther, K.; Anderson, R. J.; Blunt, N. S.; Bogdanov, N. A.; Cleland, D.; Dattani, N.; Dobrautz, W.; Ghanem, K.; Jeszenszki, P.; Liebermann, N.; Manni, G. L.; Lozovoi, A. Y.; Luo, H.; Ma, D.; Merz, F.; Overy, C.; Rampp, M.; Samanta, P. K.; Schwarz, L. R.; Shepherd, J. J.; Smart, S. D.; Vitale, E.; Weser, O.; Booth, G. H.; Alavi, A. NECI: N -Electron Configuration Interaction with an Emphasis on State-of-the-Art Stochastic Methods <em>J. Chem. Phys.</em> <strong>153</strong>, 3, 034107 (2020)

For embedding method: 

Manby, F. R.; Stella, M.; Goodpaster, J. D.; Miller, T. F. A Simple, Exact Density-Functional-Theory Embedding Scheme. <em>J. Chem. Theory Comput.</em> <strong>8</strong>, 8, 2564–2568 (2012)

## Use: 

    ## Construction of embedded Hamiltonian
    cd 0_hf
    python 0a_rohf.py > 0a_rohf.out
    python 0b_mixedrhf.py > 0b_mixedrhf.out
    cd ../1_pm
    python 1a_PM.py > 1a_PM.out
    cd ../2_proj
    python 2a_proj.py > 2a_proj.out

    ## Construction of embedded Hamiltonian
    cd 0_hf
    python 0a_rohf.py > 0a_rohf.out
    python 0b_mixedrhf.py > 0b_mixedrhf.out
    cd ../1_pm
    python 1a_PM.py > 1a_PM.out
    cd ../2_proj
    python 2a_proj.py > 2a_proj.out

    # Perform CASSCF calculation
    cd ../3_cas
    python 3a_avas4e3o.py > 3a_avas4e3o.out
    
    # Perform DMRG-MCSCF calculation
    # Install Block2 and pyscf-dmrgscf first
    cd ../3_dmrgcas
    python 3b_PM24e26o.py > 3b_PM24e26o.out
    
    # Perform FCIQMC calculation
    # Install NECI first
    cd ../../4_fciqmc/4a_cas24e26o
    python 4a_FCIDUMPgen.py > 4a_FCIDUMPgen.out
    # Then go to walker_xM_sx/ folders to perform NECI calculations with the generated FCIDUMP files. 
    
    # Perform DMRG-NEVPT2 calculation
    # Install Block2 and pyscf-dmrgscf first
    cd ../../5_dmrg_nevpt2
    cd 5a_fullcas
    python 5a1_casgen.py > 5a1_casgen.out # Generate CAS and PT space first. 
    cd s0
    python 5a2.py > 5a2.out
    cd ../s1
    python 5a2.py > 5a2.out
    # You can go to 5b_coreonly, 5c_coreonly or 5da_coreCuO_virtAll folders to perform other PT space calculations. 
    
## Some notices: 

* <strong>All .py scripts should be run in pyscf ver 2.1.0 or 2.1.1! They don't work in pyscf newer than ver 2.4.0.</strong>

* .log files contain the main output information of the corresponding .py script.
Additional outputs include data in HDF5 format (.chk), MOLDEN format (.molden) and CSV format (.csv). Their meanings are described in detail as follows. 

## Flowchart: 

0_hf/         HF runs as a starting point. 

La214_Grande.clust.xyz        Atom position file. 
La214_Grande.evjen.lat        Point charge position and charge number file. 
0a_rohf.py        An ROHF run on the whole cluster as the initial guess for mixed-RHF run. In this step, 8 open-shell electrons are set to be spin-up. 

	Dependence: La214_Grande.clust.xyz, La214_Grande.evjen.lat          Structure input. 
	Output: 0a_rohf.log, 0a_rohf.out        Main output. 
                       0a_rohf_rohf.molden        Converged ROHF MO coefficients, energy and occupation in MOLDEN format. 
                       0a_rohf-rohf0.chk        MO coefficients, energy and occupation of a brief, not-converged ROHF run for initialization, in HDF5 format.
                       0a_rohf-rohf1.chk        Converged ROHF MO coefficients, energy and occupation in HDF5 format. 

0b_mixedrhf.py        A mixed-RHF run on the whole cluster as the basis of embedding. 8 open-shell electrons are set to be spinlessly single-occupied. 

        Dependence: La214_Grande.clust.xyz, La214_Grande.evjen.lat          Structure input. 
                        0a_rohf-rohf0.chk        MO coefficients to restart from. 
        Output: 0b_rohf.log, 0b_rohf.out        Main output. 
                       0b_mixedrhf_rhf_mixed1.molden        Converged Mixed-RHF MO coefficients, energy and occupation in MOLDEN format. 
                       0b_mixedrhf-rhf_mixed1.chk        Converged Mixed-RHF MO coefficients, energy and occupation in HDF5 format. 

1_pm/        Pipek-Mezey localization.

1a_PM.py        Partition mixed-RHF 2-occ, 1-occ and 0-occ orbitals into A subsystem and B subsystem. 

        Dependence: La214_Grande.clust.xyz, La214_Grande.evjen.lat          Structure input. 
                        ../0_hf/0b_mixedrhf-rhf_mixed1.chk        MO to perform PM localization on. 
        Output: 1a_PM.log, 1a_PM.out       Main output. 
                       1a_PM.chk         Localized MO coefficients in HDF5 form. 
                       1a_PM_double_occ_A.molden, 1a_PM_double_occ_B.molden, 1a_PM_empty_occ.molden, 1a_PM_empty_occ_AB1.molden, 1a_PM_empty_occ_B2.molden, 1a_PM_single_occ_A.molden, 1a_PM_single_occ_B.molden        Localized MO coefficients stored in MOLDEN form. "double_occ", "single_occ" and "empty_occ" refer to 2-occ, 1-occ and 0-occ orbitals from Mixed-RHF, and A/B refer to subsystem A or B. 
                       1a_PMlocal_meta_lowdin_orth.csv        Meta-Lowdin AO of the system. 
                       1a_PMlocal_mo_in_meta_lowdin.csv        Localized orbitals in meta-Lowdin AO basis. 
                       
2_proj/         Projection embedding potential and energy.

	Dependence: La214_Grande.clust.xyz, La214_Grande.evjen.lat          Structure input. 
                        ../1_pm/1a_PM.chk        Localized MO that define the subsystem density matrices $\gamma_A$ and $\gamma_B$. 
        Output: 2a_proj.log, 2a_proj.out      Main output. 
                      2a_proj_rhf_mixed.chk        Embedding potential and energy in HDF5 form. 

3_cas/        Deterministic CASSCF run. 

3a_avas4e3o.py       CASSCF on CAS(4e,3o) with AVAS as CAS initializer. MO coefficients are optimized respectively for single state and triplet state. 

        Dependence: La214_Grande.clust.xyz, La214_Grande.evjen.lat          Structure input. 
                        ../2_proj/2a_proj_rhf_mixed.chk        Embedding potential and energy. 
        Output: 3a_avas4e3o.log, 3a_avas4e3o.out        Main output. 
                       3a_avas4e3o_rhf_mixed.chk        MCSCF outputs, including optimized MO coeff and CI coeff. 
                       3a_avas4e3o_rhf_mixed.molden        MCSCF-optimized MO coeff  in MOLDEN format. 
                       3a_avas4e3o4e3o_avas.molden        MCSCF initial guess of MO coeff coming from AVAS. 

3_dmrgcas/        DMRG-MCSCF run. 

3b_PM24e26o.py        MCSCF (24e, 26o) with DMRG as FCI solver. MO coefficients are optimized respectively for single state and triplet state.
 
        Dependence: La214_Grande.clust.xyz, La214_Grande.evjen.lat          Structure input. 
                        ../1_pm/1a_PM.chk        Pipek-Mezey localization as MCSCF initial guess. 
                        ../2_proj/2a_proj_rhf_mixed.chk        Embedding potential and energy. 
        Output: 3b_PM24e26o.log, 3b_PM24e26o.out        Main output. 
                       3b_PM24e26o_s0.chk, 3b_PM24e26o_s1.chk        MCSCF-optimized MO coeff for S=0 and S=1, respectively, in HDF5 format. 
                       3b_PM24e26o_s0.molden, 3b_PM24e26o_s1.molden        MCSCF-optimized MO coeff  for S=0 and S=1, respectively, in MOLDEN format. 
                       3b_PM24e26o_init.molden        MCSCF initial guess of MO coeff coming from PM localization. 

4_fciqmc/         FCIQMC run. 
         4a_cas24e26o/          FCIQMC run in CAS(24e,26o)

         4a_FCIDUMPgen.py        Script for FCIDUMP generation in the presence of embedding potential. 

                Dependence: La214_Grande.clust.xyz, La214_Grande.evjen.lat          Structure input. 
                                ../../2_proj/2a_proj_rhf_mixed.chk        Embedding potential and energy. 
                                ../../3_dmrgcas/3b_PM24e26o_s0.chk        DMRG-MCSCF-optimized MO coeff for S=0, as active space for FCIQMC run. 
                                ../../3_dmrgcas/3b_PM24e26o_s1.chk        DMRG-MCSCF-optimized MO coeff for S=1, as active space for FCIQMC run. 
                Output: 4a_FCIDUMPgen.log, 4a_FCIDUMPgen.out        Main output. 
                               4a_FCIDUMPgen_s0.fcidump        FCIDUMP file for S=0.
                               4a_FCIDUMPgen_s1.fcidump        FCIDUMP file for S=1.  

walker_xM_sy/          NECI input and output files for walker=xM and spin=y. 

5_dmrg_nevpt2/        DMRG+NEVPT2 run. 
         5a_fullcas/         CAS(24e,26o) as active space, and entire A subsystem orbital space as PT space. 
         
         5a1_casgen.py        Generate the PT space according to meta-Lowdin AO projected onto DMRG-MCSCF MO space. 

                Dependence: La214_Grande.clust.xyz, La214_Grande.evjen.lat          Structure input. 
                                ../../2_proj/2a_proj_rhf_mixed.chk        Embedding potential and energy. 
                                ../../3_dmrgcas/3b_PM24e26o_s0.chk        DMRG-MCSCF-optimized MO coeff for S=0, as active space for FCIQMC run. 
                                ../../3_dmrgcas/3b_PM24e26o_s1.chk        DMRG-MCSCF-optimized MO coeff for S=1, as active space for FCIQMC run. 
                Output: 5a1_casgen.log        Main output. 
                               5a1_casgen.chk       Orbitals for NEVPT2 calculation, including frozen core, PT core, active, PT virtual, frozen virtual in order, in HDF5 format. 
                               5a1_casgen_fullcas_s0.molden, 5a1_casgen_fullcas_s1.molden        All orbitals in MOLDEN format, including frozen core, PT core, active, PT virtual, frozen virtual in order, in MOLDEN format. 
                               5a1_casgen_mowith_fullcas_s0.molden, 5a1_casgen_mowith_fullcas_s1.molden      PT core and PT virtual only, in MOLDEN format. 
                               
                s0/        S=0 calculations

                5a2.py        Perform DMRG+NEVPT2 calculation for S=0

                        Dependence: La214_Grande.clust.xyz, La214_Grande.evjen.lat          Structure input. 
                                        ../../../2_proj/2a_proj_rhf_mixed.chk        Embedding potential and energy. 
                                        ../5a1_casgen.chk        Selected AO for NEVPT2 calculation, which contain CAS and PT orbitals with given AO label. 

                        Output: 5a2.log, 5a2.out        Main output. 
                        
                s1/        S=1 calculations
        
        5b_coreonly/ 
                Similar format to 5a_fullcas/ inside. PT space is set to include only core orbitals. 

        5c_virtonly/
                Similar format to 5a_fullcas/ inside. PT space is set to include only virtual orbitals. 

        5da_coreCuO_virtAll/
                Similar format to 5a_fullcas/ inside. PT space is set to include coreCuO orbitals and all virtual orbitals. 


