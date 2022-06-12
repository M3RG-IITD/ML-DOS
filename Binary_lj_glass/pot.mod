#LJ Potential for binary glass
pair_style	lj/cut 2.5
pair_coeff	1 1 1.0 1.0 2.5		#e_AA=1.0 , sigma_AA=1.0, rc_AA=2.5*e_AA=2.5
pair_coeff	1 2 1.5 0.8 2.0		#e_AB=1.5 , sigma_AB=0.8, rc_AB=2.5*e_AB=2.0
pair_coeff	2 2 0.5 0.88 2.2	#e_AB=0.5 , sigma_BB=0.88, rc_BB=2.5*e_BB=2.2

neighbor        2.0 bin
neigh_modify    every 1 check yes





