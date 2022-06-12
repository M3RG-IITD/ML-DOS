#LAMMPS simulation of MSD calculation for NS Glass

units           lj
dimension      	2
processors      * * *
boundary        p p p
atom_style	atomic

# read data

read_data	E_6_10000000_G_L62.dat


variable	dt equal 0.003

#inter-atomic potential#######################

include		pot.mod
timestep        ${dt}
change_box	all triclinic
reset_timestep  0

###############################################

# Compute settings ----------------------------

compute dispatom all displace/atom

# Output settings ----------------------------

# Display thermo


thermo 			1000
thermo_style		custom step temp ke pe press density   


fix      	4 all box/relax tri 0.0
min_style	cg
min_modify	line quadratic
minimize	1.0e-10 1.0e-10 100000 100000
unfix           4

fix      	3 all npt temp 0.01 0.001 0.15 tri 0.0 0.0 1.5
run           	1000						#Try different values
unfix         	3



velocity        all scale 0.25 
reset_timestep	0
thermo 			1000
fix     		1 all nvt temp 0.25 0.25 0.15
dump			atom_disp all custom/gz 1000 atom_disp1.atom.gz id type x y z c_dispatom[*]  

run 			100000

dump			Last_atom_disp all custom 1 atom_disp_final.atom id type x y z c_dispatom[*]  
run			0

