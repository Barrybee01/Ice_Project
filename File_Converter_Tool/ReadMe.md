I will be slowly developing a command line tool so that all of the converter scripts can be used with only one tool. These will be the converter options:

- `.lmp` to `.xyz`
- `.xyz` to `.lmp`
- `.lmp` to a modified `.xyz` format
- `.lammpstrj` to a `.xyz` trajectory
- `.lammpstrj` to a modified `.xyz` trajectory
- `.cif` to `.lmp` (experimental)
- `.lmp` to `.cif` (experimental)

This tool only accepts orthogonalized structure files. Triclinic cells will not work. This feature ***might*** be added in the future but it is not going to be prioritized. Conversion to Gromacs and VASP files will be added in the future

