I will be slowly developing a command-line tool so that all of the converter scripts can be used with only one tool. These will be the converter options:

- `.lmp` to `.xyz`
- `.xyz` to `.lmp`
- `.lmp` to a modified `.xyz` format
- `.lammpstrj` to a `.xyz` trajectory
- `.lammpstrj` to a modified `.xyz` trajectory
- `.cif` to `.lmp` (experimental)
- `.lmp` to `.cif` (experimental)

## Data Prerequisites:
- `.xyz` files need to have the cell parameters in the comment line

During development, many of the scripts will be best suited for orthogonalized cells, but generalization to triclinic cells will be added with time. Conversion to `.vasp` (POSCAR) files will also eventually be added. Gromacs is another popular file type, but it will be added later. Codes like Atomsk or VASP exist, and will perform similar tasks. This code will have some extra features, however. Any of the customized files used in this repo will be added to the converter, the converter will be set up to allow batch conversion, and there will be an option to make a series of files specific to atom types in the system.

