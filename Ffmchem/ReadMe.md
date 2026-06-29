I will be slowly developing a command-line tool so that all of the converter scripts can be used with only one tool. These will be the converter options:

- `.lmp` ↔ `.xyz`
- `.lmp` to a modified `.xyz` format
- `.lammpstrj` to a `.xyz` trajectory
- `.lammpstrj` to a modified `.xyz` trajectory
- `.cif` ↔ `.lmp` 
- `.xyz` ↔ `.cif` 
- `.gro` ↔ `.xyz` (experimental)

## Data Prerequisites:
- `.xyz` files need to have the cell parameters in the comment line.
- The user must specify the lattice type and space group when converting to `.cif ` format.
- Users need to provide an atom map for the code (ex. "H:1, O:2).
- Users need to provide an atomic mass map when converting to `.lmp` format.

During development, many of the scripts will be best suited for orthogonalized cells, but generalization to triclinic cells will be added with time. Conversion to `.vasp` (POSCAR) files will also eventually be added. Gromacs is another popular file type, but it will be added later. Codes like Atomsk or VASP exist and will perform similar tasks. This code will have some extra features, however. Any of the customized files used in this repo will be added to the converter, the converter will be set up to allow batch conversion, and there will be an option to make a series of files specific to atom types in the system.

