import argparse
from pathlib import Path
from cif_to_lmp import cif_to_lmp
from cif_to_xyz import cif_to_xyz
from gro_to_xyz import gro_to_xyz
from lammpstrj_to_modified_xyz import lammpstrj_to_modified_xyz
from lammpstrj_to_xyz import lammpstrj_to_xyz
from lmp_to_cif import lmp_to_cif
from lmp_to_modified_xyz import lmp_to_modified_xyz
from lmp_to_xyz import lmp_to_xyz
from xyz_to_cif import xyz_to_cif
from xyz_to_gro import xyz_to_gro
from xyz_to_lmp import xyz_to_lmp


EXTENSIONS = {
    "cif": ".cif",
    "lmp": ".lmp",
    "xyz": ".xyz",
    "gro": ".gro",
    "lammpstrj": ".lammpstrj",
}

def parse_atom_map(map_args):
    if map_args is None:
        return {}, {}

    type_map = {}
    symbol_map = {}

    for item in map_args:
        atom_type, symbol = item.split(":")
        atom_type = int(atom_type)
        type_map[atom_type] = symbol
        symbol_map[symbol] = atom_type
    return type_map, symbol_map

def parse_mass_map(mass_args):
    if mass_args is None:
        return {}

    mass_map = {}

    for item in mass_args:
        atom_type, mass = item.split(":")
        mass_map[int(atom_type)] = float(mass)

    return mass_map

def split_xyz_by_atom_type(xyz_file, output_dir, output_name=None):
    xyz_file = Path(xyz_file)
    output_dir = Path(output_dir)

    if output_name is None:
        output_name = xyz_file.name

    with open(xyz_file, "r") as infile:
        lines = infile.readlines()

    frames = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue

        n_atoms = int(lines[i].strip())
        comment = lines[i + 1]
        atom_lines = lines[i + 2:i + 2 + n_atoms]

        frames.append((comment, atom_lines))
        i += 2 + n_atoms

    atom_types = sorted(set(line.split()[0] for _, atom_lines in frames for line in atom_lines))

    for atom_type in atom_types:
        atom_dir = output_dir / f"atom_{atom_type}"
        atom_dir.mkdir(parents=True, exist_ok=True)
        atom_output_file = atom_dir / output_name

        with open(atom_output_file, "w") as outfile:
            for comment, atom_lines in frames:
                selected_atoms = [line for line in atom_lines if line.split()[0] == atom_type]
                outfile.write(f"{len(selected_atoms)}\n")
                outfile.write(comment)
                for line in selected_atoms:
                    outfile.write(line)

def run_single_conversion(args, input_file, output_file):
    input_file = Path(input_file)
    output_file = Path(output_file)
    type_map, symbol_map = parse_atom_map(args.map)
    mass_map = parse_mass_map(args.mass_map)

    if args.input_format == "cif" and args.output_format == "lmp":
        cif_to_lmp(input_file, output_file, symbol_map=symbol_map, mass_map=mass_map)

    elif args.input_format == "lmp" and args.output_format == "cif":
        lmp_to_cif(input_file,output_file,type_map=type_map,lattice_type=args.lattice_type,space_group=args.space_group)

    elif args.input_format == "lmp" and args.output_format == "xyz":
        if args.modified:
            lmp_to_modified_xyz(input_file, output_file)
        else:
            lmp_to_xyz(input_file, output_file, type_map=type_map)

    elif args.input_format == "xyz" and args.output_format == "lmp":
        xyz_to_lmp(input_file, output_file, symbol_map=symbol_map, mass_map=mass_map)

    elif args.input_format == "xyz" and args.output_format == "cif":
        xyz_to_cif(input_file,output_file,lattice_type=args.lattice_type, space_group=args.space_group)

    elif args.input_format == "cif" and args.output_format == "xyz":
        cif_to_xyz(input_file, output_file)

    elif args.input_format == "xyz" and args.output_format == "gro":
        xyz_to_gro(input_file,output_file,residue_name=args.residue_name,title=args.title)

    elif args.input_format == "gro" and args.output_format == "xyz":
        gro_to_xyz(input_file, output_file)

    elif args.input_format == "lammpstrj" and args.output_format == "xyz":
        if args.modified:
            lammpstrj_to_modified_xyz(input_file,output_file,coordinate_mode=args.coordinates)
        else:
            lammpstrj_to_xyz(input_file,output_file,type_map=type_map,coordinate_mode=args.coordinates)

    else:
        raise ValueError(f"Unsupported conversion: {args.input_format} to {args.output_format}")

def run_atom_centric_conversion(args, input_file, output_target):
    input_file = Path(input_file)
    output_target = Path(output_target)

    if args.output_format != "xyz":
        raise ValueError("--atom-centric currently only supports XYZ output.")

    if output_target.suffix:
        output_dir = output_target.parent
        output_name = output_target.name
    else:
        output_dir = output_target
        output_name = input_file.with_suffix(".xyz").name

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_file = output_dir / f"__tmp_{output_name}"
    run_single_conversion(args, input_file, temp_file)
    split_xyz_by_atom_type(temp_file, output_dir, output_name=output_name)
    temp_file.unlink()

def get_batch_output_file(input_file, input_dir, output_dir, output_format, modified=False):
    input_file = Path(input_file)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    stem = input_file.relative_to(input_dir).with_suffix("").name

    if modified:
        stem = f"{stem}_modified"
    return output_dir / f"{stem}{EXTENSIONS[output_format]}"

def run_batch_conversion(args):
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_extension = EXTENSIONS[args.input_format]
    input_files = sorted(input_dir.glob(f"*{input_extension}"))

    if not input_files:
        raise ValueError(f"No {input_extension} files found in {input_dir}")

    for input_file in input_files:
        output_file = get_batch_output_file(input_file, input_dir,output_dir,args.output_format,modified=args.modified,)

        if args.atom_centric:
            run_atom_centric_conversion(args, input_file, output_file)
        else:
            run_single_conversion(args, input_file, output_file)

        print(f"Converted: {input_file}")

def main():
    parser = argparse.ArgumentParser(description="Ffmchem: Command-line tool for converting between common atomistic structure file formats.")

    parser.add_argument("--input",required=True,help="Input file for single conversion, or input directory when using --batch.")
    parser.add_argument("--output",required=True,help="Output file for single conversion, or output directory when using --batch.")
    parser.add_argument("--from",dest="input_format",required=True,choices=["cif", "lmp", "xyz", "gro", "lammpstrj"],help="Input file format.")
    parser.add_argument("--to",dest="output_format",required=True,choices=["cif", "lmp", "xyz", "gro"],help="Output file format.")
    parser.add_argument("--map",nargs="+",default=None,help="Atom type mapping. Example: --map 1:O 2:H")
    parser.add_argument("--mass-map",nargs="+",default=None,help="Atomic masses for LAMMPS data files. Example: --mass-map 1:15.999 2:1.008")
    parser.add_argument("--batch",action="store_true",help="Convert every matching file in the input directory.")
    parser.add_argument("--modified",action="store_true",help="Use the modified XYZ format (numeric atom types).")
    parser.add_argument("--atom-centric", action="store_true", help="Produce one output file for each atom type. Output files are placed into atom-specific folders.")
    parser.add_argument("--coordinates", choices=["auto", "scaled", "unscaled"], default="auto", help="Coordinate convention used in LAMMPS trajectory files. 'auto' detects the format automatically.")
    parser.add_argument("--lattice-type",default="triclinic",help="CIF lattice type written to _symmetry_cell_setting (default: orthorhombic).")
    parser.add_argument("--space-group",default="P 1",help="CIF space group written to _symmetry_space_group_name_H-M (default: 'P 1').")
    parser.add_argument("--residue-name",default="MOL",help="Residue name used when writing GRO files (default: MOL).")
    parser.add_argument("--residue-size",type=int,default=1,help="Number of atoms assigned to each residue when writing GRO files (default: 1).",)
    parser.add_argument("--title",default="Generated by Ffmchem",help="Title written to the first line of GRO files.",)

    args = parser.parse_args()

    if args.atom_centric and args.output_format != "xyz":
        raise ValueError("--atom-centric currently only supports XYZ output.")

    if args.batch:
        run_batch_conversion(args)
    else:
        if args.atom_centric:
            run_atom_centric_conversion(args,Path(args.input),Path(args.output))
        else:
            run_single_conversion(args,Path(args.input),Path(args.output))

if __name__ == "__main__":
    main()
