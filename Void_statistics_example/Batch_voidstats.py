import os
from Void_statistics import analyze_xyz_file

input_dir = r"/input/dir"

main_output_dir = r"/output/dir"

n_wedges = 3

def main():
    os.makedirs(main_output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".xyz"):
            continue

        xyz_file = os.path.join(input_dir, filename)
        sample_name = os.path.splitext(filename)[0]
        file_output_dir = os.path.join(main_output_dir, f"{sample_name}_results")

        os.makedirs(file_output_dir, exist_ok=True)

        print(f"\nAnalyzing: {filename}")
        print(f"Output folder: {file_output_dir}")

        analyze_xyz_file(xyz_file=xyz_file,sample_name=sample_name,output_dir=file_output_dir,n_wedges=n_wedges)


if __name__ == "__main__":
    main()
