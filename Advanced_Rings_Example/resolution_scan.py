import os
import pandas as pd

from PCA_code import run_analysis, dataset_1, dataset_2, training_set1, training_set2

base_output_dir = r"/home/rielly/scratch/HDA_VHDA_PDs/LDA_fullcycle_fullringstats/PCA_regression_analysis/Test_calculations/0_to_6kbar/Resolution_test/w3"

Res_points = [100, 128, 150, 200, 256, 300]

summary = []

for res in Res_points:
    res_dir = os.path.join(base_output_dir, f"Resolution_{res}")
    os.makedirs(res_dir, exist_ok=True)

    result = run_analysis(
        resolution=res,
        output_dir=res_dir,
        dataset_1=dataset_1,
        dataset_2=dataset_2,
        training_set1=training_set1,
        training_set2=training_set2,
        weight_function="w3",
        sigma=0.05,
        n_components=20,
        test_size=0.2,
        random_state=69
    )

    summary.append(result)

pd.DataFrame(summary).to_csv(
    os.path.join(base_output_dir, "resolution_scan_summary.csv"),
    index=False
)

print("Resolution scan complete")