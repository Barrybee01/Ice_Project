import os
import pandas as pd

from PCA_code import run_analysis, dataset_1, dataset_2, training_set1, training_set2

base_output_dir = r"/path/to/out/dir"

summary = []

result = run_analysis(resolution=300, output_dir=base_output_dir,dataset_1=dataset_1,dataset_2=dataset_2,training_set1=training_set1,training_set2=training_set2,
        weight_function="w2", sigma=0.05, n_components=20, test_size=0.2, random_state=69)

summary.append(result)

pd.DataFrame(summary).to_csv(os.path.join(base_output_dir, "summary.csv"), index=False)

print("Analysis complete")
