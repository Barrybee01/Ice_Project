This will cover a basic example of persistent homology analysis. This will follow closely to an already available [Homcloud example](https://homcloud.dev/py-tutorial/ml_pc_3d_visualization_by_plotly_en.html)

This process begins with the LAMMPS trajectory in the form `dump.lammpstrj`. This will contain a series of `.lmp` files representing every frame of the MD simulation that has been stored. Because some of the regression analysis used requires an ML model, training data is needed.

Creation of the training data can be performed with the following steps:
- Obtain `dump.lammpstrj` file from LAMMPS simulation
- Perform the command `tail -n N dump.lammpstrj > training_set.txt`, where N is the total number of lines in the trajectory you want to use. This can be determined with the general expression; N = (lines per frame + 9)*(number of frames you want to use)
- To separate the individual frames, use the command `split -l M -d -a 3 --additional_suffix=.lmp training_set.txt step_`. This will create a series of `.lmp` files.
- Use the `lmp_to_xyz.py` script in a loop to convert the training data files into `.xyz` format.

From this point onward, we will primarily use the `Ring_PCA_analysis.py` script.
