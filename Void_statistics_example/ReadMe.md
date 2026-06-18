This example will involve a different type of persistent homology analysis. Instead of the ring analysis that has been emphasized, this will look at cage features found in the second persistent homology group. Data relevant to this calculation can be found in the Input_files folder.

The script `Void_statistics.py` will contain the code needed for the entire analysis, while `Batch_voidstats.py` will iterate through each of the input files. A folder specific to each file will be made and will contain all of the results related to that input file. The cage features of interest are:

- Atoms per cage
- Cage volume
- Cage radius
- Faces per cage

This script is incomplete and will be developed over time. At present, it is able to read from the input files, generate the 2D persistence diagrams, and bin the data correctly. In most glasses, the 2D PD will look like a fan. The initial steps of this analysis will be to determine the "origin" of the fan (point closest to (0,0)), and find the minimum fan angle required to bypass the points in the PD. The fan angle is relative to the birth-death line. The data will be "radially binned" in the sense that the large fan will be divided into N wedges. The void statistics will be performed on the data points in each wedge.
