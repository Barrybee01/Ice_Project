import torch
import numpy as np
from tqdm import tqdm
from collections import deque


def compare(x, y):
    return x - y > 0


import torch
from tqdm import tqdm
from collections import deque
import time

def atoms_per_cell(data_points, atom_types, subdivisions, batch_size=1):
    # Number of subdivisions
    num_subcells = len(subdivisions)
    
    # Initialize counts for total, oxygen, and hydrogen atoms per subcell
    total_counts = torch.zeros(num_subcells, dtype=torch.int).cuda()
    oxygen_counts = torch.zeros(num_subcells, dtype=torch.int).cuda()
    hydrogen_counts = torch.zeros(num_subcells, dtype=torch.int).cuda()
    
    # Atom type tensors for comparison
    H, O = torch.tensor(ord("H"), dtype=torch.int), torch.tensor(ord("O"), dtype=torch.int)
    at = torch.tensor([ord(i) for i in atom_types], dtype=torch.int).cuda()
    
    # Initialize subcell origins and boundaries
    sub_origins = torch.zeros(num_subcells, 3, dtype=torch.float32)
    bound_points = torch.zeros(num_subcells, 3, dtype=torch.float32)
    
    # Convert data points to tensor and move to GPU
    data_points_tensor = torch.tensor(data_points, dtype=torch.float32).pin_memory().cuda()
    
    # Process each subdivision to get origins and boundaries
    for i, subcell in tqdm(enumerate(subdivisions), desc="Iterating through subdivisions...."):
        sub_origins[i, :], bound_points[i, :] = torch.tensor(subcell[0]), torch.cat(subcell[1:])
    
    del data_points
    
    # Determine batch count and split data into batches
    batch_count = sub_origins.shape[0] // batch_size
    subcounts = torch.arange(sub_origins.shape[0])
    subcounts = subcounts.chunk(batch_count)
    sub_origins = sub_origins.chunk(batch_count)
    bound_points = bound_points.chunk(batch_count)
    
    # Initialize array to hold coordinates per subcell
    coordinates_per_subcell = torch.zeros(num_subcells, data_points_tensor.shape[0], dtype=torch.int)
    valid_counts = torch.zeros(num_subcells, dtype=torch.int).cuda()
    times = torch.zeros(batch_count)
    
    # Process each batch
    for i in tqdm(range(len(subcounts)), desc=f"iterating through batches {i}"):
        start = time.perf_counter()
        
        # Call coordinate function (coord_fn) to get atom counts and valid coordinates
        a, b, c, d, e = coord_fn(sub_origins[i], data_points_tensor, bound_points[i], at)
        if a is None:
            continue
        
        # Update counts and coordinates for the current batch
        total_counts[subcounts[i]], oxygen_counts[subcounts[i]], hydrogen_counts[subcounts[i]], coordinates_per_subcell[subcounts[i], :d.shape[1]], valid_counts[subcounts[i]] = (a, b, c, d.cpu(), e)
        
        # Record time taken for the batch
        times[i] = time.perf_counter() - start
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    print(f"runtime: ", sum(times))
    
    # Prepare for returning results
    ncpc = deque()
    valid_counts = valid_counts.cpu()
    coordinates_per_subcell = coordinates_per_subcell.cpu()
    total_counts = total_counts.cpu()
    oxygen_counts = oxygen_counts.cpu()
    hydrogen_counts = hydrogen_counts.cpu()
    data_points_tensor = data_points_tensor.cpu()
    at = at.unsqueeze(1).cpu()
    
    # Stack counts for total, oxygen, and hydrogen atoms
    analytics = torch.stack([total_counts, oxygen_counts, hydrogen_counts], dim=1)
    dat = torch.cat([at, data_points_tensor], dim=-1)
    
    # Collect valid coordinates and corresponding data points
    for k in tqdm(torch.nonzero(valid_counts).squeeze(1), desc="postprocessing"):
        ncpc.append(torch.as_tensor(dat[coordinates_per_subcell[k, 0 : valid_counts[k]]]))
    
    return ncpc



def coord_fn(sub_origins, data_points_tensor, bound_points, at):
    # Initialize atom type tensors for comparison
    H = torch.tensor(ord("H"), dtype=torch.int)
    O = torch.tensor(ord("O"), dtype=torch.int)
    
    # Number of subcells
    num_subcells = len(sub_origins)
    
    # Initialize counts for total, oxygen, and hydrogen atoms per subcell
    total_counts = torch.zeros(num_subcells, dtype=torch.int).cuda()
    oxygen_counts = torch.zeros(num_subcells, dtype=torch.int).cuda()
    hydrogen_counts = torch.zeros(num_subcells, dtype=torch.int).cuda()
    
    # Expand subcell origins to match the shape of data points tensor
    sub_origins = sub_origins.pin_memory().cuda().unsqueeze(1).expand(-1, data_points_tensor.shape[0], -1)
    
    # Compare data points with subcell origins
    sm = compare(data_points_tensor, sub_origins)
    
    # Expand bound points to match the shape of data points tensor
    bound_points = bound_points.pin_memory().cuda().unsqueeze(1).expand(-1, data_points_tensor.shape[0], -1)
    
    # Compare data points with bound points
    bm = compare(bound_points, data_points_tensor)
    
    # Create masks to determine if points are within bounds
    min_mask = torch.sum(sm, dim=-1, dtype=torch.int) == data_points_tensor.shape[-1]
    max_mask = torch.sum(bm, dim=-1, dtype=torch.int) == bound_points.shape[-1]
    masks = torch.zeros(min_mask.shape, dtype=torch.bool, device=torch.device("cuda")) + min_mask * max_mask
    
    # Generate master mask for valid data points
    master_mask = torch.arange(data_points_tensor.shape[0]).cuda().unsqueeze(0).expand(max_mask.shape[0], -1) * masks
    
    # Initialize valid counts for subcells
    valid_counts = torch.zeros(num_subcells, dtype=torch.int).cuda()
    
    # Get non-zero elements from the master mask
    nzm = torch.nonzero(master_mask)
    if nzm.shape[0] == 0:
        return None, None, None, None, None
    
    # Count non-zero elements per subcell
    nbm = torch.bincount(nzm[:, 0])
    valid_counts[0 : nbm.shape[0]] = nbm
    nbs = torch.cumsum(nbm, dim=-1)
    
    # Initialize array to hold coordinates per subcell
    coordinates_per_subcell = torch.zeros(num_subcells, torch.max(nbm + 1), dtype=torch.int).cuda()
    
    # Assign valid coordinates to subcells
    rx = torch.nonzero(valid_counts)
    coordinates_per_subcell[rx, 0 : valid_counts[rx]] = nzm[nbs[rx] - valid_counts[rx] : nbs[rx], 1].int()
    total_counts[rx] = valid_counts[rx]
    
    # Mask for molecule types
    mol_mask = at[master_mask[rx]]
    hydrogen_counts[rx] = torch.sum(mol_mask == H).int()
    oxygen_counts[rx] = torch.sum(mol_mask == O).int()
    
    return total_counts, oxygen_counts, hydrogen_counts, coordinates_per_subcell, valid_counts


ATOMS = 1850000
SUBDIVISIONS = 1000

import time

# random xyz within bounds
data_points = (torch.rand(ATOMS, 3) * 0.5).tolist()
#'random letters'
atom_types_int = torch.randint(65, 90, (ATOMS,))
atom_alphabet = np.array([chr(i) for i in range(65, 90)])
atom_types = atom_alphabet[atom_types_int.numpy() - 65]

# (3,1,1,1), ((xyz),x,y,z)
subdivisions = [
    (torch.zeros(3), torch.rand(1), torch.rand(1), torch.rand(1))
    for _ in range(SUBDIVISIONS)
]
start = time.perf_counter()

atoms_per_cell(data_points, atom_types, subdivisions)
