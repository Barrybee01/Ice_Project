import numpy as np

#Following the visualization method used in https://doi.org/10.1021/acs.jpclett.7b00492

PLANE_RANGES = [
    (1.2, 2.4),      # Plane 1
    (4.9, 6.1),      # Plane 2
    (8.6, 9.8),      # Plane 3
    (12.3, 13.5),    # Plane 4
    (16.0, 17.2),    # Plane 5
    (19.6, 20.9),    # Plane 6
    (23.3, 24.5),    # Plane 7
    (27.0, 28.2),    # Plane 8
    (30.7, 31.9),    # Plane 9
    (34.4, 35.6),    # Plane 10
    (38.0, 39.3),    # Plane 11
    (41.7, 42.9),    # Plane 12
    (45.4, 46.6),    # Plane 13
    (49.1, 50.3),    # Plane 14
    (52.8, 54.0),    # Plane 15
    (56.4, 57.7)     # Plane 16
]

plane_particle_ids = {}

initialized = False

def modify(frame, data):

    global initialized
    global plane_particle_ids

    particle_ids = np.asarray(data.particles["Particle Identifier"])
    positions = np.asarray(data.particles.positions)

    z = positions[:, 2]

    if not initialized: #Need to go to first frame to initiate the planes for ideal xtal

        if frame != 0:
            raise RuntimeError("Go to frame 0 and reevaluate the pipeline first.")

        print("")
        print("Initializing z-plane particle IDs from frame 0...")
        print("")

        plane_particle_ids = {}

        for plane_number, (zmin, zmax) in enumerate(PLANE_RANGES,start=1):
            mask = (z >= zmin) & (z <= zmax) #uses atom positions from INITIAL state
            ids = particle_ids[mask]
            plane_particle_ids[plane_number] = {int(pid) for pid in ids}

            print(
                f"Plane {plane_number:2d}: "
                f"{zmin:6.2f} <= z <= {zmax:6.2f}  "
                f"Particles = {len(ids)}")

        initialized = True

        print("")
        print("Plane particle IDs stored.")
        print("")


    plane_ids = np.zeros(len(particle_ids),dtype=np.int32)
    for plane_number, stored_ids in plane_particle_ids.items():
        mask = np.array([int(pid) in stored_ids for pid in particle_ids])
        plane_ids[mask] = plane_number

    data.particles_.create_property("PlaneID",data=plane_ids) #Creates a Plane ID property 
