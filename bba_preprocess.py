"""Preprocess the dataset."""

import time
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Union
import MDAnalysis
from MDAnalysis.analysis import contacts, distances


PathLike = Union[str, Path]


def traj_to_dset(
    topology: PathLike,
    traj_file: PathLike,
    save_file: Optional[PathLike] = None,
    selection: str = "protein and name CA",
    skip_every: int = 1,
    verbose: bool = False,
    print_every: int = 10,
):
    """Compute contact maps and ligand heav atom contact for each frame.

    Parameters
    ----------
    topology : PathLike
        Path to topology file: CHARMM/XPLOR PSF topology file,
        PDB file or Gromacs GRO file.
    traj_file : PathLike
        Trajectory file (in CHARMM/NAMD/LAMMPS DCD, Gromacs XTC/TRR,
        or generic. Stores coordinate information for the trajectory.
    save_file : Optional[PathLike]
        Path to output h5 dataset file name.
    selection : str
        Selection set of atoms in the protein.
    skip_every : int
        Only colelct data every `skip_every` frames.
    verbose: bool
        If true, prints verbose output.
    print_every: int
        Prints update every `print_every` frame.

    Returns
    -------
    npt.ArrayLike
        Distance matrix for each frame.
    """

    # start timer
    start_time = time.time()

    # Load simulation
    sim = MDAnalysis.Universe(str(topology), str(traj_file))

    if verbose:
        print("Traj length: ", len(sim.trajectory))

    # Atom selection for reference
    atoms = sim.select_atoms(selection)

    distance_matrices = []

    for i, _ in tqdm(enumerate(sim.trajectory[::skip_every])):

        # Point cloud positions of selected atoms in frame i
        positions = atoms.positions

        # Compute distance matrix
        dist = contacts.distance_array(
            atoms.positions, atoms.positions
        )
        distance_matrices.append(dist)
        
        if verbose:
            if i % print_every == 0:
                msg = f"Frame {i}/{len(sim.trajectory)}"
                msg += f"\tshape: {dist.shape}"
                print(msg)

                
    distance_matrices = np.array(distance_matrices)
    
    if save_file:
        np.save(save_file, distance_matrices)

    if verbose:
        print(f"Duration {time.time() - start_time}s")

    return distance_matrices


if __name__ == "__main__":
    pdb_file = "/homes/abrace/data/bba/1FME-unfolded.pdb"
    traj_files = sorted(Path("/homes/heng.ma/Research/FoldingTraj/1FME-0/1FME-0-protein/").glob("*dcd"))

    save_dir = Path("/homes/abrace/src/fourier_neural_operator/data/preprocessed/bba/")

    for traj_file in traj_files:

        save_file = save_dir / traj_file.with_suffix(".npy").name
        print(save_file)
        print(traj_file)

        traj_to_dset(
            topology=pdb_file,
            traj_file=traj_file,
            save_file=save_file,
            selection="protein and name CA",
            verbose=False,
        )
