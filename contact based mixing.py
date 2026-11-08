import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance


def _build_bond_exclusion_keys(u, all_atoms):
    """
    Build a vector of encoded keys for bonded (1-2) pairs in LOCAL index space of all_atoms.
    Key encoding: key = i*N + j with i<j, where N = len(all_atoms).
    Returns: excl_keys (1D np.int64 array), or None if no bonds.
    """
    if not hasattr(u, "bonds") or u.bonds is None or len(u.bonds) == 0:
        return None

    N = len(all_atoms)

    # global atom indices in this AtomGroup
    gidx = all_atoms.indices
    g2l = {g: i for i, g in enumerate(gidx)}  # global -> local

    pairs_local = []
    for bond in u.bonds:
        gi = bond.atoms[0].index
        gj = bond.atoms[1].index
        if gi in g2l and gj in g2l:
            i = g2l[gi]
            j = g2l[gj]
            if i == j:
                continue
            if i > j:
                i, j = j, i
            pairs_local.append((i, j))

    if not pairs_local:
        return None

    pairs_local = np.array(pairs_local, dtype=np.int64)
    excl_keys = pairs_local[:, 0] * N + pairs_local[:, 1]
    # sort once for faster isin
    excl_keys.sort()
    return excl_keys


def phi_mix_timeseries_fast(
    data_file,
    traj_file,
    type_A=1,
    type_B=2,
    r_cut=1.4,
    start=0,
    stop=None,
    step=1,
    exclude_bonded=True,
):
    """
    Compute phi_mix = N_AB / (N_AA + N_BB + N_AB) per frame using ONE neighbor search per frame.

    - Uses all beads' neighbor pairs within r_cut.
    - Classifies each pair as AA/BB/AB based on atom "type".
    - Optionally excludes bonded (1–2) pairs (recommended for polymers).

    Returns:
      phi_arr, phi_avg, counts_list, nframes
      where counts_list elements are (N_AA, N_BB, N_AB)
    """
    u = mda.Universe(data_file, traj_file)
    all_atoms = u.atoms
    N = len(all_atoms)

    # MDAnalysis often stores types as strings; convert once
    # This produces an integer array of length N
    atom_types = np.asarray(all_atoms.types)
    try:
        atom_types = atom_types.astype(np.int32)
    except ValueError:
        # if types are like "1", "2" strings, astype works; if not, map them
        atom_types = np.array([int(t) for t in atom_types], dtype=np.int32)

    isA = (atom_types == int(type_A))
    isB = (atom_types == int(type_B))

    # Precompute bonded exclusion keys (constant across frames)
    excl_keys = _build_bond_exclusion_keys(u, all_atoms) if exclude_bonded else None

    phi_list = []
    counts_list = []
    used = 0

    for ts in u.trajectory[start:stop:step]:
        box = ts.dimensions  # orthorhombic OK

        pos = all_atoms.positions

        # One neighbor search: all pairs within r_cut
        pairs = capped_distance(pos, pos, max_cutoff=r_cut, box=box, return_distances=False)

        # keep unique pairs i<j only (avoid double counting + self pairs)
        pairs = pairs[pairs[:, 0] < pairs[:, 1]]
        if pairs.size == 0:
            phi_list.append(np.nan)
            counts_list.append((0, 0, 0))
            used += 1
            continue

        # Exclude bonded pairs if requested
        if excl_keys is not None and excl_keys.size > 0:
            keys = pairs[:, 0].astype(np.int64) * N + pairs[:, 1].astype(np.int64)
            # remove keys that are in excl_keys
            mask_keep = ~np.isin(keys, excl_keys, assume_unique=False)
            pairs = pairs[mask_keep]
            if pairs.size == 0:
                phi_list.append(np.nan)
                counts_list.append((0, 0, 0))
                used += 1
                continue

        i = pairs[:, 0]
        j = pairs[:, 1]

        # Vectorized classification
        AA = isA[i] & isA[j]
        BB = isB[i] & isB[j]
        AB = (isA[i] & isB[j]) | (isB[i] & isA[j])

        N_AA = int(np.count_nonzero(AA))
        N_BB = int(np.count_nonzero(BB))
        N_AB = int(np.count_nonzero(AB))

        N_tot = N_AA + N_BB + N_AB
        phi = (N_AB / N_tot) if N_tot > 0 else np.nan

        phi_list.append(phi)
        counts_list.append((N_AA, N_BB, N_AB))
        used += 1

    phi_arr = np.array(phi_list, dtype=float)
    phi_avg = np.nanmean(phi_arr)
    return phi_arr, phi_avg, counts_list, used


if __name__ == "__main__":
    data = "PCL_CS_blend_with_dihedral_structure2.data"
    traj = "kphi1=0_kphi2=1_cool_r06_epso_1.0_structure2.dcd"

    phi_ts, phi_avg, counts, nframes = phi_mix_timeseries_fast(
        data_file=data,
        traj_file=traj,
        type_A=1,
        type_B=2,
        r_cut=1.4,
        start=190, stop=200, step=1,
        exclude_bonded=True
    )

    print("Frames used:", nframes)
    print("phi_mix average:", phi_avg)
    print("Last-frame counts (N_AA, N_BB, N_AB):", counts[-1])
