import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt


def wrap_positions(pos, L):
    return pos - np.floor(pos / L) * L


def deposit_to_grid(pos, L, ngrid):
    nx, ny, nz = ngrid
    frac = pos / L
    frac -= np.floor(frac)

    ix = np.floor(frac[:, 0] * nx).astype(np.int32) % nx
    iy = np.floor(frac[:, 1] * ny).astype(np.int32) % ny
    iz = np.floor(frac[:, 2] * nz).astype(np.int32) % nz

    grid = np.zeros((nx, ny, nz), dtype=np.float64)

    np.add.at(grid, (ix, iy, iz), 1.0)

    return grid


def spherical_average_Sq(Sk, L, ngrid, nbins=None, drop_q0=True):
    nx, ny, nz = ngrid
    Lx, Ly, Lz = L

    qx = 2.0 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
    qy = 2.0 * np.pi * np.fft.fftfreq(ny, d=Ly / ny)
    qz = 2.0 * np.pi * np.fft.fftfreq(nz, d=Lz / nz)

    Qx, Qy, Qz = np.meshgrid(qx, qy, qz, indexing="ij")
    qmag = np.sqrt(Qx ** 2 + Qy ** 2 + Qz ** 2).ravel()

    Sk_flat = Sk.ravel()

    if nbins is None:
        nbins = min(nx, ny, nz) // 2

    qmax = qmag.max()
    edges = np.linspace(0.0, qmax, nbins + 1)

    centers = 0.5 * (edges[:-1] + edges[1:])

    which = np.digitize(qmag, edges) - 1

    Sq = np.zeros(nbins, dtype=np.float64)
    counts = np.zeros(nbins, dtype=np.int64)

    for i in range(Sk_flat.size):
        b = which[i]
        if 0 <= b < nbins:
            Sq[b] += Sk_flat[i]
            counts[b] += 1

    Sq /= np.maximum(counts, 1)

    if drop_q0:
        # remove first non-empty bin near q=0
        # safest: just set it to NaN so it won't be picked as q*
        Sq[0] = np.nan

    return centers, Sq


def structure_factor_composition(
        data_file,
        traj_file,
        type_A=1,
        type_B=2,
        ngrid=(64, 64, 64),
        start=0,
        stop=None,
        step=1,
        nbins_q=None,
        subtract_mean=True,
        drop_q0=True
):
    """
    Compute spherically averaged S(q) from composition field psi(r)=rho_A-rho_B.
    Returns:
      q_centers, Sq_avg, q_star, Sq_star
    """
    u = mda.Universe(data_file, traj_file)
    A = u.select_atoms(f"type {type_A}")
    B = u.select_atoms(f"type {type_B}")

    Sq_acc = None
    nframes = 0

    for ts in u.trajectory[start:stop:step]:
        # MDAnalysis dimensions: [Lx, Ly, Lz, alpha, beta, gamma]
        L = ts.dimensions[:3].astype(np.float64)
        V = float(np.prod(L))

        posA = wrap_positions(A.positions.astype(np.float64), L)
        posB = wrap_positions(B.positions.astype(np.float64), L)

        rhoA = deposit_to_grid(posA, L, ngrid)
        rhoB = deposit_to_grid(posB, L, ngrid)

        psi = rhoA - rhoB
        if subtract_mean:
            psi -= psi.mean()

        psi_k = np.fft.fftn(psi)
        Sk = (np.abs(psi_k) ** 2) / V  # matches paper form up to discretization constant

        Sk[0, 0, 0] = np.nan

        # spherical average
        q, Sq = spherical_average_Sq(Sk, L, ngrid, nbins=nbins_q, drop_q0=drop_q0)

        if Sq_acc is None:
            Sq_acc = np.zeros_like(Sq, dtype=np.float64)

        # accumulate ignoring NaNs
        valid = np.isfinite(Sq)
        Sq_acc[valid] += Sq[valid]

        nframes += 1

    Sq_avg = Sq_acc / max(nframes, 1)

    # Global peak and pick peak q* excluding NaNs
    peak_idx = np.nanargmax(Sq_avg)
    q_star = float(q[peak_idx])
    Sq_star = float(Sq_avg[peak_idx])

    # Finite q peak
    q_cut = 0.25
    mask = q > q_cut
    if np.any(mask):
        peak_idx_finite = np.nanargmax(Sq_avg[mask])
        q_star_finite = float(q[mask][peak_idx_finite])
        Sq_star_finite = float(Sq_avg[mask][peak_idx_finite])
    else:
        q_star_finite = np.nan
        Sq_star_finite = np.nan

    return q, Sq_avg, q_star, Sq_star, q_star_finite, Sq_star_finite, nframes


if __name__ == "__main__":
    # ---- EDIT THESE ----
    data = "PCL_CS_blend_with_dihedral_structure4.data"  # example
    traj = "kphi1=0_kphi2=1_cool_r06_epso_1.0_structure4.dcd"
    q, Sq_avg, q_star, Sq_star, q_star_finite, Sq_star_finite, nframes = structure_factor_composition(
        data_file=data,
        traj_file=traj,
        type_A=1,  # PCL
        type_B=2,  # CS
        ngrid=(32, 32, 32),
        start=190, stop=200, step=1,  # sample every 10 frames
        nbins_q=50,
        subtract_mean=True,
        drop_q0=True
    )

    plt.figure(figsize=(9, 7))
    plt.plot(q, Sq_avg, marker='o', markersize=3, linewidth=2)
    plt.axvline(q_star, color='red', linestyle='--', linewidth=1.5, label='Global peak')
    plt.axvline(q_star_finite, color='green', linestyle='--', linewidth=1.5, label='Finite-q peak')
    plt.legend()
    # plt.axvline(2 * (2 * np.pi / 39.0), linestyle='--')  # q_cut

    ax = plt.gca()
    ax.set_xlabel("q", fontname="Times New Roman", fontsize=23, fontweight="bold", labelpad=0)
    ax.set_ylabel("S(q)", fontname="Times New Roman", fontsize=23, fontweight="bold", labelpad=5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 80)
    plt.title("Composition Structure Factor")
    for label in ax.get_xticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontsize(23)

    for label in ax.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontsize(23)
    plt.show()

    print("Frames used:", nframes)

    print("\nGlobal peak:")
    print("q* =", q_star)
    print("S(q*) =", Sq_star)
    print("Domain size D ~ 2pi/q* =", 2.0 * np.pi / q_star)

    print("\nFinite-q peak (excluding q < 0.25):")
    print("q*_finite =", q_star_finite)
    print("S(q*_finite) =", Sq_star_finite)
    if np.isfinite(q_star_finite):
        print("Domain size D_finite ~ 2pi/q* =", 2.0 * np.pi / q_star_finite)
