# Author: Oscar Törnquist
import matplotlib.pyplot as plt
import numpy as np
import itertools

class TransitionMetalDichalcogenides:
    def __init__(self, degree_theta):
        """Initialize TMD model with given twist angle"""
        self.degree_theta = degree_theta

        # Parameters
        self.hbar = 0.6582  # (eV*fs)
        self.m0 = 5.68568  # (eV fs² nm⁻²)

        self.me = 0.43 * self.m0
        self.phi = 128 * (np.pi / 180)
        self.V0 = 0.009
        self.w = 0.018
        self.a = 0.3317
        self.theta = self.degree_theta * (np.pi / 180)  # Twist angle (degrees)

        # Moire lattice constant
        self.aM = self.a / (2.0 * np.sin(0.5 * self.theta))

        # g-vectors of mBZ
        self.g0 = (4 * np.pi) / (np.sqrt(3) * self.aM)
        self.q0 = self.g0 / np.sqrt(3)
        self.gvecs = self.g0 * np.array([[np.cos(n * np.pi / 3), np.sin(n * np.pi / 3)] for n in range(6)])
        self.Ktop = (self.gvecs[0] + self.gvecs[5]) / 3
        self.Kbot = (self.gvecs[0] + self.gvecs[1]) / 3
        self.g1 = self.gvecs[0]
        self.g2 = self.gvecs[1]

        # Moiré shells, generating the lattice
        self.Nshells = 4
        self.Nzf = (2 * self.Nshells + 1) ** 2  # Total number of "zone-folded" moiré unit cells
        # Map band index to the generators n1, n2
        self.map_zf_idx = list(itertools.product(range(-self.Nshells, self.Nshells + 1),
                                                range(-self.Nshells, self.Nshells + 1)))
        self.Nbands = self.Nzf * 2 # 2 layers * 2 sublattices

    def compute_H(self, k):
        H = np.zeros((self.Nbands, self.Nbands), dtype=complex)

        for n1 in range(self.Nzf):
            n11, n12 = self.map_zf_idx[n1]
            i1_t = 2 * n1
            i1_b = i1_t + 1

            km_t = np.linalg.norm(k + n11 * self.g1 + n12 * self.g2 - self.Ktop)
            km_b = np.linalg.norm(k + n11 * self.g1 + n12 * self.g2 - self.Kbot)

            # Kinetic energy
            H[i1_t, i1_t] = self.hbar**2 * km_t**2 / (2 * self.me)
            H[i1_b, i1_b] = self.hbar**2 * km_b**2 / (2 * self.me)

            # Tunneling (q=0)
            H[i1_t, i1_b] = self.w
            H[i1_b, i1_t] = self.w

            for n2 in range(self.Nzf):
                n21, n22 = self.map_zf_idx[n2]
                i2_t = 2 * n2
                i2_b = i2_t + 1

                if (n21 == n11 + 1) and (n22 == n12):
                    # g1
                    H[i1_t, i2_t] = -self.V0 * np.exp(1j * self.phi)
                    H[i1_b, i2_b] = -self.V0 * np.exp(-1j * self.phi)
                    # g4 (Hermitian counterpart)
                    H[i2_t, i1_t] = -self.V0 * np.exp(-1j * self.phi)
                    H[i2_b, i1_b] = -self.V0 * np.exp(1j * self.phi)

                elif (n21 == n11) and (n22 == n12 + 1):
                    # g2
                    H[i1_t, i2_t] = -self.V0 * np.exp(-1j * self.phi)
                    H[i1_b, i2_b] = -self.V0 * np.exp(1j * self.phi)
                    H[i1_t, i2_b] = self.w  # momentum-shifted tunneling
                    # g5
                    H[i2_t, i1_t] = -self.V0 * np.exp(1j * self.phi)
                    H[i2_b, i1_b] = -self.V0 * np.exp(-1j * self.phi)
                    H[i2_b, i1_t] = self.w

                elif (n21 == n11 - 1) and (n22 == n12 + 1):
                    # g3
                    H[i1_t, i2_t] = -self.V0 * np.exp(1j * self.phi)
                    H[i1_b, i2_b] = -self.V0 * np.exp(-1j * self.phi)
                    H[i1_t, i2_b] = self.w
                    # g6
                    H[i2_t, i1_t] = -self.V0 * np.exp(-1j * self.phi)
                    H[i2_b, i1_b] = -self.V0 * np.exp(1j * self.phi)
                    H[i2_b, i1_t] = self.w

        return H


    def solve_H(self, k):
        """Solve eigenvalue problem for Hamiltonian at k-point"""
        H = self.compute_H(k)
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        Ek = np.real(eigenvalues)
        WF = eigenvectors
        return Ek, WF

    def calculate_bandstructure(self):
        """Calculate band structure along high-symmetry path"""
        # Brillouin zone path Γ -> K -> M -> Γ
        # number of points on each segment
        Nk1 = 100
        Nk2 = 50
        Nk3 = 100
        Nk = Nk1 + Nk2 + Nk3
        
        # Initialize kvec as list of 2D points
        kvec = []
        
        # Γ -> K (n from 0 to Nk1-1)
        Gamma = np.array([0, 0])
        for n in range(Nk1):
            kvec.append(Gamma + (self.Ktop - Gamma) * (n / Nk1))

        # K -> M (n from 0 to Nk2-1)
        Mpoint = 0.5 * (self.Ktop + self.Kbot)
        for n in range(Nk2):
            kvec.append(self.Ktop + (Mpoint - self.Ktop) * (n / Nk2))
        
        # M -> Γ (n from 0 to Nk3-1)
        for n in range(Nk3):
            kvec.append(Mpoint + (Gamma - Mpoint) * (n / Nk3))
                
        kvec = np.array(kvec)  # Now kvec has shape (Nk, 2)

        # Solve for band energies
        Ek = np.zeros((self.Nbands, Nk))
        for nk in range(Nk):
            k = kvec[nk, :]  # Extract 2D k-point
            Ek[:, nk], WF = self.solve_H(k)
            
        # Return band structure data
        bz_points = [0, Nk1, Nk1 + Nk2, Nk]

        return Ek, WF, bz_points, Nk
    
    def compute_Chern_number(self, band):
        # Define k-grid in mBZ (map hexagon into square)
        Nkx = 21 # Number of points in x direction
        Nky = 21

        WF = np.zeros((self.Nbands, Nkx, Nky), dtype=complex)
        for nkx in range(Nkx):
            for nky in range(Nkx):
                k = ((nkx-1)/(Nkx-1))*self.g1 + ((nky-1)/(Nky-1))*self.g2
                _, WF_hlp = self.solve_H(k)
                WF[:,nkx,nky] = WF_hlp[:,band]


        Berry_curvature = np.zeros((Nkx - 1, Nky - 1))
        for nkx in range(Nkx - 1):
            for nky in range(Nky - 1):
                Ux  = np.vdot(WF[:, nkx, nky], WF[:, nkx + 1, nky]) # <WF(k+dx)|WF(k)>
                Uy  = np.vdot(WF[:, nkx, nky], WF[:, nkx, nky + 1]) # <WF(k)|WF(k+dy)>
                Uxy = np.vdot(WF[:, nkx + 1, nky], WF[:, nkx + 1, nky + 1]) # < WF(k+dx)|WF(k+dy+dx)>
                Uyx = np.vdot(WF[:, nkx, nky + 1], WF[:, nkx + 1, nky + 1]) # <WF(k+dy)|WF(k+dy+dx)>
    
                # Ensure phase ∈ [-π, π]
                phase = np.real((1/1.0j)*np.log(Ux * Uxy / (Uyx * Uy)))
                phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi

                Berry_curvature[nkx, nky] = phase 

        chernnum = np.sum(Berry_curvature) / (2 * np.pi)

        return chernnum
    
def plot_single_bandstructure(degree_theta, chern_bands):
    """
    Plot band structure for a single twist angle and print Chern numbers
    for the specified band indices.

    Parameters
    ----------
    degree_theta : float
        Twist angle in degrees.
    chern_bands : Iterable[int]
        Band indices for which to compute/print Chern numbers.
    """
    tmd = TransitionMetalDichalcogenides(degree_theta)
    Ek, WF, bz_points, Nk = tmd.calculate_bandstructure()
    
    # Plot first 10 bands with negative energies in meV
    plt.figure(figsize=(4, 8))
    for n in range(10):
        plt.plot(range(1, Nk + 1), -Ek[n, :] * 1e3)  # meV and negative
    
    # --- Replace previous annotation block with this: ---
    for idx, band in enumerate(chern_bands):
        c_rounded = int(np.rint(tmd.compute_Chern_number(band)))
        x_curve = np.arange(1, Nk + 1)
        y_curve = -Ek[band, :] * 1e3

        # Alternate placement to reduce clutter: even → above max, odd → below min
        if idx % 2 == 0:
            j = int(np.argmax(y_curve))   # highest point → label above
            dy_pts, va = 8, "bottom"
        else:
            j = int(np.argmin(y_curve))   # lowest point → label below
            dy_pts, va = -8, "top"

        plt.annotate(
            f"{c_rounded}",
            (x_curve[j], y_curve[j]),            # anchor exactly on the band
            xytext=(15, dy_pts),                  # shift up/down in points
            textcoords="offset points",
            ha="center", va=va,
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
            arrowprops=dict(arrowstyle="-", lw=0.8, color="black")  # leader line to band
        )

    # BZ point labels
    labels = [r"$\gamma$", r"$k_+$", "m", r"$\gamma$"]
    
    # Vertical dotted lines at BZ points (except start and end)
    for pos in bz_points:
        plt.axvline(x=pos, linestyle='--', linewidth=0.8, color='k')
    
    # Axes & limits
    plt.xticks(bz_points, labels)
    plt.xlabel("")
    plt.xlim(0, Nk)
    plt.ylim(0, 40)
    plt.ylabel("Energy (meV)")
    plt.title(rf"$\theta$={degree_theta}$^\circ$")
    plt.tight_layout()
    plt.show()

def plot_multiple_bandstructures(angles, chern_bands=None):
    """
    Plot band structures for multiple twist angles side by side
    and show/print Chern numbers for the specified bands.

    Parameters
    ----------
    angles : Iterable[float]
        Twist angles (degrees).
    chern_bands : Iterable[int] or None
        Band indices for which to compute Chern numbers. If None, no Chern numbers are computed.
    """
    if chern_bands is None:
        chern_bands = []

    n_plots = len(angles)
    fig, axes = plt.subplots(1, n_plots, figsize=(3 * n_plots, 6))

    # Handle single plot case
    if n_plots == 1:
        axes = [axes]

    chern_map = {}  # {angle: {band: (exact, rounded)}}

    for idx, degree_theta in enumerate(angles):
        ax = axes[idx]

        # Calculate band structure for this angle
        tmd = TransitionMetalDichalcogenides(degree_theta)
        Ek, _, bz_points, Nk = tmd.calculate_bandstructure()

        # Plot first 10 bands with negative energies in meV
        xvals = np.arange(1, Nk + 1)
        for n in range(10):
            ax.plot(xvals, -Ek[n, :] * 1e3, linewidth=0.8)

        # Compute & annotate Chern numbers for requested bands
        chern_map[degree_theta] = {}
        for j, band in enumerate(chern_bands):
            if band < 0 or band >= Ek.shape[0]:
                continue  # skip out-of-range requests safely

            c_exact = tmd.compute_Chern_number(band)
            c_rounded = int(np.rint(c_exact))
            chern_map[degree_theta][band] = (c_exact, c_rounded)

            # Choose an anchor point on the band (extremum), then offset inward
            y_curve = -Ek[band, :] * 1e3
            if j % 2 == 0:
                k_idx = int(np.argmax(y_curve))  # topmost point
                dy_pts, va = -12, "top"          # pull text inward (down)
            else:
                k_idx = int(np.argmin(y_curve))  # bottommost point
                dy_pts, va = 12, "bottom"        # pull text inward (up)

            # Horizontal nudge into the plot (points). Negative moves left, positive right.
            # Use side-dependent nudge so labels tend to sit away from the edges.
            if k_idx < Nk // 2:
                dx_pts = 15
            else:
                dx_pts = -15

            ax.annotate(
                f"{c_rounded}",
                (xvals[k_idx], y_curve[k_idx]),
                xytext=(dx_pts, dy_pts),
                textcoords="offset points",
                ha="center", va=va,
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
                arrowprops=dict(arrowstyle="-", lw=0.8, color="black")
            )

            # Also print to terminal as requested
            print(f"Theta {degree_theta}°, Band {band}: Chern ≈ {c_exact:.6f} → {c_rounded}")

        # BZ point labels
        labels = [r"$\gamma$", r"$k_+$", "m", r"$\gamma$"]

        # Vertical dotted lines at BZ points (except start and end)
        for pos in bz_points[:-1]:
            ax.axvline(x=pos, color='k', linestyle='--', linewidth=0.8)

        # Label BZ points
        ax.set_xticks(bz_points)
        ax.set_xticklabels(labels)

        # Set limits
        ax.set_xlim(0, Nk)
        ax.set_ylim(0, 40)

        # Labels and title
        if idx == 0:
            ax.set_ylabel("Energy (meV)")
        ax.set_title(rf"$\theta$={degree_theta}$^\circ$")

    plt.tight_layout()
    plt.show()

    return chern_map

# Uncomment the ones that you want to use. First the degree of twist angle and then the bands which you want the chern number of.
#plot_single_bandstructure(1.0,[0, 1])

#plot_multiple_bandstructures([1.0, 1.43, 1.67, 2.5], chern_bands=[0, 1])