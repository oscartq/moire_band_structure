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
        self.Nbands = self.Nzf * 4  # 2 layers * 2 sublattices

    def compute_H(self, k):
        H = np.zeros((self.Nbands, self.Nbands), dtype=complex)

        for n1 in range(self.Nzf):
            n11, n12 = self.map_zf_idx[n1]
            # Calculate k-vectors for top and bottom layers
            kmx_t, kmy_t = k + n11 * self.g1 + n12 * self.g2 - self.Ktop
            kmx_b, kmy_b = k + n11 * self.g1 + n12 * self.g2 - self.Kbot

            # Kinetic energy
            H[n1*4, n1*4] = (self.hbar**2) * (kmx_t**2 + kmy_t**2) / (2 * self.me)
            H[n1*4+1, n1*4+1] = (self.hbar**2) * (kmx_t**2 + kmy_t**2) / (2 * self.me)
            H[n1*4+2, n1*4+2] = (self.hbar**2) * (kmx_b**2 + kmy_b**2) / (2 * self.me)
            H[n1*4+3, n1*4+3] = (self.hbar**2) * (kmx_b**2 + kmy_b**2) / (2 * self.me)

            # Tunneling (q=0)
            H[n1*4, n1*4+2] = self.w
            H[n1*4+2, n1*4] = self.w
            H[n1*4+1, n1*4+3] = self.w
            H[n1*4+3, n1*4+1] = self.w

            for n2 in range(self.Nzf):
                n21, n22 = self.map_zf_idx[n2]
                if (n21 == n11 + 1) and (n22 == n12):
                    # g1
                    H[n1*4, n2*4] = -self.V0 * np.exp(1j * self.phi)
                    H[n1*4+1, n2*4+1] = -self.V0 * np.exp(1j * self.phi)
                    H[n1*4+2, n2*4+2] = -self.V0 * np.exp(-1j * self.phi)
                    H[n1*4+3, n2*4+3] = -self.V0 * np.exp(-1j * self.phi)
                    # g4
                    H[n2*4, n1*4] = -self.V0 * np.exp(-1j * self.phi)
                    H[n2*4+1, n1*4+1] = -self.V0 * np.exp(-1j * self.phi)
                    H[n2*4+2, n1*4+2] = -self.V0 * np.exp(1j * self.phi)
                    H[n2*4+3, n1*4+3] = -self.V0 * np.exp(1j * self.phi)
                elif (n21 == n11) and (n22 == n12 + 1):
                    # g2
                    H[n1*4, n2*4] = -self.V0 * np.exp(-1j * self.phi)
                    H[n1*4+1, n2*4+1] = -self.V0 * np.exp(-1j * self.phi)
                    H[n1*4+2, n2*4+2] = -self.V0 * np.exp(1j * self.phi)
                    H[n1*4+3, n2*4+3] = -self.V0 * np.exp(1j * self.phi)
                    H[n1*4, n2*4+2] = self.w
                    H[n1*4+1, n2*4+3] = self.w
                    # g5
                    H[n2*4, n1*4] = -self.V0 * np.exp(1j * self.phi)
                    H[n2*4+1, n1*4+1] = -self.V0 * np.exp(1j * self.phi)
                    H[n2*4+2, n1*4+2] = -self.V0 * np.exp(-1j * self.phi)
                    H[n2*4+3, n1*4+3] = -self.V0 * np.exp(-1j * self.phi)
                    H[n2*4+2, n1*4] = self.w
                    H[n2*4+3, n1*4+1] = self.w
                elif (n21 == n11 - 1) and (n22 == n12 + 1):
                    # g3
                    H[n1*4, n2*4] = -self.V0 * np.exp(1j * self.phi)
                    H[n1*4+1, n2*4+1] = -self.V0 * np.exp(1j * self.phi)
                    H[n1*4+2, n2*4+2] = -self.V0 * np.exp(-1j * self.phi)
                    H[n1*4+3, n2*4+3] = -self.V0 * np.exp(-1j * self.phi)
                    H[n1*4, n2*4+2] = self.w
                    H[n1*4+1, n2*4+3] = self.w
                    # g6
                    H[n2*4, n1*4] = -self.V0 * np.exp(-1j * self.phi)
                    H[n2*4+1, n1*4+1] = -self.V0 * np.exp(-1j * self.phi)
                    H[n2*4+2, n1*4+2] = -self.V0 * np.exp(1j * self.phi)
                    H[n2*4+3, n1*4+3] = -self.V0 * np.exp(1j * self.phi)
                    H[n2*4+2, n1*4] = self.w
                    H[n2*4+3, n1*4+1] = self.w

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
        Nkx = 21 # Number of points in x direction
        Nky = 21

        dkx = 1/(Nkx)
        dky = 1/(Nky)

        WF = np.zeros((self.Nbands, Nkx, Nky), dtype=complex)
        for nkx in range(Nkx):
            tx = nkx / (Nkx - 1)
            for nky in range(Nky):
                ty = nky / (Nky - 1)
                k = tx * self.g1 + ty * self.g2  # 2D vector
                
                _, WF_hlp = self.solve_H(k)
                WF[:,nkx,nky] = WF_hlp[:,band]

        Berry_curvature = np.zeros((Nkx - 1, Nky - 1))
        for nkx in range(Nkx - 1):
            for nky in range(Nky - 1):
                Ux  = np.vdot(WF[:, nkx, nky], WF[:, nkx + 1, nky])     
                Uy  = np.vdot(WF[:, nkx, nky], WF[:, nkx, nky + 1])
                Uxy = np.vdot(WF[:, nkx + 1, nky], WF[:, nkx + 1, nky + 1])
                Uyx = np.vdot(WF[:, nkx, nky + 1], WF[:, nkx + 1, nky + 1])
                
                # Ensure phase ∈ [-π, π]
                phase = np.angle(Ux * Uxy * np.conj(Uyx * Uy))
                phase = (phase + np.pi) % (2.0 * np.pi) - np.pi

                Berry_curvature[nkx, nky] = phase #* (dkx * dky) #if you want normalization

        chernnum = np.sum(Berry_curvature) / (2 * np.pi) #* dkx * dky 

        print(f"Chern number: {chernnum}")
        
        return chernnum
    
def plot_single_bandstructure(degree_theta):
    """Plot band structure for a single twist angle"""
    tmd = TransitionMetalDichalcogenides(degree_theta)
    Ek, WF, bz_points, Nk = tmd.calculate_bandstructure()
    
    # Plot first 15 bands with negative energies in meV
    plt.figure(figsize=(4, 8))
    for n in range(15):  # First 15 bands
        plt.plot(range(1, Nk + 1), -Ek[n, :] * 1e3)  # Convert to meV and negate
        
        c = tmd.compute_Chern_number(n) #tmd.compute_Chern_number(Ek[n], WF[:, n])

    # BZ point labels
    labels = ["$\gamma$", "$k_+$", "m", "$\gamma$"]
    
    # Vertical dotted lines at BZ points (except start and end)
    for pos in bz_points:
        plt.axvline(x=pos, color='k', linestyle='--', linewidth=0.8)
    
    # Label BZ points
    plt.xticks(bz_points, labels)
    plt.xlabel("")
    
    # Set limits
    plt.xlim(0, Nk)
    plt.ylim(0, 40)
    
    plt.ylabel("Energy (meV)")
    plt.title(rf"$\theta$={degree_theta}$^\circ$")
    plt.tight_layout()
    plt.show()

def plot_multiple_bandstructures(angles):
    """Plot band structures for multiple twist angles side by side"""
    n_plots = len(angles)
    fig, axes = plt.subplots(1, n_plots, figsize=(3*n_plots, 6))
   
    # Handle single plot case
    if n_plots == 1:
        axes = [axes]
   
    for idx, degree_theta in enumerate(angles):
        ax = axes[idx]
       
        # Calculate band structure for this angle
        tmd = TransitionMetalDichalcogenides(degree_theta)
        Ek, bz_points, Nk = tmd.calculate_bandstructure()
       
        # Plot first 25 bands with negative energies in meV (matching your single plot function)
        for n in range(25):  # First 25 bands
            ax.plot(range(1, Nk + 1), -Ek[n, :] * 1e3)  # Convert to meV and negate
       
        # BZ point labels
        labels = ["$\gamma$", "$k_+$", "m", "$\gamma$"]
       
        # Vertical dotted lines at BZ points (except start and end)
        for pos in bz_points[:-1]:
            ax.axvline(x=pos, color='k', linestyle='--', linewidth=0.8)
       
        # Label BZ points
        ax.set_xticks(bz_points)
        ax.set_xticklabels(labels)
       
        # Set limits (matching your single plot function)
        ax.set_xlim(0, Nk)
        ax.set_ylim(0, 40)
       
        # Labels and title
        if idx == 0:  # Only add y-label to leftmost plot
            ax.set_ylabel("Energy (meV)")
        ax.set_title(rf"$\theta$={degree_theta}$^\circ$")
   
    plt.tight_layout()
    plt.show()

plot_single_bandstructure(1.0)

# tmd = TransitionMetalDichalcogenides(1.0)
# Ek, WF, bz_points, Nk = tmd.calculate_bandstructure()