# Author: Oscar Törnquist
import matplotlib.pyplot as plt
import numpy as np
import itertools

class TwistedBilayerGraphene:
    def __init__(self, degree_theta):
        """Initialize TBG model with given twist angle"""
        self.degree_theta = degree_theta
        
        # Define constants
        self.a = 0.247           # Graphene lattice constant (nm)
        self.K0 = (4*np.pi)/(3*self.a) # K point of monolayer BZ  
        self.vF = 19.81/(2*self.K0)    # From PRL 122, 106405 (2019)
        self.wAA = 0              # Energy of AA stacking
        self.wAB = 0.11           # Tunneling strength (eV)
        self.theta = (degree_theta * np.pi) / 180 # Convert to radians
        self.ktheta = 2*self.K0*np.sin(self.theta/2)
        self.alpha = self.wAB/(self.vF*self.ktheta)
        self.w = np.exp(1.0j*2*np.pi/3)
        
        # Sublattice (symmetry-breaking) potential
        self.m = 0.1
        
        # Moiré lattice vectors
        self.g0 = self.ktheta*np.sqrt(3)
        self.g1 = self.g0 * np.array([0.5, 0.5*np.sqrt(3)])
        self.g2 = self.g0 * np.array([-0.5, 0.5*np.sqrt(3)])
        
        # Momentum coordinates of top and bottom layer K points
        self.Ktop = np.array([0.5*np.sqrt(3), 0.5]) * self.ktheta
        self.Kbot = np.array([0.5*np.sqrt(3), -0.5]) * self.ktheta
        
        # Moiré shells, generating the lattice
        self.Nshells = 4
        self.Nzf = (2*self.Nshells + 1)**2  # Total number of "zone-folded" moiré unit cells
        # Map band index to the generators n1, n2
        self.map_zf_idx = list(itertools.product(range(-self.Nshells, self.Nshells+1), 
                                                range(-self.Nshells, self.Nshells+1)))
        self.Nbands = self.Nzf * 4  # 2 layers * 2 sublattices

    def compute_H(self, k):
        """Compute Hamiltonian matrix for given k-point"""
        k = k.flatten()
        H = np.zeros((self.Nbands, self.Nbands), dtype=complex)
        
        for n1 in range(self.Nzf):
            n11, n12 = self.map_zf_idx[n1]
            
            # Calculate k-vectors for top and bottom layers
            kmx_t, kmy_t = k + n11*self.g1 + n12*self.g2 - self.Ktop
            kmx_b, kmy_b = k + n11*self.g1 + n12*self.g2 - self.Kbot
            
            # Intralayer sub-lattice coupling
            H[(n1)*4+2, (n1)*4+0] = (kmx_t + 1j*kmy_t) / self.ktheta
            H[(n1)*4+3, (n1)*4+1] = (kmx_b + 1j*kmy_b) / self.ktheta
            H[(n1)*4+0, (n1)*4+2] = (kmx_t - 1j*kmy_t) / self.ktheta
            H[(n1)*4+1, (n1)*4+3] = (kmx_b - 1j*kmy_b) / self.ktheta
            
            # Sublattice symmetry breaking potential
            H[(n1)*4+0, (n1)*4+0] = self.m
            H[(n1)*4+1, (n1)*4+1] = self.m
            H[(n1)*4+2, (n1)*4+2] = -self.m
            H[(n1)*4+3, (n1)*4+3] = -self.m
            
            # Interlayer tunneling
            H[(n1)*4+2, (n1)*4+1] = self.alpha
            H[(n1)*4+3, (n1)*4+0] = self.alpha
            H[(n1)*4+1, (n1)*4+2] = self.alpha
            H[(n1)*4+0, (n1)*4+3] = self.alpha
            
            # Interlayer coupling between different unit cells
            for n2 in range(self.Nzf):
                n21, n22 = self.map_zf_idx[n2]
                
                if (n21 == n11-1) and (n22 == n12):
                    H[(n1)*4+2, (n2)*4+1] = self.alpha * self.w
                    H[(n2)*4+3, (n1)*4+0] = self.alpha * self.w
                    H[(n2)*4+1, (n1)*4+2] = self.alpha * np.conj(self.w)
                    H[(n1)*4+0, (n2)*4+3] = self.alpha * np.conj(self.w)
                    
                elif (n21 == n11) and (n22 == n12-1):
                    H[(n1)*4+2, (n2)*4+1] = self.alpha * np.conj(self.w)
                    H[(n2)*4+3, (n1)*4+0] = self.alpha * np.conj(self.w)
                    H[(n2)*4+1, (n1)*4+2] = self.alpha * self.w
                    H[(n1)*4+0, (n2)*4+3] = self.alpha * self.w
        
        return H

    def solve_H(self, k):
        """Solve eigenvalue problem for Hamiltonian at k-point"""
        H = self.compute_H(k)
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        Ek = np.real(eigenvalues)
        WF = eigenvectors
        return Ek, WF
    
    def calculate_bandstructure(self):
        """Calculate band structure along high-symmetry path: K' -> K -> Γ -> K'"""
        # Segment lengths
        Nk1 = 100
        Nk2 = 100
        Nk3 = 50
        Nk = Nk1 + Nk2 + Nk3        
        
        # Initialize kvec as list of 2D points
        kvec = []
        
        # Γ -> K (n from 0 to Nk1-1)
        Kprime_1 = np.array([0.0, -2*self.ktheta])
        Gamma_1 = np.array([0.0, 0.0])
        for n in range(Nk1):
            kvec.append(Kprime_1 + (Gamma_1 - Kprime_1) * (n / Nk1))

        # K -> M (n from 0 to Nk2-1)
        Gamma_2 = np.array([self.g0, 0.0])#np.array([self.ktheta, -(3/2)*self.ktheta])#
        for n in range(Nk2):
            kvec.append(Gamma_1  + (Gamma_2 - Gamma_1 ) * (n / Nk2))
        
        # M -> Γ (n from 0 to Nk3-1)
        Kprime_2 = np.array([self.g0, self.ktheta])
        for n in range(Nk3):
            kvec.append(Gamma_2 + (Kprime_2 - Gamma_2) * (n / Nk3))
                
        kvec = np.array(kvec)  # Now kvec has shape (Nk, 2)

        # Solve for band energies
        Ek = np.zeros((self.Nbands, Nk))
        for nk in range(Nk):
            Ek[:, nk], _ = self.solve_H(kvec[nk])

        # Brillouin zone tick marks (start of each segment)
        bz_points = [0, Nk1, Nk1 + Nk2, Nk]

        return Ek, bz_points, Nk


def plot_single_bandstructure(degree_theta):
    """Plot band structure for a single twist angle"""
    tbg = TwistedBilayerGraphene(degree_theta)
    Ek, bz_points, Nk = tbg.calculate_bandstructure()
    
    # Select bands around Fermi level
    mid_band = tbg.Nbands // 2
    band_indices = range(mid_band - 5, mid_band + 5)
    
    plt.figure(figsize=(3, 6))
    for n in band_indices:
        plt.plot(range(1, Nk + 1), Ek[n, :])
    
    # BZ point labels
    labels = ["K'", r"$\Gamma$", "K'"]
    
    # Vertical dotted lines at BZ points (except start and end)
    for pos in bz_points[1:-1]:
        plt.axvline(x=pos, color='k', linestyle='--', linewidth=0.8)
    
    # Label BZ points
    #plt.xticks(bz_points, labels)
    plt.xlabel("")
    
    # Set limits
    plt.xlim(1, Nk)
    ymin, ymax = Ek[band_indices, :].min(), Ek[band_indices, :].max()
    plt.ylim(ymin, ymax)
    
    plt.ylabel("Energy (eV)")
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
        tbg = TwistedBilayerGraphene(degree_theta)
        Ek, bz_points, Nk = tbg.calculate_bandstructure()
        
        # Select bands around Fermi level
        mid_band = tbg.Nbands // 2
        band_indices = range(mid_band - 5, mid_band + 5)
        
        # Plot bands
        for n in band_indices:
            ax.plot(range(1, Nk + 1), Ek[n, :])
        
        # BZ point labels
        labels = ["K'", "K", r"$\Gamma$", "K'"]
        
        # Vertical dotted lines at BZ points (except start and end)
        for pos in bz_points[0:-1]:
            ax.axvline(x=pos, color='k', linestyle='--', linewidth=0.8)
        
        # Label BZ points
        ax.set_xticks(bz_points)
        ax.set_xticklabels(labels)
        
        # Set limits
        ax.set_xlim(1, Nk)
        ymin, ymax = Ek[band_indices, :].min(), Ek[band_indices, :].max()
        ax.set_ylim(ymin, ymax)
        
        # Labels and title
        if idx == 0:  # Only add y-label to leftmost plot
            ax.set_ylabel("Energy (eV)")
        ax.set_title(rf"$\theta$={degree_theta}$^\circ$")
    
    plt.tight_layout()
    plt.show()