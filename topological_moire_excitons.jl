using NLopt
using Cubature
using LinearAlgebra
using Plots
using DelimitedFiles
plotly()

# Fundamental constants
m0 = 5.685642 # eV fs² nm⁻²
hbar = 0.6582119569 # eV fs
eps0 = 5.526349406e-2 # Vacuum permittivity (e0²/eV/nm)
e0 = 1.0 # Elementary charge

# Parameters (from Das Sarma's PRL)
me = 0.8*m0
mh = 0.43*m0
d = 0.67 # Interlayer distance (nm)
eps = 3.8 # Why?

Mx = mh + me
mred = mh*me/Mx
alpha_e = me/Mx

a0 = 0.3317 # 
K0 = 4pi/(3*a0) # K point of monolayer BZ

# Moiré potential
U_m0 = 0.009 # (eV)
phi_m = 128pi/180
t = 0.018 # Tunneling strength (eV)

# Number of moiré shells
Nshells = 4
Nzf = (2*Nshells+1)^2 # Total number of "zone-folded" moiré unit cells
map_zf_idx = collect.(Iterators.product(-Nshells:Nshells, -Nshells:Nshells)) # Map band index to the generators n1,n2 (moiré cell is determined by n1*g1+n2*g2)
Nbands = 2*Nzf # 2 layers

### Compute exciton wave function parameters and energy via variational minimization of hydrogenic ansatz
function exciton_states_variational(l::Int64) # l: hole layer index
    # Variational function to minimize
    function variational_energy(a,l) # a: Bohr radius
        kin = hbar^2/(2*mred*a^2)
        pot = - Cubature.hquadrature(r -> r*exp(-2*r/a)/sqrt(r^2+(d*l)^2), 0.0, 100*a, reltol=1e-9)[1] * 2pi * 2/(pi*a^2) * e0^2/(4pi*eps*eps0)
        return kin + pot
    end
    function varfun(x, grad)
        return variational_energy(x[1],l)
    end
    opt = Opt(:LN_NELDERMEAD, 1)
    opt.lower_bounds = 0.5
    opt.upper_bounds = 5.0
    opt.xtol_rel = 1e-12
    opt.ftol_rel = 1e-12
    opt.maxeval = 100
    opt.min_objective = varfun
    (minf,minx,ret) = optimize(opt, [1.0])
    numevals = opt.numevals # the number of function evaluations
    # println("got $minf at $minx after $numevals iterations (returned $ret)")
    return minf[1], minx[1] # Return binding energy and Bohr radius
end

### Initialize Brillouin zone vectors
function initialize_BZ(theta)
    ktheta = 2*K0*sin(0.5*theta)
    # kappa points of mBZ
    kappa1 = ktheta * [0.5*sqrt(3), -0.5]
    kappa2 = ktheta * [0.5*sqrt(3), 0.5]
    # g-vectors of mBZ
    g0 = ktheta*sqrt(3)
    g1 = g0 * [1, 0]
    g2 = g0 * [0.5, 0.5*sqrt(3)]
    aM = a0/(2.0*sin(0.5*theta))
    println(aM)
    return kappa1, kappa2, ktheta, g1, g2, g0
end

### Compute exciton matrix elements for moiré potential and tunneling 
function exciton_matrix_elements(a_l, BZparams)
    kappa1, kappa2, ktheta, g1, g2, g0 = BZparams
    FF11 = (1+(0.5*a_l[1]*alpha_e*g0)^2)^-1.5
    FF22 = (1+(0.5*a_l[2]*alpha_e*g0)^2)^-1.5
    xi = 4*a_l[1]*a_l[2]/(a_l[1]+a_l[2])^2
    FF12 = xi*(1+xi*(0.5*sqrt(a_l[1]*a_l[2])*alpha_e*ktheta)^2)^-1.5
    U_mx_1 = -U_m0*FF11
    U_mx_2 = -U_m0*FF22
    t_x = -t*FF12
    matrix_elements = Vector{Float64}(vcat(U_mx_1,U_mx_2,t_x))
    println("FF11: $FF11, FF22: $FF22, FF12: $FF12")
    return matrix_elements
end

function compute_H(k::Vector{Float64}, DE::Float64, matrix_elements::Vector{Float64}, BZparams)
    kappa1, kappa2, ktheta, g1, g2, g0 = BZparams
    U_x1, U_x2, t_x = matrix_elements
    H = zeros(ComplexF64, Nbands, Nbands)
    for n1=1:Nzf
        n11, n12 = map_zf_idx[n1]
        k1 = norm(k + n11*g1 + n12*g2 - kappa1)
        k2 = norm(k + n11*g1 + n12*g2 - kappa2)
        # Kinetic term
        H[(n1-1)*2+1,(n1-1)*2+1] = (hbar*k1)^2/(2.0*Mx)
        H[(n1-1)*2+2,(n1-1)*2+2] = DE + (hbar*k2)^2/(2.0*Mx)
        # Moiré potential
        for n2=1:Nzf
            n21, n22 = map_zf_idx[n2]
            if (n21==n11+1) && (n22==n12)
                H[(n1-1)*2+1,(n2-1)*2+1] = U_x1 * exp(1im*phi_m)
                H[(n1-1)*2+2,(n2-1)*2+2] = U_x2 * exp(-1im*phi_m)
                H[(n2-1)*2+1,(n1-1)*2+1] = U_x1 * exp(-1im*phi_m)
                H[(n2-1)*2+2,(n1-1)*2+2] = U_x2 * exp(1im*phi_m)
            elseif (n21==n11) && (n22==n12+1)
                H[(n1-1)*2+1,(n2-1)*2+1] = U_x1 * exp(-1im*phi_m)
                H[(n1-1)*2+2,(n2-1)*2+2] = U_x2 * exp(1im*phi_m)
                H[(n2-1)*2+1,(n1-1)*2+1] = U_x1 * exp(1im*phi_m)
                H[(n2-1)*2+2,(n1-1)*2+2] = U_x2 * exp(-1im*phi_m)
            elseif (n21==n11-1) && (n22==n12+1)
                H[(n1-1)*2+1,(n2-1)*2+1] = U_x1 * exp(1im*phi_m)
                H[(n1-1)*2+2,(n2-1)*2+2] = U_x2 * exp(-1im*phi_m)
                H[(n2-1)*2+1,(n1-1)*2+1] = U_x1 * exp(-1im*phi_m)
                H[(n2-1)*2+2,(n1-1)*2+2] = U_x2 * exp(1im*phi_m)
            end
        end
        # Tunneling
        H[(n1-1)*2+1,(n1-1)*2+2] = t_x
        H[(n1-1)*2+2,(n1-1)*2+1] = t_x
        for n2=1:Nzf
            n21, n22 = map_zf_idx[n2]
            if (n21==n11) && (n22==n12+1)
                H[(n1-1)*2+1,(n2-1)*2+2] = t_x
                H[(n2-1)*2+2,(n1-1)*2+1] = t_x
            elseif (n21==n11-1) && (n22==n12+1)
                H[(n1-1)*2+1,(n2-1)*2+2] = t_x
                H[(n2-1)*2+2,(n1-1)*2+1] = t_x
            end
        end
    end
    return H
end

function exciton_bands(DE, a_l, BZparams)
    kappa1, kappa2, ktheta, g1, g2, g0 = BZparams

    # Matrix elements
    matrix_elements = exciton_matrix_elements(a_l, BZparams)
    
    # Brillouin zone path G->K->K'->G
    Nk1 = 50
    Nk2 = 50
    Nk3 = 50
    Nk = Nk1 + Nk2 + Nk3
    # G->K
    Gamma = [0, 0]
    klist = [ Gamma+(kappa1-Gamma)*(n/Nk1) for n=0:Nk1-1]
    # K->K'
    append!(klist, [ kappa1+(kappa2-kappa1)*(n/Nk2) for n=0:Nk2-1])
    # K'->G
    append!(klist, [ kappa2+(Gamma-kappa2)*(n/(Nk3-1)) for n=0:Nk3-1])
    # display(scatter([k[1] for k in klist],[k[2] for k in klist]))

    
    Elist = zeros(Nbands,Nk)
    layer_weight = zeros(Nbands, Nk)
    for nk=1:Nk
        k = klist[nk]
        H = compute_H(k, DE, matrix_elements, BZparams)
        F = eigen(H)
        Elist[:,nk] = real(F.values)
        for nb=1:Nbands
            layer_weight[nb,nk] = sum(abs2.(F.vectors[2:2:end,nb]))
        end
    end

    display(plot(1:Nk, [Elist[n,:].*1e3 for n=1:4]))
    # writedlm("Results/bands_layerweights_DE$(DE*1e3).dat", hcat(1e3.*Elist[1:4,:]', layer_weight[1:4,:]'))

end

function compute_Chern_number(band, DE, a_l, BZparams)
    kappa1, kappa2, ktheta, g1, g2, g0 = BZparams
    matrix_elements = exciton_matrix_elements(a_l, BZparams)

    # Define k-grid in mBZ (map hexagon into square)
    Nkx = 11 # Number of points in x direction
    Nky = 11

    E = zeros(2,Nkx,Nky)
    WF = zeros(ComplexF64,Nbands,Nkx,Nky)
    for nkx=1:Nkx
        for nky=1:Nky
            # k = [ kxvec[nkx] kyvec[nky] ]
            k = (nkx-1)/(Nkx-1)*g1 + (nky-1)/(Nky-1)*g2
            H = compute_H(k, DE, matrix_elements, BZparams)
            F = eigen(H)
            E[:,nkx,nky] = real.([F.values[band], F.values[band+1]])
            WF[:,nkx,nky] = F.vectors[:,band]
        end
    end

    # Compute Berry curvature and Chern number following based on https://doi.org/10.1143/JPSJ.74.1674 and PRB 99, 075127 (2019)
    Berry_curvature = zeros(Nkx-1,Nky-1)
    for nkx=1:Nkx-1
        for nky=1:Nky-1
            Ux = dot(WF[:,nkx,nky],WF[:,nkx+1,nky])
            Uy = dot(WF[:,nkx,nky],WF[:,nkx,nky+1])
            Uxy = dot(WF[:,nkx+1,nky],WF[:,nkx+1,nky+1])
            Uyx = dot(WF[:,nkx,nky+1],WF[:,nkx+1,nky+1])
            Wloop = Ux*Uxy*conj(Uyx*Uy) # Wilson loop
            phase = rem2pi(angle(Wloop),RoundNearest) # Ensure that the phase is between -pi and +pi
            Berry_curvature[nkx,nky] = phase #/(dkx*dky)
        end
    end

    # Chern number
    Chernnum = sum(Berry_curvature)/2pi # *dkx*dky

    # Gaps and bandwidth
    Edgap = minimum(E[2,:,:].-E[1,:,:])
    Eigap = minimum(E[2,:,:]) - maximum(E[1,:,:])
    BW = maximum(E[1,:,:]) - minimum(E[1,:,:])

    return [Chernnum, Edgap, Eigap, BW]
end

function compute_quantum_metric(k::Vector{Float64}, band, DE, matrix_elements, BZparams)
    kappa1, kappa2, ktheta, g1, g2, g0 = BZparams
    hx = 1e-6*abs(g0)
    hy = 1e-6*abs(g0)
    H = compute_H(k, DE, matrix_elements, BZparams)
    F = eigen(H)
    Ek = real(F.values)
    WF = F.vectors
    WF0 = WF[:,band]
    # Derivatives of Hamiltonian
    vx = (compute_H(k+hx*[1, 0], DE, matrix_elements, BZparams)-compute_H(k-hx*[1, 0], DE, matrix_elements, BZparams))/(2*hx)
    vy = (compute_H(k+hy*[0, 1], DE, matrix_elements, BZparams)-compute_H(k-hy*[0, 1], DE, matrix_elements, BZparams))/(2*hy)
    BC = 0.0
    FS = 0.0
    # for n in [j for j in 1:Nbands if j!=band]
    for n in [j for j in 1:Nbands-1 if Ek[j]!=Ek[band]]
        den = (Ek[band]-Ek[n])^2
        BC += 2*imag(dot(WF0,vx,WF[:,n])*dot(WF[:,n],vy,WF0))/den
        FS += ( abs2(dot(WF0,vx,WF[:,n])) + abs2(dot(WF0,vy,WF[:,n])) )/den
    end
    return BC, FS
end

function quantum_metric(band, DE, a_l, BZparams)
    kappa1, kappa2, ktheta, g1, g2, g0 = BZparams
    matrix_elements = exciton_matrix_elements(a_l, BZparams)

    # Define k-grid in mBZ (map hexagon into square)
    Nkx = 100 # Number of points in x direction
    Nky = 100 # Number of points in y direction
    kvec = zeros(2,Nkx,Nky)
    Threads.@threads for nkx=1:Nkx
        Threads.@threads for nky=1:Nky
            kvec[:,nkx,nky] = nkx/Nkx*g1 + nky/Nky*g2
        end
    end
    dkfactor = g0/Nkx * g0/Nky * 0.5*sqrt(3)

    Berry_curvature = zeros(Nkx,Nky)
    FS_metric = zeros(Nkx,Nky)
    Threads.@threads for nkx=1:Nkx
        Threads.@threads for nky=1:Nky
            k = kvec[:,nkx,nky] # Convert k to the right type
            Berry_curvature[nkx,nky], FS_metric[nkx,nky] = compute_quantum_metric(k, band, DE, matrix_elements, BZparams)
        end
    end

    # Integral of the trace of the FS metric
    trg = sum(FS_metric)*dkfactor/2pi
    # Integral of the Berry curvature (i.e. Chern number)
    Chern = sum(Berry_curvature)*dkfactor/2pi
    println("tr(g)=$trg, C=$Chern")


    # Store for plotting
    # Copy four rhomboidal BZs. We will clip a hexagonal shape when plotting
    kvec_plot = zeros(2,4*Nkx*Nky)
    BC_plot = zeros(4*Nkx*Nky)
    FS_plot = zeros(4*Nkx*Nky)
    Threads.@threads for nkx=1:2*Nkx
        Threads.@threads for nky=1:2*Nky
            nk = (nkx-1)*2*Nky+nky
            kvec_plot[:,nk] = (nkx-Nkx)/Nkx*g1 + (nky-Nky)/Nky*g2
            nkx0 = mod(nkx,1:Nkx)
            nky0 = mod(nky,1:Nky)
            BC_plot[nk] = Berry_curvature[nkx0,nky0]
            FS_plot[nk] = FS_metric[nkx0,nky0]
        end
    end
    mkpath("Results/")
    open("Results/quantum_metric_$(Nkx)_$(Nky).dat", "w") do f
        write(f, "#kx\tky\tBerry curvature\tFS metric\n")
        for nk=1:Nkx*Nky*4 
            k = reshape(kvec_plot[:,nk],1,2) # Convert k to the right type
            kx, ky = k./g0
            BC = BC_plot[nk]
            FS = FS_plot[nk] # Trace of Fubini-Study metric (nm²)
            write(f, "$kx\t$ky\t$BC\t$FS\n")
        end
    end


    return trg, Chern
end

function Chern_scan(a_l)
    band = 1

    Ntheta = 101
    NDE = 101
    theta_list = LinRange(1.0, 3.0, Ntheta)
    DE_list = LinRange(-0.01, 0.01, NDE)
    Chern_scan = zeros(4,Ntheta, NDE)
    Threads.@threads for ntheta=1:Ntheta
        for nDE=1:NDE
            theta = theta_list[ntheta]*pi/180
            BZparams = initialize_BZ(theta)
            DE = DE_list[nDE]
            # println("θ=$(theta*180/pi), DE=$(DE*1e3) meV")
            Chern_scan[:, ntheta, nDE] = compute_Chern_number(band, DE, a_l, BZparams)
        end
    end

    open("Results/Chern_scan.dat", "w") do f
        write(f, "#theta\tDE (meV)\tChern number\tDirect gap (meV)\tIndirect gap (meV)\tBandwidth (meV)")
        for ntheta=1:Ntheta
            for nDE=1:NDE
                theta = theta_list[ntheta]*pi/180
                DE = DE_list[nDE]
                C, Edgap, Eigap, BW = Chern_scan[:, ntheta, nDE]
                write(f, "\n$(theta*180/pi)\t$(DE*1e3)\t$(C)\t$(Edgap*1e3)\t$(Eigap*1e3)\t$(BW*1e3)")
            end
        end
    end

end

function find_minBW(a_l)
    band = 1
    function varfun(x, grad)
        theta = x[1]*pi/180
        BZparams = initialize_BZ(theta)
        DE = x[2]
        results = compute_Chern_number(band, DE, a_l, BZparams)
        C = results[1]
        if abs(C)<0.1 # C=0
            BW = 1e12 # Artificially return large bandwidth if the band is trivial
        else
            BW = results[4]
        end
        return BW
    end
    opt = Opt(:LN_NELDERMEAD, 2)
    opt.lower_bounds = [1.5, -0.005]
    opt.upper_bounds = [3.0, 0.005]
    opt.xtol_rel = 1e-12
    opt.ftol_rel = 1e-12
    opt.maxeval = 1000
    opt.min_objective = varfun
    (minf,minx,ret) = optimize(opt, [2.0,-0.0039])
    numevals = opt.numevals # the number of function evaluations
    println("got $minf at $minx after $numevals iterations (returned $ret)")
end


# Compute exciton Bohr radius and energy difference
par1, par2 = exciton_states_variational.([1,2]) # Bohr radii from variational calculation
E_l = [par1[1], par2[1]]
a_l = [par1[2], par2[2]]
# a_l = [1.0,1.0]
println(E_l)
println(a_l)

# Parameters
DE = -0.0038
theta = 1.95pi/180
# theta = LinRange(1.6,3.5,20)[parse(Int,ARGS[0])]*pi/180
BZparams = initialize_BZ(theta)

kappa1, kappa2, ktheta, g1, g2, g0 = BZparams
# println(ktheta)
# exciton_matrix_elements(a_l, BZparams)

# Plot bands
@time exciton_bands(DE, a_l, BZparams)

# Compute Chern numbers of lowest bands
for band in [1]
    @time C = compute_Chern_number(band, DE, a_l, BZparams)
    println("C=$(C[1])")
end

# band = 1
# @time quantum_metric(band, DE, a_l, BZparams)

# Parameter scan of Chern number
# @time Chern_scan(a_l)

# Find parameters that minimize the bandwidth (in the regime where the band is topological)
# @time find_minBW(a_l)