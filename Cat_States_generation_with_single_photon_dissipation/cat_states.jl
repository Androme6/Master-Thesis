using QuantumToolbox
using CairoMakie

# 1. System Parameters
wa1 = 0.0
wa2 = 0.0
g = 0.1

k = 1.0
F = 0.5

Na1 = 20
Na2 = 5

tmax = 50
steps = 1000

# 2. Operators
a1 = tensor(destroy(Na1), qeye(Na2))
a2 = tensor(qeye(Na1), destroy(Na2))

# 3. Hamiltonian Construction
function Hamiltonian(wa1, wa2, g, F)
    H = wa1 * a1' * a1 + wa2 * a2' * a2 + g * (a1 * a1 * a2' + a1' * a1' * a2) + F * (a2 + a2')
    return H
end

# 4. Evolution
function cat_evolution(wa1, wa2, g, k, F, na1_0, tmax, steps = 1000)
    H = Hamiltonian(wa1, wa2, g, F)

    # Jump operator
    c_ops = [sqrt(k) * a2 ]

    # Initial state: |na1_0, 0>
    psi0 = tensor(fock(Na1, na1_0), fock(Na2, 0))

    # Evolve the system
    t = LinRange(0.0, tmax, steps)
    sol = mesolve(H, psi0, t, c_ops)

    return sol 
end

# 5. Run the evolution for both even and odd cat states
sol_p = cat_evolution(wa1, wa2, g, k, F, 0, tmax, steps)
out_p = ptrace(sol_p.states[end], 1)

sol_m = cat_evolution(wa1, wa2, g, k, F, 1, tmax, steps)
out_m = ptrace(sol_m.states[end], 1)

# 6. Compare simulation and analytical results
alpha_analytical = sqrt(Complex(-F/g))
println("Analytical amplitude α: ", round(alpha_analytical, digits=3))

cat_plus = coherent(Na1, alpha_analytical) + coherent(Na1, -alpha_analytical)
cat_plus_analytical = normalize(cat_plus)
rho_plus_analytical = ket2dm(cat_plus_analytical)
fid_plus = fidelity(out_p, rho_plus_analytical)

println("Even: Fidelity (Simulation vs Analytical): ", round(real(fid_plus), digits=5))

cat_minus = coherent(Na1, alpha_analytical) - coherent(Na1, -alpha_analytical)
cat_minus_analytical = normalize(cat_minus)
rho_minus_analytical = ket2dm(cat_minus_analytical)
fid_minus = fidelity(out_m, rho_minus_analytical)

println("Odd: Fidelity (Simulation vs Analytical): ", round(real(fid_minus), digits=5))

#%%
# 7. Plotting the Wigner functions for both cats
xvec = LinRange(-5, 5, 100)
yvec = LinRange(-5, 5, 100)

W_p = wigner(out_p, xvec, yvec)'
W_m = wigner(out_m, xvec, yvec)'


fig = Figure(size = (900, 1100))

# Even cat state
ax2D_p = Axis(fig[1, 1], 
            title = "2D + Cat State Wigner Function",
            xlabel = "Re(α)", 
            ylabel = "Im(α)",
            aspect = 1) 
hm_p = CairoMakie.heatmap!(ax2D_p, xvec, yvec, W_p, colormap = :RdBu)

ax3D_p = Axis3(fig[1, 3], 
             title = "3D + Cat State Wigner Function",
             xlabel = "Re(α)", 
             ylabel = "Im(α)", 
             zlabel = "W(α)",
             elevation = pi/6, 
             azimuth = pi/4) 
CairoMakie.surface!(ax3D_p, xvec, yvec, W_p, colormap = :RdBu)

Colorbar(fig[1, 2], hm_p, label = "W(α)")

# Odd cat state
ax2D_m = Axis(fig[2, 1], 
            title = "2D - Cat State Wigner Function",
            xlabel = "Re(α)", 
            ylabel = "Im(α)",
            aspect = 1) 
hm_m = CairoMakie.heatmap!(ax2D_m, xvec, yvec, W_m, colormap = :RdBu)

CairoMakie.colgap!(fig.layout, 50)

ax3D_m = Axis3(fig[2, 3], 
             title = "3D - Cat State Wigner Function",
             xlabel = "Re(α)", 
             ylabel = "Im(α)", 
             zlabel = "W(α)",
             elevation = pi/6, 
             azimuth = pi/4) 
CairoMakie.surface!(ax3D_m, xvec, yvec, W_m, colormap = :RdBu)

Colorbar(fig[2, 2], hm_m, label = "W(α)")


display(fig)
