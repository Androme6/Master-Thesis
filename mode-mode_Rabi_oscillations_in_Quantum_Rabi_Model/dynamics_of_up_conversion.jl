using QuantumToolbox
using Plots
using LaTeXStrings
using CUDA

# 1. System Parameters
wa1 = 1.0
wa2 = 2.143813
wq = 2.5
ga1 = 0.2
ga2 = 0.4
theta = π / 6.0

gap_simulated = 0.039721

Na1= 30
Na2= 30
Nq = 2

tmax = 300
steps = 1000

wa2_off = 2

# 2. Operators
a1 = tensor(destroy(Na1), qeye(Na2), qeye(Nq))
a2 = tensor(qeye(Na1), destroy(Na2), qeye(Nq))
sz   = tensor(qeye(Na1), qeye(Na2), sigmaz())
sx   = tensor(qeye(Na1), qeye(Na2), sigmax())

# 3. Initial, Target and Third States and projectors
psi_bare_20g = tensor(fock(Na1, 2), fock(Na2, 0), basis(Nq, 1))
psi_bare_01g = tensor(fock(Na1, 0), fock(Na2, 1), basis(Nq, 1))
psi_bare_10e = tensor(fock(Na1,1), fock(Na2,0), basis(Nq,0))

P_bare_20g = ket2dm(psi_bare_20g)
P_bare_01g = ket2dm(psi_bare_01g)
P_bare_10e = ket2dm(psi_bare_10e)

# 4. Hamiltonian Construction
function FreeHamiltonian(wa1, wa2, wq)
    H0 = wa1 * a1' * a1 + wa2 * a2' * a2 + (wq / 2) * sz
    return H0
end

function GenQRabiHamiltonian(ga1, ga2, theta)
    Hint = (ga1 * (a1 + a1') + ga2 * (a2 + a2')) * (sx * cos(theta) + sz * sin(theta))
    return Hint
end

function EffectiveHamiltonian(wa1, wq, ga1, ga2, theta)
    #numerator = 3  * ga2 * (ga1^2) * (wq^2) * sin(2 * theta) * cos(theta)
    #denominator = 4 * (wa1^4) - 5 * (wa1^2) * (wq^2) + (wq^4)
    #g_eff = numerator / denominator
    #g_eff = g_eff/sqrt(2) 
    
    g_eff = gap_simulated/(2*sqrt(2))
    Heff = g_eff* (a1' * a1' * a2 + a1 * a1 * a2') 
    return Heff
end

H0 = FreeHamiltonian(wa1, wa2, wq) 
H0_off = FreeHamiltonian(wa1, wa2_off, wq)
Hint_rabi = GenQRabiHamiltonian(ga1, ga2, theta)
Hint_eff = EffectiveHamiltonian(wa1, wq, ga1, ga2, theta)

H_rabi = H0 + Hint_rabi
H_rabi_off = H0_off + Hint_rabi
H_eff = FreeHamiltonian(wa1, 2.0, wq) + Hint_eff


#%%
# 5. Find Dressed 20g and 10e states at the off-resonant frequency
# We look for the states that have the highest fidelity with our target states,
# which are the bare |2,0,g⟩ and |1,0,e⟩ states.
# We look off resonance to avoid the strong hybridization that occurs at resonance,
# which would make it harder to identify the correct states.

_, ψ, _ = eigenstates(H_rabi_off)

# 20g state
f_max_20g, idx_max_20g = findmax(vi -> fidelity(vi, psi_bare_20g), ψ[1:5])
psi_dressed_20g = ψ[idx_max_20g]

# 01g state
f_max_01g, idx_max_01g = findmax(vi -> fidelity(vi, psi_bare_01g), ψ[1:5])
psi_dressed_01g = ψ[idx_max_01g]

# 10e state
f_max_10e, idx_max_10e = findmax(vi -> fidelity(vi, psi_bare_10e), ψ[1:7])
psi_dressed_10e = ψ[idx_max_10e]

println("Fidelity for |2,0,g⟩: ", round(f_max_20g, digits=4))
println("Fidelity for |0,1,g⟩: ", round(f_max_01g, digits=4))
println("Fidelity for |1,0,e⟩: ", round(f_max_10e, digits=4))

P_dressed_20g = ket2dm(psi_dressed_20g)
P_dressed_01g = ket2dm(psi_dressed_01g)
P_dressed_10e = ket2dm(psi_dressed_10e)

#%%

# 6. Time Evolution
t = LinRange(0, tmax, steps)

sol_rabi = sesolve(H_rabi, psi_dressed_20g, t, e_ops=[P_dressed_20g, P_dressed_01g, P_dressed_10e])
sol_rabi_off = sesolve(H_rabi_off, psi_dressed_20g, t, e_ops=[P_dressed_20g, P_dressed_01g])
sol_eff = sesolve(H_eff, psi_bare_20g, t, e_ops=[P_bare_20g, P_bare_01g])




#%%
# 7. Plotting
# Panel 1: Off-Resonant Dynamics
p_off = Plots.plot(t, real.(sol_rabi_off.expect[1, :]), 
             label="Full "*L"|2, 0, g\rangle", linewidth=2, color=:blue, linestyle=:solid,
             xlabel="Time " * L"(1/\omega_1)", ylabel="Population",
             title="Off Resonant "*L"(\omega_2 = " * string(wa2_off) * ")", framestyle=:box)
Plots.plot!(p_off, t, real.(sol_rabi_off.expect[2, :]), 
      label="Full "*L"|0, 1, g\rangle", linewidth=2, color=:orange, linestyle=:solid)

# Panel 2: On-Resonant Dynamics
p_res = Plots.plot(t, real.(sol_rabi.expect[1, :]), 
             label="Full "*L"|2, 0, g\rangle", linewidth=2, color=:blue, linestyle=:solid,
             xlabel="Time " * L"(1/\omega_1)", ylabel="Population",
             title="On Resonance "*L"(\omega_2 \approx %$(round(wa2, digits=3)))", framestyle=:box)
Plots.plot!(p_res, t, real.(sol_rabi.expect[2, :]), 
      label="Full "*L"|0, 1, g\rangle", linewidth=2, color=:orange, linestyle=:solid)
Plots.plot!(p_res, t, real.(sol_eff.expect[1, :]), 
      label="Eff "*L"|2, 0, g\rangle", linewidth=2, color=:blue, linestyle=:dash)
Plots.plot!(p_res, t, real.(sol_eff.expect[2, :]), 
      label="Eff "*L"|0, 1, g\rangle", linewidth=2, color=:orange, linestyle=:dash)

# Panel 3: Zoomed-In On-Resonant Dynamics to show the small oscillations
p_zoom = Plots.plot(t[1:30], real.(sol_rabi.expect[1, 1:30]), 
             label="Full "*L"|2, 0, g\rangle", linewidth=2, color=:blue, linestyle=:solid,
             xlabel="Time " * L"(1/\omega_1)", ylabel="Population",
             title="On Resonance small oscillations "*L"(\omega_2 \approx %$(round(wa2, digits=3)))", framestyle=:box)
Plots.plot!(p_zoom, t[1:30], real.(sol_rabi.expect[3, 1:30]), 
      label="Full "*L"|1, 0, e\rangle", linewidth=2, color=:green, linestyle=:solid)


l = @layout [a b; c]

fig_master = Plots.plot(p_off, p_zoom, p_res, layout=l, size=(1200, 800), 
                  left_margin=5Plots.mm, right_margin=5Plots.mm, bottom_margin=5Plots.mm)
display(fig_master)
