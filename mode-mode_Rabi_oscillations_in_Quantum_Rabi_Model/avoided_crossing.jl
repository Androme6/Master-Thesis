using QuantumToolbox
using Plots
using LaTeXStrings

# 1. System Parameters
wa1 = 1.0                  
wq = 2.5                  
ga1 = 0.2            
ga2 = 0.4          
theta = π / 6.0           

Na1 = 40 
Na2 = 40
Nq = 2

# 2. Operators 
a1  = tensor(destroy(Na1), qeye(Na2), qeye(Nq))
a2  = tensor(qeye(Na1), destroy(Na2), qeye(Nq))
sz = tensor(qeye(Na1), qeye(Na2), sigmaz())
sx = tensor(qeye(Na1), qeye(Na2), sigmax())
sm = tensor(qeye(Na1), qeye(Na2), sigmam())
sp = sm' 

# 3. Frequency Sweep
wa2_vals = range(1 * wa1, 2.5 * wa1, length=300)

# 4. Preallocate Arrays for Eigenenergies (first 15)
n_states = 15
evals_jc   = zeros(length(wa2_vals), n_states)
evals_rabi = zeros(length(wa2_vals), n_states)
evals_gen  = zeros(length(wa2_vals), n_states)

# 5. Precompute interaction terms that don't depend on wa2
Hint_jc = ga1 * (a1 * sp + a1' * sm) + ga2 * (a2 * sp + a2' * sm)
Hint_rabi = (ga1 * (a1 + a1') + ga2 * (a2 + a2')) * sx
Hint_gen = (ga1 * (a1 + a1') + ga2 * (a2 + a2')) * (sx * cos(theta) + sz * sin(theta))

# 6. Compute Eigenenergies 
for (i, wa2) in enumerate(wa2_vals)
    H0 = wa1 * a1' * a1 + wa2 * a2' * a2 + wq * sz / 2.0

    H_jc = H0 + Hint_jc
    H_rabi = H0 + Hint_rabi
    H_gen = H0 + Hint_gen
    
    println("Calculating eigenenergies for w_2 = ", round(wa2, digits=3), " (", round(wa2/wa1, digits=3), " ω_1) - Progress: ", round(100 * i / length(wa2_vals), digits=1), "%")
    
    evals_jc[i, :]   = sort(real.(eigsolve(H_jc, eigvals=n_states, sigma=-2.0).values))
    evals_rabi[i, :] = sort(real.(eigsolve(H_rabi, eigvals=n_states, sigma=-2.0).values))
    evals_gen[i, :]  = sort(real.(eigsolve(H_gen, eigvals=n_states, sigma=-2.0).values))
end


#%%
# 7. Plotting
fig = Plots.plot(xlims=(1.0, 2.5), ylims=(-1, 5),
           xlabel=L"\omega_2/\omega_1", ylabel=L"E/\hbar\omega_1",
           legend=:bottomright, size=(800, 600), framestyle=:box)

for i in 1:15  
    Plots.plot!(fig, wa2_vals ./ wa1, evals_jc[:, i] -evals_jc[:, 1],
          ls=:dashdot, color=:blue, alpha=0.5, label=(i==2 ? "JC" : ""))
          
    Plots.plot!(fig, wa2_vals ./ wa1, evals_rabi[:, i] -evals_rabi[:, 1],
          ls=:dash, color=:orange, alpha=0.5, label=(i==2 ? "Rabi" : ""))
          
    Plots.plot!(fig, wa2_vals ./ wa1, evals_gen[:, i] -evals_gen[:, 1],
          ls=:solid, color=:red, linewidth=1.5, label=(i==2 ? "Gen. Rabi" : ""))
end

# 8. Find the minimum gap between |2,0,g⟩ and |0,1,g⟩ in the Gen. Rabi case
window_mask = (wa2_vals .> 1.5 * wa1) .& (wa2_vals .< 2.5 * wa1)
window_indices = findall(window_mask)
gaps_in_window = evals_gen[window_indices, 4] - evals_gen[window_indices, 3]
min_gap, local_min_idx = findmin(gaps_in_window)
actual_min_idx = window_indices[local_min_idx]

println("Simulated minimum gap between |2,0,g⟩ and |0,1,g⟩: ", round(min_gap, digits=6))
println("Occurs at ω_2/ω_1 = ", round(wa2_vals[actual_min_idx] / wa1, digits=6))

# 9. Theoretical calculation
numerator = 3 * sqrt(2) * ga2 * (ga1^2) * (wq^2) * sin(2 * theta) * cos(theta)
denominator = 4 * (wa1^4) - 5 * (wa1^2) * (wq^2) + (wq^4)
g_eff = numerator / denominator
theoretical_gap = 2*abs(g_eff) / sqrt(2)  

println("Perturbation theory minimum gap: ", round(theoretical_gap, digits=6))
println("Absolute difference: ", round(abs(min_gap - theoretical_gap), digits=6))


# 10. Plot the bare energies of |2,0,g⟩ and |0,1,g⟩ for reference
E_20g = 0 .* wa2_vals .+ (2 * wa1) .- (wq / 2)  # This is constant across the wa sweep
E_01g = 1 .* wa2_vals .+ (0 * wa1) .- (wq / 2)  # This increases linearly with wa

Plots.plot!(fig, wa2_vals ./ wa1, E_20g- evals_rabi[:, 1], 
       color=:black, linestyle=:dot, linewidth=3, alpha=0.8, 
       label=L"|2, 0, g\rangle \quad \texttt{ unperturbed}")
   
Plots.plot!(fig, wa2_vals ./ wa1, E_01g - evals_rabi[:, 1],
      color=:green, linestyle=:dot, linewidth=3, alpha=0.8, 
      label=L"|0, 1, g\rangle \quad \texttt{ unperturbed}")

display(fig)
