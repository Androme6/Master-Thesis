using QuantumToolbox
using Plots
using LaTeXStrings
using Printf
using CUDA 
using SparseArrays

# 1. System Parameters
wa1 = 1.0
wa2 = 2.066  
wq  = 3.0
ga1 = 0.1
ga2 = 0.3
theta = π / 6.0

ka1 = 0.02
ka2 = 0.04

Na1 = 8
Na2 = 8
Nq = 2
dim_tot = Na1 * Na2 * Nq 

tmax = 300
steps = 300

# 2. Operators 
a1 = tensor(destroy(Na1), qeye(Na2), qeye(Nq))
a2 = tensor(qeye(Na1), destroy(Na2), qeye(Nq))
sz   = tensor(qeye(Na1), qeye(Na2), sigmaz())
sx   = tensor(qeye(Na1), qeye(Na2), sigmax())

# 3. Hamiltonian Construction
function FreeHamiltonian(wa1, wa2, wq)
    H0 = wa1 * a1' * a1 + wa2 * a2' * a2 + (wq / 2) * sz
    return H0
end

function GenQRabiHamiltonian(ga1, ga2, theta)
    Hint = (ga1 * (a1 + a1') + ga2 * (a2 + a2')) * (sx * cos(theta) + sz * sin(theta))
    return Hint
end

H0 = FreeHamiltonian(wa1, wa2, wq)
Hint = GenQRabiHamiltonian(ga1, ga2, theta)
H = H0 + Hint

# 4. Jump Operators
field1 = 1im * sqrt(ka1 / wa1) * (a1 - a1')
field2 = 1im * sqrt(ka2 / wa2) * (a2 - a2')
fields = [field1, field2]
T_baths = [0.0, 0.0]

# 5. Dressed Liouvillian
println("Generating Dressed Liouvillian on CPU...")
e_d, v_d, L_cpu = liouvillian_dressed_nonsecular(H, fields, T_baths)

# 6. Finding the true Dressed Ground State Energy
E_gs = e_d[1] 
println("Dressed Ground State Energy: ", round(E_gs, digits=4))

# 7. Constructing the V matrix for basis transformation
V_mat = Array(v_d.data)

#%%
# 8. GPU transfer of the Liouvillian and initial state
println("Transferring Liouvillian to GPU...")
L_gpu = cu(L_cpu)

# 9. Initial state: Dressed Vacuum State
psi0_dressed_cpu = ket2dm(tensor(fock(Na1, 0), fock(Na2, 0), basis(Nq, 0))) 
psi0_dressed_gpu = cu(psi0_dressed_cpu)

# 10. Time Evolution
t = LinRange(0.0, tmax, steps)
println("Time evolution on GPU...")
sol_gpu = mesolve(L_gpu, psi0_dressed_gpu, t) 


# 11. Post-processing: Transforming back to the bare basis and calculating observables
println("Post-processing results...")

energy = zeros(length(t))
pop_gs  = zeros(length(t))

labels_list = String[]
bare_projs = Matrix{ComplexF64}[]

# basis(2, 1) is |g> and basis(2, 0) is |e> 
g_state = 1

# Construct projectors for all bare states |n1, n2, q><n1, n2, q|
for n1 in 0:(Na1-1)
    for n2 in 0:(Na2-1)
        for q in 0:(Nq-1)
            proj = Array(ket2dm(tensor(fock(Na1, n1), fock(Na2, n2), basis(Nq, q))).data)
            push!(bare_projs, proj)
            q_str = (q == g_state) ? "g" : "e"
            push!(labels_list, "|$(n1), $(n2), $(q_str)>")
        end
    end
end

# Track energy, population of dressed ground state, and leakage into bare states
leakage_matrix = zeros(length(bare_projs), length(t))

# Convert H to matrix form for energy calculation
H_mat = Array(H.data)

# Dressed ground state projector for population calculation
vec_gs = V_mat[:, 1]
psi_gs_proj = vec_gs * vec_gs'

# Loop over all time points to calculate observables
for (i, rho_dressed_qobj) in enumerate(sol_gpu.states)
    rho_dressed = Array(rho_dressed_qobj.data)
    
    rho_bare = V_mat * rho_dressed * V_mat'
    
    energy[i] = real(tr(rho_bare * H_mat))
    pop_gs[i]  = real(tr(rho_bare * psi_gs_proj))
    
    for (j, proj) in enumerate(bare_projs)
        leakage_matrix[j, i] = real(tr(rho_bare * proj))
    end
end

# 12. Analyze leakage: Find the most populated bare states over time
leakage_peaks = []
for j in 1:length(bare_projs)
    push!(leakage_peaks, (maximum(leakage_matrix[j, :]), j))
end
sort!(leakage_peaks, rev=true)

println("\n--- Top Populated Bare States (Should be flat!) ---")
for k in 1:min(5, length(leakage_peaks))
    val, idx = leakage_peaks[k]
    @printf("%-15s : %.6f\n", labels_list[idx], val)
end

# 13. Population of Bare Vacuum State state
idx_00g = findfirst(x -> x == "|0, 0, g>", labels_list)
pop_00g = real.(sol.expect[idx_00g, :])

#%%
# 14. Plotting

# Panel 1: Energy 
p1 = Plots.plot(t, energy, label="<H>", color=:red, lw=2,
          ylabel=L"Energy \ (\hbar\omega_1)", xlabel=L"Time \ (1/\omega_1)",
          title="System Energy vs Time", framestyle=:box)

# Panel 2: Population of Bare Vacuum State
p2 = Plots.plot(ylims = (0.98,1.01), t, leakage_matrix[idx_00g, :], label=L"|0, 0, g\rangle", color=:blue, lw=2,
          xlabel=L"Time \ (1/\omega_1)", ylabel="Population", 
          title="Bare Vacuum State Population vs Time", framestyle=:box)

# Panel 3: Population of Dressed Ground State
p3 = Plots.plot(ylims = (0.98,1.01), t, pop_gs, label=L"|G\rangle", color=:blue, lw=2,
          xlabel=L"Time \ (1/\omega_1)", ylabel="Population", 
          title="Dressed Ground State Population vs Time", framestyle=:box)

# Panel 4: Plot the top 5 leakage states
p4 = Plots.plot(xlabel=L"Time \ (1/\omega_1)", ylabel="Population", 
          title="Top Populated Leakage States", framestyle=:box)

for k in 1:5
    val, idx = leakage_peaks[k]
    Plots.plot!(p4, t, leakage_matrix[idx, :], label=labels_list[idx], lw=2)
end

# Combine into a 3-row layout
fig = Plots.plot(p1, p2, p3, p4, layout=(4, 1), size=(800, 1000), margin=5Plots.mm)
display(fig)