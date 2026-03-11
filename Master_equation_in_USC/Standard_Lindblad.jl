using QuantumToolbox
using Plots
using LaTeXStrings
using Printf


# 1. System Parameters
wa1 = 1.0
wa2 = 2.066
wq = 3.0
ga1 = 0.1
ga2 = 0.3
theta = π / 6.0

ka1 = 0.02
ka2 = 0.04

Na1 = 10
Na2 = 10
Nq = 2

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

# 4. Finding the true Dressed Ground State
evals, ψ = eigenstates(H)
E_gs = evals[1]
psi_dressed_gs = ψ[1] 
println("Dressed Ground State Energy: ", round(E_gs, digits=4))

# 5. Costructing the tracker list for mesolve 
e_ops_list = QuantumObject[H, ket2dm(psi_dressed_gs)] 
labels_list = ["Energy", "|G> (Dressed Ground State)"]

# basis(2, 1) is |g> and basis(2, 0) is |e> 
g_state = 1

println("Generating trackers for all $(Na1 * Na2 * Nq) states...")
for n1 in 0:(Na1-1)
    for n2 in 0:(Na2-1)
        for q in 0:(Nq-1)
            # Create projector |n1, n2, q><n1, n2, q|
            state_proj = ket2dm(tensor(fock(Na1, n1), fock(Na2, n2), basis(Nq, q)))
            push!(e_ops_list, state_proj)
            # Create a readable label
            q_str = (q == g_state) ? "g" : "e"
            push!(labels_list, "|$(n1), $(n2), $(q_str)>")
        end
    end
end

# 6. Solving the Master Equation
t = LinRange(0.0, tmax, steps)
c_ops = [sqrt(ka1) * a1, sqrt(ka2) * a2]
sol = mesolve(H, psi_dressed_gs, t, c_ops, e_ops = e_ops_list)

# 7. Data Processing 
energy = real.(sol.expect[1, :])

# Find the specific index of our starting state
idx_00g = findfirst(x -> x == "|0, 0, g>", labels_list)
pop_00g = real.(sol.expect[idx_00g, :])

# Search for the highest populated states (excluding Energy and |0,0,g>)
leakage_peaks = []
for i in 3:length(labels_list)
    if i != idx_00g
        peak_pop = maximum(real.(sol.expect[i, :]))
        push!(leakage_peaks, (peak_pop, i))
    end
end

# Sort descending to find the biggest culprits
sort!(leakage_peaks, rev=true)

println("\n--- Top 5 Leakage States ---")
for k in 1:5
    val, idx = leakage_peaks[k]
    @printf("%-15s : %.4f\n", labels_list[idx], val)
end

#%%
# 8. Plotting
# Panel 1: Energy
p1 = Plots.plot(t, energy, label="<H>", color=:red, lw=2,
          ylabel=L"Energy \ (\hbar\omega_1)", xlabel=L"Time \ (1/\omega_1)",
          title="System Energy vs Time", framestyle=:box)

# Panel 2: Population 
p2 = Plots.plot(t, pop_00g, label=L"|0,0,g\rangle", color=:blue, lw=2,
          xlabel=L"Time \ (1/\omega_1)", ylabel="Population", 
          title="Bare Vacuum Population vs Time", framestyle=:box)

# Panel 3: Top Leakage States
p3 = Plots.plot(xlabel=L"Time \ (1/\omega_1)", ylabel="Population", 
          title="Top Populated Leakage States", framestyle=:box)
for k in 1:5
    val, idx = leakage_peaks[k]
    Plots.plot!(p3, t, real.(sol.expect[idx, :]), label=labels_list[idx], lw=2)
end

# Combine into a 3-row layout
fig = Plots.plot(p1, p2, p3, layout=(3, 1), size=(800, 900), margin=5Plots.mm)
Plots.display(fig)
