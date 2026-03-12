using QuantumToolbox
using Plots
using CairoMakie
using LaTeXStrings
using Printf
using CUDA 
using SparseArrays
using Dates
using JLD2



# Save Function
function save_simulation_minimal(filepath, sol_gpu, V_mat, t)
    println("Preparing minimal data for saving (moving CuArrays to CPU)...")
    
    # Extract states and move to CPU Arrays
    states_cpu_mats = [Array(state.data) for state in sol_gpu.states]
    
    # Save only the fundamental parameters needed for reconstruction
    params = Dict(
        "wa1" => wa1, "wa2" => wa2, "wq" => wq, "ga1" => ga1, "ga2" => ga2, 
        "theta" => theta, "gap_simulated" => gap_simulated, "wd" => wd, "F" => F, 
        "na1_0" => na1_0, "ka2" => ka2, "Na1" => Na1, "Na2" => Na2, "Nq" => Nq, 
        "tmax" => tmax, "steps" => steps
    )

    # Save to disk
    jldsave(filepath; 
        states_cpu_mats = states_cpu_mats,
        V_mat = V_mat,
        t = t,
        params = params
    )
    println("Minimal simulation data saved to: ", filepath)
end

# Load Function
function load_simulation_minimal(filepath)
    println("Loading and reconstructing data from: ", filepath)
    data = load(filepath)
    
    params = data["params"]
    V_mat = data["V_mat"]
    t = data["t"]
    
    # 1. Reconstruct dims_sys and Operators
    Na1, Na2, Nq = params["Na1"], params["Na2"], params["Nq"]
    dims_sys = (Na1, Na2, Nq)
    
    n1 = tensor(num(Na1), qeye(Na2), qeye(Nq))
    n2 = tensor(qeye(Na1), num(Na2), qeye(Nq))
    
    n1_bare_mat = Array(n1.data)
    n2_bare_mat = Array(n2.data)

    # 2. Reconstruct QuantumObjects
    states_qobj = [QuantumObject(mat, type=Operator(), dims=dims_sys) for mat in data["states_cpu_mats"]]
    
    # 3. Recalculate Observables
    println("Recalculating expectations...")
    expect_n1 = zeros(length(t))
    expect_n2 = zeros(length(t))

    for (i, rho_dressed_mat) in enumerate(data["states_cpu_mats"]) # Using the raw matrix directly is faster here
        rho_bare = V_mat * rho_dressed_mat * V_mat'
        expect_n1[i] = real(tr(rho_bare * n1_bare_mat))
        expect_n2[i] = real(tr(rho_bare * n2_bare_mat))
    end
    
    return (
        states = states_qobj,
        V_mat = V_mat,
        t = t,
        expect_n1 = expect_n1,
        expect_n2 = expect_n2,
        dims_sys = dims_sys,
        params = params
    )
end

# Animated Wigner Function
function animate_wigner_evolution(filepath, sol_states, V_mat, n1, t_array, w_exact, dims_sys; total_frames=150, fps=15)
    println("Preparing Wigner evolution animation...")
    
    # 1. Setup the grid (slightly lower resolution for faster rendering)
    xvec = LinRange(-5, 5, 80)
    yvec = LinRange(-5, 5, 80)
    
    # 2. Precompute the number operator matrix
    n1_mat = Array(n1.data)
    
    # 3. Create Observables (Makie's way of handling dynamic data)
    W_obs = Observable(zeros(Float64, length(xvec), length(yvec)))
    time_obs = Observable(0.0)
    
    # 4. Setup the Figure
    fig = Figure(size = (600, 500))
    
    # The title dynamically updates based on the time_obs
    ax = Axis(fig[1, 1], 
              title = @lift("Wigner Function at t = $(round($time_obs, digits=1))"), 
              xlabel = "Re(α)", ylabel = "Im(α)", aspect = 1)
              
    # Fix the color range so the colors don't wildly flash as the amplitude grows
    hm = CairoMakie.heatmap!(ax, xvec, yvec, W_obs, colormap = :RdBu, colorrange=(-0.3, 0.3))
    Colorbar(fig[1, 2], hm, label="W(α)")
    
    # 5. Determine which frames to calculate
    step_size = max(1, length(sol_states) ÷ total_frames)
    frame_indices = 1:step_size:length(sol_states)
    
    println("Recording $(length(frame_indices)) frames to $filepath ...")
    
    # 6. Record the animation
    CairoMakie.record(fig, filepath, frame_indices, framerate = fps) do i
        # Extract state safely
        state = sol_states[i]
        rho_dressed = state isa QuantumObject ? Array(state.data) : state
        
        # Transform to bare basis
        rho_bare = V_mat * rho_dressed * V_mat'
        
        # Apply Exact Rotating Frame Transformation
        t_current = t_array[i]
        U_rot = exp(1im * w_exact * t_current * n1_mat)
        rho_rot = U_rot * rho_bare * U_rot'
        
        # Wrap back into QuantumObject and trace out other modes
        rho_rot_qobj = QuantumObject(rho_rot, type=Operator(), dims=dims_sys)
        rho_mode1_rotated = ptrace(rho_rot_qobj, 1)
        
        # Calculate Wigner function
        W_cat = wigner(rho_mode1_rotated, xvec, yvec)'
        
        # Update the Observables (this instantly updates the plot!)
        W_obs[] = W_cat
        time_obs[] = t_current
    end
    
    println("Animation successfully saved!")
end



# 1. System Parameters
wa1 = 1.0
wa2 = 2.048495  
wq  = 2.5
ga1 = 0.1
ga2 = 0.2
theta = π / 6.0

gap_simulated = 0.006229

wd = wa2
F = 0.01

na1_0 = 0
ka2 = 0.01

Na1 = 25   
Na2 = 2
Nq = 2
dim_tot = Na1 * Na2 * Nq 

tmax = 6000
steps = 2000

save_dir = "C:\\Users\\andre\\Desktop\\Università\\Magistrale\\MA4\\Thesis\\Code\\Cats"
# Create the folder if it doesn't already exist
if !isdir(save_dir)
    mkdir(save_dir)
end

# 2. Operators 
a1 = tensor(destroy(Na1), qeye(Na2), qeye(Nq))
a2 = tensor(qeye(Na1), destroy(Na2), qeye(Nq))
sz   = tensor(qeye(Na1), qeye(Na2), sigmaz())
sx   = tensor(qeye(Na1), qeye(Na2), sigmax())
n1 = tensor(num(Na1), qeye(Na2), qeye(Nq))
n2 = tensor(qeye(Na1), num(Na2), qeye(Nq))
dims_sys = a1.dims 

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
field2 = 1im * sqrt(ka2 / wa2) * (a2 - a2')
fields = [field2]
T_baths = [0.0]

# 5. Dressed Liouvillian
println("Generating Dressed Liouvillian on CPU...")
e_d, v_d, L_cpu = liouvillian_dressed_nonsecular(H, fields, T_baths)

# 6. Constructing the V matrix for basis transformation
V_mat = Array(v_d.data)

# 7. GPU transfer of the Liouvillian and initial state
println("Transferring Liouvillian to GPU...")
L_gpu = cu(L_cpu)
# 8. Drive Hamiltonian 
H_drive_bare = F * (a2 + a2')
H_drive_dressed = V_mat' * Array(H_drive_bare.data) * V_mat

H_drive_dressed_qobj = QuantumObject(H_drive_dressed, type=Operator(), dims=dims_sys)

L_drive_dressed_cpu = liouvillian(H_drive_dressed_qobj)

L_drive_dressed_gpu = cu(L_drive_dressed_cpu)


# --------
# Drive frequency search
E_gs = e_d[1]
g_state = 1
P_20g = Array(ket2dm(tensor(fock(Na1, 2), fock(Na2, 0), basis(Nq, g_state))).data)
overlaps = [real(tr(P_20g * (V_mat[:, i] * V_mat[:, i]'))) for i in 1:dim_tot]
idx_20g = argmax(overlaps)
# The frequency is the energy difference between the |2,0,g> state and the ground state
true_wd = e_d[idx_20g] - E_gs
println("Drive frequency: ", true_wd)
wd = true_wd 
# --------


drive_func(p, t) = cos(wd * t)

# 9. Initial State (Dressed Ground State)
psi0_dressed_cpu = ket2dm(tensor(fock(Na1, 0), fock(Na2, 0), basis(Nq, 0))) 
psi0_dressed_gpu = cu(psi0_dressed_cpu)


#%%

# 10. Time Evolution
t = LinRange(0.0, tmax, steps)
println("Time evolution on GPU...")
L_tot_gpu = (L_gpu, (L_drive_dressed_gpu, drive_func))
sol_gpu = mesolve(L_tot_gpu, psi0_dressed_gpu, t)


# Save the Simulation Data
timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
filename_data = "Stabilised_Cat_$timestamp.jld2"
save_path_data = joinpath(save_dir, filename_data)
save_simulation_minimal(save_path_data, sol_gpu, V_mat, t)

# Define the save path for the video
video_filepath = joinpath(save_dir, "Cat_Evolution_$timestamp.mp4")
# Generate the video!
animate_wigner_evolution(
    video_filepath, 
    sol_gpu.states, 
    V_mat, 
    n1, 
    t, 
    w_rf, 
    dims_sys, 
    total_frames = 300, # Increase this if you want a smoother, longer video
    fps = 15            # Adjust playback speed
)



#%%
# 11. Post-processing: Transforming back to the bare basis
println("Post-processing results...")

t_selected = 4000
t_selected_idx = floor(Int, t_selected * steps / tmax)

# Final state in the dressed basis
rho_final_dressed_mat = Array(sol_gpu.states[t_selected_idx].data)
# Final state in the bare basis
rho_final_bare_mat = V_mat * rho_final_dressed_mat * V_mat'
# Convert to QuantumObject for easier manipulation
rho_final_bare_qobj = QuantumObject(rho_final_bare_mat, type=Operator(), dims=dims_sys)

# 12. Rotating frame transformation to remove fast oscillations
w_rf = wd/2
t_final = t[t_selected_idx]

U_rot = exp(1im * w_rf * t_final * Array(n1.data))

rho_final_bare_mat_rotated = U_rot * rho_final_bare_mat * U_rot'

rho_final_bare_rotated_qobj = QuantumObject(rho_final_bare_mat_rotated, type=Operator(), dims=dims_sys)

# 13. Partial Trace to get the reduced state of mode a1
rho_mode1 = ptrace(rho_final_bare_qobj, 1)
rho_mode1_rotated = ptrace(rho_final_bare_rotated_qobj, 1)

# 14. Wigner function calculation for mode a1
xvec = LinRange(-5, 5, 100)
yvec = LinRange(-5, 5, 100)
W_cat = wigner(rho_mode1_rotated, xvec, yvec)'

#%%

# 15. Calculations for Summary 
g_eff = gap_simulated/(2*sqrt(2))
alpha_analytical = sqrt(Complex(-F/g_eff))

cat_plus = coherent(Na1, alpha_analytical) + coherent(Na1, -alpha_analytical)
cat_plus_analytical = normalize(cat_plus)
rho_plus_analytical = ket2dm(cat_plus_analytical)

fid_plus = real(fidelity(rho_mode1, rho_plus_analytical))
fid_plus_rotated = real(fidelity(rho_mode1_rotated, rho_plus_analytical))

exp_a_bare = expect(a1, rho_final_bare_qobj)
exp_n_bare = expect(n1, rho_final_bare_qobj)
exp_a_rot  = expect(a1, rho_final_bare_rotated_qobj)
exp_n_rot  = expect(n1, rho_final_bare_rotated_qobj)

# Print to console just like before
println("Bare a expectation: ", exp_a_bare)
println("Bare n1 expectation: ", exp_n_bare)
println("Bare Rotating a expectation: ", exp_a_rot)
println("Bare Rotating n1 expectation: ", exp_n_rot)
n1_analtytical = tensor(num(Na1))
println("Analytical n1 expectation: ", expect(n1_analtytical, rho_plus_analytical))
println("Analytical amplitude α: ", round(alpha_analytical, digits=3))
println("Fidelity Bare: ", round(fid_plus, digits=5))
println("Fidelity Bare Rotating: ", round(fid_plus_rotated, digits=5))

# Time-resolved Photon Tracking
n1_bare_mat = Array(n1.data)
n2_bare_mat = Array(n2.data)
expect_n1 = zeros(length(t))
expect_n2 = zeros(length(t))

for (i, rho_dressed_qobj) in enumerate(sol_gpu.states)
    rho_dressed = Array(rho_dressed_qobj.data)
    rho_bare = V_mat * rho_dressed * V_mat'
    expect_n1[i] = real(tr(rho_bare * n1_bare_mat))
    expect_n2[i] = real(tr(rho_bare * n2_bare_mat))
end


# 16. Define Custom Directory and Dynamic Filename

# Generate a unique name using the current time
timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
filename_img = "Stabilised_Cat_$timestamp.png"
filename_txt = "Stabilised_Cat_$timestamp.txt"

save_path_img = joinpath(save_dir, filename_img)
save_path_txt = joinpath(save_dir, filename_txt)



# 17. Comprehensive Plotting (Text removed for a clean visual layout)
fig_master = Figure(size = (1200, 1000))

# Panel 1: Population Dynamics (Top row, spans both columns)
ax_pop = Axis(fig_master[1, 1:2], 
              title="Photon Transfer Dynamics", 
              xlabel="Time", 
              ylabel="Average Photon Number")
lines!(ax_pop, t, expect_n2, label="<n2> (Driven Cavity)", linewidth=3, color=:orange)
lines!(ax_pop, t, expect_n1, label="<n1> (Target Cavity)", linewidth=3, color=:blue)
axislegend(ax_pop, position=:lt)

# Panel 2: 2D Wigner function (Bottom row, left)
ax2D = Axis(fig_master[2, 1], 
            title = "2D Wigner Function (Mode a1) (t = $(round(t_selected, digits=2)))",
            xlabel = "Re(α)", 
            ylabel = "Im(α)",
            aspect = 1) 
hm = CairoMakie.heatmap!(ax2D, xvec, yvec, W_cat, colormap = :RdBu)
Colorbar(fig_master[2, 1, Right()], hm, label = "W(α)") # Attaches colorbar directly to the 2D plot

# Panel 3: 3D Wigner function (Bottom row, right)
ax3D = Axis3(fig_master[2, 2], 
             title = "3D Wigner Function (Mode a1) (t = $(round(t_selected, digits=2)))",
             xlabel = "Re(α)", 
             ylabel = "Im(α)", 
             zlabel = "W(α)",
             elevation = pi/6, 
             azimuth = pi/4) 
CairoMakie.surface!(ax3D, xvec, yvec, W_cat, colormap = :RdBu)
CairoMakie.colgap!(fig_master.layout, 150)

display(fig_master)

# Save the final image
CairoMakie.save(save_path_img, fig_master, px_per_unit = 2) 


# 18. Save the matching Text File
summary_text = """
--- System Parameters ---
ω_a1 = $(wa1)
ω_a2 = $(wa2)
ω_q  = $(wq)
g_a1 = $(ga1)
g_a2 = $(ga2)
θ    = $(round(theta, digits=3))
κ_a2 = $(ka2)
ω_d  = $(round(wd, digits=5))
F    = $(F)
N_a1 = $(Na1)
N_a2 = $(Na2)
N_q  = $(Nq)


Observables:
Bare ⟨a1⟩ = $(round(exp_a_bare, digits=4))
Bare ⟨n1⟩ = $(round(exp_n_bare, digits=4))
Rot. ⟨a1⟩ = $(round(exp_a_rot, digits=4))
Rot. ⟨n1⟩ = $(round(exp_n_rot, digits=4))

--- Fidelities ---

Analytical α = $(round(alpha_analytical, digits=3))
Bare Fid.    = $(round(fid_plus, digits=5))
Rot. Fid.    = $(round(fid_plus_rotated, digits=5))
"""

open(save_path_txt, "w") do file
    write(file, summary_text)
end

println("Saved image to: ", save_path_img)
println("Saved logs to:  ", save_path_txt)






#%%

#saved_file = "C:\\Users\\andre\\Desktop\\Università\\Magistrale\\MA4\\Thesis\\Code\\Cats\\Stabilised_Cat_2026-03-12_083412.jld2"
#sim_data = load_simulation_minimal(saved_file)

# Access your variables directly
#t_load = sim_data.t
#V_mat_load = sim_data.V_mat
#expect_n1_load = sim_data.expect_n1

# Example: Get the final state (already safely wrapped in a QuantumObject)
#rho_final_dressed_qobj_load = sim_data.states[end]