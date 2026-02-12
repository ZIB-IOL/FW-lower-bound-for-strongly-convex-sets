using Printf

include("utils.jl")

precision = 1000
setprecision(BigFloat, precision)

# ==============================================================================
# 1. THEORETICAL BACKWARD DYNAMICS (Robust Version)
# ==============================================================================

function inverse_step_minus(r_next, s_next)
    # Using BigFloat is critical here because we are inverting a chaotic map
    u = BigFloat(r_next)
    v = BigFloat(s_next)

    v2 = v^2

    X = (1+u)*v2 - u
    Y = sqrt((1-v2)*(1-(1+u)^2*v2))
    if X + Y <= 0
        return nothing
    end
    r_prev = u / (X + Y)
    return r_prev
end

function generate_start_point_safe(epsilon_target)
    println("-"^60)
    println("PHASE 1: Backward Pass")
    
    # 1. Start at 'infinity' (very close to convergence)
    #    The paper gives the asymptotic curve: s ≈ 1 - 4/3 * r + 2*r^2
    r = BigFloat(epsilon_target)
    s = 1 - 4/3 * r + 2*r^2
    
    println("  Starting trace from r ≈ $(Float64(r))")
    
    # 2. Trace backwards carefully
    #    We stop IF:
    #    a) The next step is invalid (discriminant < 0)
    #    b) The radius gets 'unsafe' (e.g. > 1.5). The max residual is 2.0.
    #       As you noted, steps get huge near the boundary, so we stop early
    #       to emulate starting at 'x_5' rather than 'x_0'.
    
    step_count = 0
    safe_r = r
    safe_s = s

    while true
        r_prev = inverse_step_minus(r, s)
        
        # STOPPING CONDITION 1: Mathematics breaks (overshoot boundary)
        if isnothing(r_prev)
            println("  Next step would overshoot boundary. Stopping at current r.")
            break
        end        
        
        # Calculate s_{t-1} = r_t / r_{t-1}
        # (Definition: r_{t} = r_{t-1} * s_{t-1})
        s_prev = r / r_prev 
        
        # Update
        r = r_prev
        s = s_prev
        
        safe_r = r
        safe_s = s
        step_count += 1

        # STOPPING CONDITION 2: "Like x_5" - Stop before the singularity
        # r = 1.5 is a safe "early iterate" residual (boundary is 2.0)
        if r_prev > 0.5
            println("  Residual getting too close to singularity (r > 0.5). Stopping.")
            break
        end
        
        # Progress reporting every 100 steps
        if step_count % 100 == 0
            @printf("  Step %d: r = %.6e\n", step_count, Float64(r))
        end
    end
    
    println("  Backward trace stopped after $(step_count) steps.")
    println("  Selected Start Residual r_start: $(Float64(safe_r))")
    println("-"^60)
    
    return safe_r, safe_s, step_count
end

# ==============================================================================
# 2. EXECUTION & PLOTTING
# ==============================================================================

# Start from deep convergence
epsilon_start = 1/precision

# Find the "Hard" starting point (stopping before boundary)
r0, s0, step_count = generate_start_point_safe(epsilon_start)

# Run Forward verification with Frank-Wolfe
p_opt = [0.0, 1.0]
path, gaps, x_start = run_frank_wolfe_experiment(r0, s0, p_opt, min(precision, step_count))

# --- Plotting ---

# Trajectory Plot
plt_traj = plot_trajectory_2d(path, p_opt, title="Trajectory of the worst-case instance \n (Backward Reconstruction)")
display(plt_traj)
println("Displayed trajectory plot.")

# Convergence Plot
plt_conv = plot_convergence(gaps, title="Convergence Rate (Backward Reconstruction)")
display(plt_conv)
println("Displayed convergence plot.")
