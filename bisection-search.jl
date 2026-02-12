using Printf

include("utils.jl")

precision = 1000
setprecision(BigFloat, precision)

# ==============================================================================
# 1. BISECTION SEARCH ALGORITHMS
# ==============================================================================

# Count the number of iterations for which the contraction rates are monotone 
# increasing, i.e. s_0 < s_1 < ... < s_n
function count_epoch(r0, s0; eps=1e-10)
    if s0 > 1/(1+r0)
        return -10
    end
    counter = 0
    r = r0
    s = s0
    while r > eps
        s_new = sqrt((1 - (r+1)^2 * s^2)/(2 - 2*s - (2+r)*r*s^2))
        if s_new < s
            return counter
        end
        r = r * s
        s = s_new
        counter += 1
    end
    return counter
end

# Parity-based bisection search (faster, exploits parity structure)
# As described in Appendix Section "Bisection search":
# - Starting points on the left vs right of the maximum have different parity
# - This allows efficient binary search to find the maximum
function parity_bisection_search(precision; eps=1e-10, max_steps=1000)
    println("-"^60)
    println("BISECTION SEARCH: Finding worst startpoint")
    
    setprecision(BigFloat, precision)
    eps_bisect = 1.0/10^(precision/10*3)

    # Initialize bounds for an interval containing a single peak
    # Based on the paper, [0.4, 0.5] isolates a single peak for r_0 = 1
    l = BigFloat("0.4")
    u = BigFloat("0.5")

    s_max = BigFloat("0.0")
    max_count = 0
    steps = 0

    # Compute initial parity at bounds
    count_u = count_epoch(BigFloat("1.0"), u, eps=eps)
    count_l = count_epoch(BigFloat("1.0"), l, eps=eps)
    parity_l = count_l % 2
    parity_u = count_u % 2

    while u - l > eps_bisect && steps < max_steps
        steps += 1

        # Compute midpoint
        m = (l + u) / 2

        # Compute stable phase length at midpoint
        count_m = count_epoch(BigFloat("1.0"), m, eps=eps)
        parity_m = count_m % 2

        @assert parity_m >= min(parity_l, parity_u)

        if steps % 100 == 0
            @printf("  Step %d: max_count=%d, parity_m=%d\n", steps, max(count_m, count_l, count_u), parity_m)
        end

        # Update bounds based on parity comparison
        # If m and u have the same parity, the maximum is in [l, m]
        # Otherwise, the maximum is in [m, u]
        if parity_m == parity_u
            u = m
            count_u = count_m
            parity_u = parity_m
        else
            l = m
            count_l = count_m
            parity_l = parity_m
        end

        # Track the best found so far
        if count_m > max_count
            max_count = count_m
            s_max = m
        end
    end

    # Final check at the converged point
    final_s = (l + u) / 2
    final_count = count_epoch(BigFloat("1.0"), final_s, eps=eps)
    if final_count > max_count
        max_count = final_count
        s_max = final_s
    end

    println("  Search completed after $steps steps.")
    println("  Worst startpoint s_max: $(Float64(s_max))")
    println("  Maximum stable phase length: $max_count")
    println("-"^60)

    return s_max, max_count
end

# ==============================================================================
# 2. EXECUTION & PLOTTING
# ==============================================================================

# Run bisection search to find worst startpoint
s_max, max_count = parity_bisection_search(precision)

# --- Plotting ---

# Run Forward verification with Frank-Wolfe
p_opt = [0.0, 1.0]
r0 = 1.0  # Starting from r_0 = 1 (on the boundary)
path, gaps, x_start = run_frank_wolfe_experiment(r0, s_max, p_opt, min(max_count, precision))

# Trajectory Plot
plt_traj = plot_trajectory_2d(path, p_opt, title="Trajectory of the worst-case instance \n (Bisection Search)")
display(plt_traj)
println("Displayed trajectory plot.")

# Convergence Plot
plt_conv = plot_convergence(gaps, title="Convergence Rate (Bisection Search)")
display(plt_conv)
println("Displayed convergence plot.")
