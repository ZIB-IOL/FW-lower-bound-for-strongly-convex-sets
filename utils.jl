using Plots
using LinearAlgebra
using FrankWolfe

# ==============================================================================
# FRANK-WOLFE VERIFICATION
# ==============================================================================

"""
    run_frank_wolfe_experiment(r_start, s_start, p_opt, iterations)

Run Frank-Wolfe algorithm starting from a reconstructed initial point.

Given the theoretical (r, s) coordinates, reconstruct the actual starting point x0
and verify the convergence behavior matches the theoretical predictions.

Returns: (path, gaps, x0) where path is the trajectory and gaps are the objective values.
"""
function run_frank_wolfe_experiment(r_start, s_start, p_opt, iterations)
    println("-"^60)
    println("FORWARD SIMULATION: Verifying with Frank-Wolfe")
    
    # 1. Reconstruction x0 from (r, s) coordinates
    theta0 = -s_start^2 * (r_start + 1) - sqrt((1 - s_start^2 * (r_start + 1)^2) * (1 - s_start^2))

    q = [1.0, 0.0]  # orthogonal to p_opt
    c0 = theta0 * p_opt + sqrt(1 - theta0^2) * q
    x0 = p_opt + r_start * c0

    println("  Starting point x0: $(Float64.(x0))")
    println("  Distance to optimum: $(Float64(norm(x0 - p_opt)))")
    
    # 2. Setup problem: min ||x - p_opt||^2 over unit ball
    f(x) = norm(x - p_opt)^2
    function grad!(storage, x)
        storage .= 2 * (x - p_opt)
    end

    lmo = FrankWolfe.LpNormBallLMO{2}(1.0)

    path = [x0]
    gaps = Float64[f(x0)]

    function path_callback(state, args...)
        push!(path, copy(state.x))
        push!(gaps, f(state.x))
    end

    # 3. Run Frank-Wolfe
    println("  Running $iterations iterations...")
    FrankWolfe.frank_wolfe(f, grad!, lmo, x0; 
        callback=path_callback, 
        max_iteration=iterations, 
        verbose=true, 
        print_iter=100
    )

    println("  Final gap: $(gaps[end])")
    println("-"^60)

    return path, gaps, x0
end

# ==============================================================================
# PLOTTING UTILITIES
# ==============================================================================

"""
    plot_trajectory_2d(path, p_opt; title="Trajectory of the worst-case instance")

Plot the Frank-Wolfe trajectory on the unit ball in 2D.
"""
function plot_trajectory_2d(path, p_opt; title="Trajectory of the worst-case instance")
    # Unit circle
    theta = range(0, 2pi, length=200)
    circle_x = cos.(theta)
    circle_y = sin.(theta)

    xs = [Float64(pt[1]) for pt in path]
    ys = [Float64(pt[2]) for pt in path]

    plt = plot(circle_x, circle_y, 
        aspect_ratio=:equal, 
        label="Unit Ball", 
        linecolor=:black,
        size=(600, 600),
        title=title,
        ticks=false,
        axis=false,
    )
    plot!(xs, ys, label="Trajectory", color=:blue, marker=:circle, markersize=3)
    scatter!([p_opt[1]], [p_opt[2]], label="Optimum p", color=:red, markersize=6)

    return plt
end

"""
    plot_convergence(gaps; title="Convergence Rate")

Plot the convergence of the Frank-Wolfe algorithm with O(1/t^2) reference.
"""
function plot_convergence(gaps; title="Convergence Rate")
    t_range = 1:length(gaps)
    
    plt = plot(t_range, gaps, 
        yaxis=:log, xaxis=:log,
        label="f(x_t)", 
        title=title, 
        xlabel="Iteration t", 
        ylabel="Gap",
        linewidth=2
    )
    
    # O(1/t^2) reference line
    ref = [gaps[1] * t^-2 for t in t_range]
    plot!(t_range, ref, linestyle=:dash, color=:red, label="O(1/t²) Reference")

    return plt
end
