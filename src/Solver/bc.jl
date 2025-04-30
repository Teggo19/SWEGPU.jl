include("FVM.jl")

function neumannBC(h, hu, hv, F, compute_eigenvalues_F)
    # Neumann boundary condition
    f1, f2_rot, f3_rot, lambda = central_upwind_flux_kurganov(h, hu, hv, h, hu, hv, F, compute_eigenvalues_F)
    return f1, f2_rot, f3_rot, lambda
end

function wallBC(h, hu, hv, F, compute_eigenvalues_F)
    # Wall boundary condition
    f1, f2_rot, f3_rot, lambda = central_upwind_flux_kurganov(h, hu, hv, h, -hu, hv, F, compute_eigenvalues_F)
    return f1, f2_rot, f3_rot, lambda
end