function solve_BTR_BFGS(m::Models; verbose::Bool = false, nmax::Int64 = 1000)
    return OPTIM_btr_BFGS(m.f, m.∇f!, zeros(m.dim), verbose = verbose, nmax = nmax)
end

function solve_BTR_TH(m::Models; verbose::Bool = false, nmax::Int64 = 1000)
    return OPTIM_btr_TH(m.f, m.∇f!, m.Hf!,  zeros(m.dim), verbose = verbose, nmax = nmax)
end

function solve_RSAG(m::Models; verbose::Bool = false, nmax::Int64 = 1000)
    return OPTIM_AGRESSIVE_RSAG(lm.f, lm.∇f!, lm.batch, lm.f::Function; x0 = zeros(m.dim), L = 1.0, nmax = 500, 
        ϵ = 1e-4, verbose = false, n_test = 500, n_optim = 100)
end

function solve_AG(m::Models; verbose::Bool = false, nmax::Int64 = 1000)
    return OPTIM_AGRESSIVE_AG(m.f, m.∇f!, x0 = zeros(m.dim), L = 1.0, nmax = nmax, 
        ϵ = 1e-4, verbose = verbose)
end

function solve_BFGS(m::Models; verbose::Bool = false, nmax::Int64 = 1000)
    return OPTIM_BFGS(m.f, m.∇f!, x0 = zeros(m.dim), nmax = nmax, 
        ϵ = 1e-4, verbose = verbose)
end

"""
chose between :

  'solve_BTR_BFGS'

  'solve_BTR_TH'

  'solve_RSAG'
  
  'solve_AG'
  
  'solve_BFGS'
"""
solve = solve_BTR_BFGS