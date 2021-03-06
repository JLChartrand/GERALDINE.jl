mutable struct AGState <: AbstractState
    x::Vector
    x_ag::Vector
    x_md::Vector
    ∇f_md::Vector
    fx_md::Float64
    it::Int64
    function AGState(x0::Vector)
        n = new()
        n.x = copy(x0)
        n.x_ag = copy(x0)
        n.x_md = copy(x0)
        n.∇f_md = zeros(length(x0))
        n.fx_md = 0.0
        n.it = 0
        return n
    end
end

function Stop_optimize_weak_AG(st::AbstractState; tol::Float64 = 1e-5, nmax::Int64 = 500)
    return Stop_optimize_weak(norm(st.∇f_md), st.it; tol = tol, nmax = nmax)
end

function println(st::AGState)
    println("it = $(st.it)")
    println("x = $(round.(st.x, digits = 3))")
    println("x_ag = $(round.(st.x_ag, digits = 3))")
    println("x_md = $(round.(st.x_md, digits = 3))")
    println("grad = $(st.∇f_md)")
end
