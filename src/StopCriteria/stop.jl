"""
'Stop_optimize_robuste(value::Float64, grad::Vector, k::Int64; x::Vector = ones(length(grad)),
        typVal::Float64 = 1.0, typX::Vector = ones(length(grad)), tol::Float64 = 1e-4, nmax::Int64 = 500)'
robust stoping criteria
"""

function Stop_optimize(value::Float64, grad::Vector, k::Int64; x::Vector = ones(length(grad)),
        typVal::Float64 = 1.0, typX::Vector = ones(length(grad)), tol::Float64 = 1e-4, nmax::Int64 = 500)
    if k >= nmax
        return true
    end
    for i in 1:length(x)
        if abs(grad[i]*max(x[i], typX[i])/max(value, typVal)) > tol
            return true
        end
    end
    return false
end

"""
'Stop_optimize_weak(value::Float64, grad::Vector, k::Int64; x::Vector = ones(length(grad)),
        typVal::Float64 = 1.0, typX::Vector = ones(length(grad)), tol::Float64 = 1e-4, nmax::Int64 = 500)'
robust stoping criteria
"""

function Stop_optimize_weak(nrm::Float64, k::Int64; tol::Float64 = 1e-5, nmax::Int64 = 500)
    if k<1
        return false
    end
    if k >= nmax
        return true
    end
    if nrm < tol
        return true
    end
    return false
end

function Stop_optimize_weak(st::AbstractState; tol::Float64 = 1e-5, nmax::Int64 = 500)
    return Stop_optimize_weak(norm(st.grad), st.it; tol = tol, nmax = nmax)
end

stop = Stop_optimize_weak
