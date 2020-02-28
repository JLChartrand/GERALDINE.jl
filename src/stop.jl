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

function Stop_optimize_weak(value::Float64, grad::Vector, k::Int64; x::Vector = ones(length(grad)),
        typVal::Float64 = 1.0, typX::Vector = ones(length(grad)), tol::Float64 = 1e-4, nmax::Int64 = 500)
    if k >= nmax
        return true
    end
    if norm(grad) < tol
        return true
    end
    return false
end