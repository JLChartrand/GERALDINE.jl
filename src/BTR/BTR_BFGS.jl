mutable struct BFGS_Matrix <: AbstractMatrix{Float64}
    H::Matrix
end
function BFGS!(bfgs::BFGS_Matrix, y::Vector, s::Vector)
    Bs = bfgs.H*s
    bfgs.H[:, :] += (y*y')/(y'*s) - (Bs*Bs')/(s'*Bs)
end



import Base.size
function size(a::BFGS_Matrix)
    return size(a.H)
end
import Base.getindex
function getindex(a::BFGS_Matrix, index...)
    return getindex(a.H, index...)
end
import Base.setindex!
function setindex!(a::BFGS_Matrix, value, index...)
    setindex!(a.H, value, index...)
end
function OPTIM_btr_BFGS(f::Function, g!::Function, x0::Vector; verbose::Bool = true, nmax::Int64 = 1000, epsilon::Float64)
    H = Array{Float64, 2}(I, length(x0), length(x0))
    
    function accumulate!(state::BTRState{BFGS_Matrix}, acc::Vector)
        push!(acc, state.fx)
    end
    accumulator = []
    state = BTRState(BFGS_Matrix(H))
    state.x = x0
    state.it = 0
    state.grad = zeros(length(x0))
    state, accumulator = btr(f, g!, state, x0,
        verbose = verbose, nmax = nmax, epsilon = epsilon, accumulate! = accumulate!, accumulator = accumulator)
    return state, accumulator
end
