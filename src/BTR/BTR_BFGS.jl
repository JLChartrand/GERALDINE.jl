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


function btr_bfgs(f::Function, g!::Function, state::BTRState{Array{Float64,2}}, x0::Vector; 
        verbose::Bool = true, nmax::Int64 = 1000, epsilon::Float64 = 1e-6, 
        accumulate!::Function, accumulator::Array)
    b = BTRDefaults()
    state.fx = f(x0)
    g!(x0, state.grad)
    state.Δ = 0.1*norm(state.grad)
    H!(x0, state.H)
    
    oldGrad = copy(state.grad)
    y = zeros(length(x0))
    while (!stop(state, nmax = nmax, tol = epsilon))
        accumulate!(state, accumulator)
        if verbose
            println(state)
        end
        
        state.step = TruncatedCG(state)
        state.xcand = state.x+state.step
        fcand = f(state.xcand)
        state.ρ = (fcand-state.fx)/(dot(state.step, state.grad)+0.5*dot(state.step, state.H*state.step))
        if acceptCandidate!(state, b)
            state.x = copy(state.xcand)
            oldGrad = copy(state.grad)
            g!(state.x, state.grad)
            y = state.grad - oldGrad
            BFGS!(state.H, y, state.step)
            state.fx = fcand
        end
        
        updateRadius!(state, b)
        state.it += 1
    end
    return state, accumulator
end

function OPTIM_btr_BFGS(f::Function, g!::Function, x0::Vector; verbose::Bool = true, nmax::Int64 = 1000, epsilon::Float64)
    H = Array{Float64, 2}(I, length(x0), length(x0))
    state = BTRState(BFGS_Matrix(H))
    state.x = x0
    state.it = 0
    state.grad = zeros(length(x0))
    state, accumulator = btr(f, g!, state, x0,
        verbose = verbose, nmax = nmax, epsilon = epsilon, accumulate! = (st, acc) -> nothing, accumulator = [])
    return state.x, state, accumulator
end
