struct BasicTrustRegion{T<:Real}
    η1::T
    η2::T
    γ1::T
    γ2::T
end

function BTRDefaults()
    return BasicTrustRegion(0.01, 0.9, 0.5, 0.5)
end

mutable struct BTRState{T} <: AbstractState where T
    it::Int64
    x::Vector
    xcand::Vector
    grad::Vector
    H::T
    step::Vector
    Δ::Float64
    ρ::Float64
    fx::Float64
    function BTRState(H::T) where T
        state = new{T}()
        state.H = H
        return state
    end
end


function println(state::BTRState)
    println(round.(state.x, digits = 3))
end
function acceptCandidate!(state::BTRState, b::BasicTrustRegion)
    if state.ρ >= b.η1
        return true
    else
        return false
    end
end

function updateRadius!(state::BTRState, b::BasicTrustRegion)
    if state.ρ >= b.η2
        stepnorm = norm(state.step)
        state.Δ = min(10e12, max(4*stepnorm, state.Δ))
    elseif state.ρ >= b.η1
        state.Δ *= b.γ2
    else
        state.Δ *= b.γ1
    end
end

function stopCG(normg::Float64, normg0::Float64, k::Int, kmax::Int)
    χ::Float64 = 0.1
    θ::Float64 = 0.5
    if (k == kmax) || (normg <= normg0*min(χ, normg0^θ))
        return true
    else
        return false
    end
end



function TruncatedCG(state::BTRState)
    H = state.H
    g = state.grad
    Δ = state.Δ*state.Δ
    n = length(g)
    s = zeros(n)
    normg0 = norm(g)
    v = g
    d = -v
    gv = dot(g, v)
    norm2d = gv
    norm2s = 0
    sMd = 0
    k = 0
    while ! stopCG(norm(g), normg0, k, n)
        Hd = H*d
        κ = dot(d, Hd)
        if κ <= 0
            σ = (-sMd+sqrt(sMd*sMd+norm2d*(Δ-dot(s, s))))/norm2d
            s += σ*d
            break
        end
        α = gv/κ
        norm2s += α*(2*sMd+α*norm2d)
        if norm2s >= Δ
            σ = (-sMd+sqrt(sMd*sMd+norm2d*(Δ-dot(s, s))))/norm2d
            s += σ*d
            break
        end
        s += α*d
        g += α*Hd
        v = g
        newgv = dot(g, v)
        β = newgv/gv
        gv = newgv
        d = -v+β*d
        sMd = β*(sMd+α*norm2d)
        norm2d = gv+β*β*norm2d
        k += 1
    end
    return s
end

function btr(f::Function, g!::Function, H!::Function, state::BTRState{Array{Float64,2}}, x0::Vector; 
        verbose::Bool = true, nmax::Int64 = 1000, epsilon::Float64 = 1e-6, 
        accumulate!::Function, accumulator::Array)
    b = BTRDefaults()
    state.fx = f(x0)
    g!(x0, state.grad)
    state.Δ = 0.1*norm(state.grad)
    H!(x0, state.H)
    

    function model(s::Vector, g::Vector, H::Matrix)
        return dot(s, g)+0.5*dot(s, H*s)
    end
    
    while (!stop(state, nmax = nmax, tol = epsilon))
        accumulate!(state, accumulator)
        if verbose
            println(state)
        end
        
        state.step = TruncatedCG(state, H)
        state.xcand = state.x+state.step
        fcand = f(state.xcand)
        state.ρ = (fcand-state.fx)/(dot(state.s, state.grad)+0.5*dot(state.s, state.H*state.s))
        if acceptCandidate!(state, b)
            state.x = copy(state.xcand)
            g!(state.x, state.grad)
            H!(state.x, state.H)
            
            state.fx = fcand
        end
        updateRadius!(state, b)
        state.it += 1
    end
    return state, accumulator
end
