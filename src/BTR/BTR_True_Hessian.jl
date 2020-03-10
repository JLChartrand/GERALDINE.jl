function btr(f::Function, g!::Function, H!::Function, state::BTRState{Array{Float64,2}}, x0::Vector; 
        verbose::Bool = true, nmax::Int64 = 1000, epsilon::Float64 = 1e-6, 
        accumulate!::Function, accumulator::Array)
    b = BTRDefaults()
    state.fx = f(x0)
    g!(x0, state.grad)
    state.Δ = 0.1*norm(state.grad)
    H!(x0, state.H)
    
    
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
            g!(state.x, state.grad)
            H!(state.x, state.H)
            
            state.fx = fcand
        end
        updateRadius!(state, b)
        state.it += 1
    end
    return state, accumulator
end


function OPTIM_btr_TH(f::Function, g!::Function, H!::Function, 
                x0::Vector; verbose::Bool = true, 
                nmax::Int64 = 1000, epsilon::Float64 = 1e-4, 
                accumulate! = (st, acc) -> nothing, accumulator = [])
        
    
        
    H = Array{Float64, 2}(I, length(x0), length(x0))
    state = BTRState(H)
    state.x = copy(x0)
    state.it = 0
    state.grad = zeros(length(x0))
        
    
    state, accumulator = btr(f, g!, H!, state, x0, 
                verbose = verbose, nmax = nmax, epsilon = epsilon, accumulate! = accumulate!, accumulator = accumulator)
    return state.x, state, accumulator
end
