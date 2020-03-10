function OPTIM_btr_TH(f::Function, g!::Function, H!::Function, 
                x0::Vector; verbose::Bool = true, 
                nmax::Int64 = 1000, epsilon::Float64 = 1e-4)
        
        
    function accumulate!(state::BTRState{Matrix}, acc::Vector)
        push!(acc, state.fx)
    end
    accumulator = []
        
    H = Array{Float64, 2}(I, length(x0), length(x0))
    state = BTRState(H)
    state.x = copy(x0)
    state.it = 0
    state.grad = zeros(length(x0))
        
    
    state, accumulator = btr(f, g!, H!, state, x0, 
                verbose = verbose, nmax = nmax, epsilon = epsilon, accumulate! = accumulate!, accumulator = accumulator)
    return state.x, state, accumulator
end
