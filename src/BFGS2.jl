abstract type Approx end

mutable struct BFGS <: Approx
    H::Matrix
    inv::Matrix
    First::Bool
    function BFGS() #m is a dumy parameter
        n = new()
        n.First = true
        return new()
    end
end

function println(bfgs::BFGS)
    println(H)
end

function update!(b::BFGS, y::Vector, s::Vector)
    if b.First
        b.First = false
        n = length(y)
        b.H = Array{Float64, 2}(I, n, n)
        b.inv = Array{Float64, 2}(I, n, n)
    end
    Bs = b.H*s
    b.H = b.H-(Bs*Bs')/dot(s, Bs)+(y*y')/dot(s, y)
    
    term_1 = Array{Float64, 2}(I, b.dim, b.dim)- (s*y')/(y'*s)
    term_2 = Array{Float64, 2}(I, b.dim, b.dim) - (y*s')/(y'*s)
    b.inv = term_1*b.inv*term_2+(s*s')/(y'*s)
end

function direction(grad::Vector, x::Vector, bfgs::BFGS)
    return -bfgs.inv*grad
end


function Arimijo(f::Function, x::Vector, d::Vector, grad::Vector, β::Float64 = 0.1)
    α = 1.0
    κ = 0.7
    while f(x + α*d) > f(x) + α*β*grad'*d
        α*=κ
    end
    return α
end

function optimize(f::Function, ∇f!::Function, x_0::Vector;
        nmax::Int64 = 500, epsilon::Float64 = 1e-4, Bfgs::BFGS = BFGS(), verbose::Bool = false)
    grad = zeros(length(x_0))
    x = copy(x_0)
    ∇f!(x, grad)
    bfgs = Bfgs(length(x), m)
    α_k = Arimijo(f, x, -grad, grad)
    
    s_k = -α_k*grad
    x += s_k
    new_grad = zeros(length(x_0))
    ∇f!(x, new_grad)
    y_k = new_grad - grad
    
    update!(bfgs, y_k, s_k)
    
    grad[:] = new_grad
    it = 1
    while !Stop_optimize(f(x), grad, it, tol = epsilon)
        if verbose
            println(x)
        end
        p_k = direction(grad, x, bfgs)
        α_k = Arimijo(f, x, p_k, grad)
        
        s_k = α_k*p_k
        x += s_k
        
        ∇f!(x, new_grad)
        update!(bfgs, new_grad - grad, s_k)
        
        grad[:] = new_grad
        it += 1
    end
    return x, bfgs, it
    
end


function OPTIM_BFGS(f::Function, ∇f!::Function, x0::Vector; nmax::Int64 = 500, 
        epsilon::Float64 = 1e-4, verbose::Bool = false)
    
    return optimize(f, ∇f!, x0, nmax = nmax, verbose = verbose, epsilon = epsilon)
    
end
