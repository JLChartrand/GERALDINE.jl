"""

AMLET, An Other Mixed Logit Estimation Tool
"""

module AMLET

#The used Package
#using RDST, Statistics, LinearAlgebra, Distributions, Random, GERALDINE
using Statistics, LinearAlgebra


#function redefined in AMLET
import Base.iterate, Base.copy

export 
    # batch types
    Batch, BatchLM, BatchMLM, 
    
    #basic function
    iterate, copy, 

    #Individus Type
    Individual, LM_Individual, MLM_Individual,

    #Models type
    Models, LM, MLM,
    #function
    get_cov_matrix, CI,

    #Utilities type
    Utilities, LinearUtilities,

    #predefine utility
    LU, UVINLU,


    #the solve function 
    #solve,

    #The well known complete_Model! function
    complete_Model!,
    
    #others
    mean,
    
    #Solving gears
    solve, solve_BTR_BFGS, solve_BTR_TH, solve_RSAG, solve_AG, solve_BFGS

    


include("Library/main.jl")



end # module
