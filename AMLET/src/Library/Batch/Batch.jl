
"""
'abstract type Batch end'
"""
abstract type Batch end

"""
'abstract type BatchLM end'
# Methods:
### iterate(::BatchLM)
### iterate(::BatchLM, state)

both iterate methods have to return Union{( ind <: Individual{T}, state::Any), nothing}
"""
abstract type BatchLM <: Batch end
