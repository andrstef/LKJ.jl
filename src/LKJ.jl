module LKJ

"""
A Julia package implementing the Lewandowski, Kurowicka and Joe (LKJ) probability distribution on the space of correlation matrices.
The probability density function of a d-dimensional LKJ-distribution (''d \\geq 2'') with parameter ''\\boldsymbol{\\eta} \\geq 1'' is:
```math
f(\\mathbf{A}; \\boldsymbol{\\eta}) = c_d |det(A)|^{\\eta-1}
```
where ''\\mathbf{A}'' is ''d \\times d'' correlation matrix and ''c_d'' is a normalization constant depending on the dimension 'd'.
"""
LKJ

using Distributions
using SpecialFunctions: lgamma
using LinearAlgebra
using ArgCheck

import Base: length, size
import Distributions: rand, logpdf, params

export LKJcorr, LKJcorrChol
export length, size, params, rand, logpdf, transform_free_to_chol, transform_chol_to_free, log_jacobian_det_free_to_chol

## LKJ distribution for correlation matrices
struct LKJcorr <: ContinuousMatrixDistribution
    d::Int
    eta::Float64

    function LKJcorr(d::Int, eta::Float64)
        @argcheck (d >= 2) & (eta >= 1)
        new(d, eta)
    end
end

## LKJ distribution for cholesky factors of correlation matrices
struct LKJcorrChol <: ContinuousMatrixDistribution
    d::Int
    eta::Float64

    function LKJcorrChol(d::Int, eta::Float64)
        @argcheck (d >= 2) & (eta >= 1)
        new(d, eta)
    end
end

params(distr::Union{LKJcorr, LKJcorrChol}) = (distr.d, distr.eta)
length(distr::Union{LKJcorr, LKJcorrChol}) = distr.d
size(distr::Union{LKJcorr, LKJcorrChol}) = (distr.d, distr.d)

function rand(distr::Union{LKJcorr, LKJcorrChol}, n::Int)
    res=[rand(distr)]
    for i in 2:n
        push!(res,rand(distr))
    end
    res
end

rand(distr::LKJcorr) = _rand(distr.d, distr.eta)
rand(distr::LKJcorrChol) = cholesky!(_rand(distr.d, distr.eta)).L

# function _rand(d::Int, eta::Float64)
#     beta = eta + (d-2)/2
#     u = rand(Beta(beta, beta), 1)[1]
#
#     stdn = Normal(0, 1)
#
#     R = ones(Float64, d, d)
#     R[1,2] = 2*u - 1
#     R[2,1] = R[1,2]
#
#     for k in 2:(d-1)
#         beta -= 0.5
#
#         ## sample point uniformly from (k-1)-sphere
#         u = rand(stdn, k)
#         u ./= sqrt(sum(u.^2))
#
#         w = sqrt(rand(Beta(k/2, beta), 1)[1]) .* u
#         z = cholesky(R[1:k, 1:k]).U' * w
#
#         R[1:k, k+1] = z
#         R[k+1, 1:k] = z'
#     end
#     R
# end

function _rand(d::Int, eta::Float64)
    beta = eta + (d-2)/2
    u = rand(Beta(beta, beta))

    stdn = Normal(0, 1)

    R = ones(Float64, d, d)
    R[1,2] = 2*u - 1
    R[2,1] = R[1,2]

    for k in 2:(d-1)
        beta -= 0.5

        ## sample point uniformly from (k-1)-sphere
        u = rand(stdn, k)
        u ./= sqrt(sum(u.^2))

        w = sqrt(rand(Beta(k/2, beta))) .* u
        z = cholesky(@view R[1:k, 1:k]).L * w

        R[1:k, k+1] .= z
        R[k+1, 1:k] .= z
    end
    Hermitian(R)
end

function logpdf(distr::LKJcorr, x::AbstractMatrix{T}; norm=false) where T
    n, m = size(x)

    if n != distr.d
        throw(ArgumentError("Number of rows/columns ($n) does not match dimension of distribution ($(distr.d))"))
    end

    c = norm ? log_normalizing_const(distr.d, distr.eta) : 0

    (distr.eta - 1)*log(det(x)) + c
end

function logpdf(distr::LKJcorrChol, x::LowerTriangular; norm=false)
    n, m = size(x)

    if n != m
        throw(ArgumentError("x is not square"))
    end

    if n != distr.d
        throw(ArgumentError("Number of rows/columns ($n) does not match dimension of distribution ($(distr.d))"))
    end

    c = norm ? log_normalizing_const(distr.d, distr.eta) : 0

    sum((distr.d .- (1:distr.d) .+ 2*distr.eta .- 2).*log.(diag(x))) + c
end

function log_normalizing_const(d::Int, eta::T) where T <: Real
    c = zero(T)

    if eta == one(T)
        for k in 1:floor(Int, (d-1)/2)
            c += lgamma(2*k)
        end

        if iseven(d)
            c += d*(d-2)/4 * log(pi) + (3*d^2 - 4*d) / 4*log(2) + d*lgamma(d/2) - (d-1)*lgamma(d)
        else
            c += (d^2 - 1)/4 * log(pi) - (d-1)^2 / 4*log(2) - (d-1)*lgamma((d+1)/2)
        end
    else
        c = (2*eta + d - 3)*log(2) + 2*lgamma(eta + d/2 - 1) - lgamma(2*eta + d - 2) - (d-2)*lgamma(eta + (d-1)/2)

        for k in 2:(d-1)
            c += k/2 * log(pi) + lgamma(eta + (d-1-k)/2)
        end
    end
    c
end

## Stan Manual (version 2.17) Section 35.11. Cholesky Factors of Correlation Matrices
## see https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/cholesky_corr_constrain.hpp
## y = vector of unconstrained values
function transform_free_to_chol(y::AbstractVector{T}, K) where T
    z = tanh.(y)
    w = zeros(T, K, K)
    w[1,1] = 1.0

    k = 1
    for i in 2:K
        w[i,1] = z[k]
        k += 1
        sum_sqs = w[i,1]^2
        for j in 2:(i-1)
            if sum_sqs > 1.0
                w[i,j] = 0.0
            else
                w[i,j] = z[k] * sqrt(1.0 - sum_sqs)
            end
            k += 1
            sum_sqs += w[i,j]^2
        end

        if sum_sqs > 1.0
            w[i,i] = 0.0
        else
            w[i,i] = sqrt(1.0 - sum_sqs)
        end
    end

    LowerTriangular(w)
end

## see https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/cholesky_corr_free.hpp
## w = Cholesky factor of correlation matrix
function transform_chol_to_free(w::LowerTriangular)
    K = size(w,1)
    y = zeros(eltype(w), Int(K*(K-1)/2))

    k = 1
    for i in 2:K
        y[k] = atanh(w[i,1])
        k += 1
        sum_sqs = w[i,1]^2

        for j in 2:(i-1)
            y[k] = atanh(w[i,j] / sqrt(1.0 - sum_sqs))
            k += 1
            sum_sqs += w[i,j]^2
        end
    end
    y
end

## K = (1 + sqrt(1 + 8*length(y))) / 2

"""
    log_jacobian_det_free_to_chol(y, K)

Log absolute determinant of Jacobian of transformation from unconstrained vector (length K*(K-1)/2) to cholesky factor.

# Examples
```jldoctest
julia> log_jacobian_det_free_to_chol([1.0, 2.0, 3.0], 3)
-9.461226912195217
```
"""
function log_jacobian_det_free_to_chol(y, K)
    w = transform_free_to_chol(y, K)
    log_jacobian_det_free_to_chol(y, K, w)
end

function log_jacobian_det_free_to_chol(y, K, w::LowerTriangular)
    res = -2*sum(log.(cosh.(y)))

    for i in 2:K
        sum_sq = 0.0
        for j in 1:(i-1)
            ##res += 0.5*log(1 - sum(w[i,1:(j-1)].^2))
            res += 0.5*log(1 - sum_sq)
            sum_sq += w[i,j]^2
        end
    end
    res
end

end
