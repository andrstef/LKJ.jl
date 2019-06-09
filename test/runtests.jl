#!/usr/bin/env julia

using LKJ
using Test

import Pkg
Pkg.add("ForwardDiff")

using ForwardDiff
using LinearAlgebra

is_corr_matrix(R) = issymmetric(R) & all(diag(R) .== 1.0) & all(R .<= 1.0) & all(R .>= -1.0)
is_corr_cholfac(L) = isapprox(sum(L.^2, dims=2), ones(eltype(L), size(L,1))) && all(diag(L) .> 0.0)

function chol_from_vec(x, K)
    w = vec_to_lt(x, K)
    for i in 1:K
        if sum(w[i,:].^2) > 1
            w[i,i] = 0.0
        else
            w[i,i] = sqrt(1 - sum(w[i,:].^2))
        end
    end
    LowerTriangular(w)
end

function vec_to_lt(y, K)
    w = zeros(eltype(y), K, K)
    k = 1
    for i in 2:K
        for j in 1:(i-1)
            w[i,j] = y[k]
            k += 1
        end
    end
    LowerTriangular(w)
end

function lt_to_vec(w::LowerTriangular)
    K = size(w,1)
    x = Vector{eltype(w)}(undef, Int(K*(K-1)/2))
    k = 1
    for i in 2:K
        for j in 1:(i-1)
            x[k] = w[i,j]
            k += 1
        end
    end
    x
end

## log jacobian factors using automatic differentiation
function log_jacobian_det_free_to_chol_ad(y, K)
    h(x) = lt_to_vec(transform_free_to_chol(x, K))
    log(abs(det(ForwardDiff.jacobian(h, y))))
end

function log_jacobian_det_chol_to_free_ad(w::LowerTriangular, K)
    h(x) = transform_chol_to_free(chol_from_vec(x, K))
    log(abs(det(ForwardDiff.jacobian(h, lt_to_vec(w)))))
end

d = 5
eta = 2.36

distr = LKJcorr(d, eta)
@test params(distr) == (d, eta)
@test length(distr) == d

R = rand(distr)
@test is_corr_matrix(R)
    
distr = LKJcorrChol(d, eta)
@test params(distr) == (d, eta)
@test length(distr) == d

L = rand(distr)
@test is_corr_cholfac(L)

y = transform_chol_to_free(L)

## check that transform_free_to_chol is inverse of transform_chol_to_free
@test isapprox(transform_free_to_chol(y, d), L)

## check that log jacobian factor from automatic differentation is the same
@test isapprox(log_jacobian_det_free_to_chol(y, d), log_jacobian_det_free_to_chol_ad(y, d))
