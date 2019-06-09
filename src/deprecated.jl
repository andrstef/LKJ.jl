corr_from_chol(w) = w*w'

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

## transform cholesky factor (vector repr) to KxK corr matrix (vector repr)
function chol_vec_to_corr_vec(y, K)
    lt_to_vec(LowerTriangular(corr_from_chol(chol_from_vec(y, K))))
end

function log_jacobian_det_chol_to_corr(w::LowerTriangular)
    K = size(w,1)
    sum((K .- (2:K)).*log.(diag(w)[2:K]))
end

function log_jacobian_det_free_to_corr(y, K)
    w = free_to_chol(y, K)
    log_jacobian_det_free_to_chol(y, K) + log_jacobian_det_chol_to_corr(w)
end

function lt_to_vec(w::LowerTriangular)
    K = size(w)[1]
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

ut_to_vec(w::UpperTriangular) = lt_to_vec(LowerTriangular(w'))
vec_to_ut(y, K) = UpperTriangular(vec_to_lt(y, K)')

function corr_to_free(w::UpperTriangular)
    K = size(w,1)
    
    z = UpperTriangular(zeros(eltype(w), K, K))
    z[1,2:end] = w[1,2:end]
    
    for i in 2:K
        for j in (i+1):K
            ##z[i,j] = w[i,j] / sqrt(prod(1 .- z[1:(i-1),j].^2))
            z[i,j] = w[i,j] / sqrt(1 - sum(w[1:(i-1),j].^2))
        end
    end
    atanh.(z)
end
    
function free_to_corr(y::UpperTriangular)
    z = tanh.(y)
    
    K = size(z)[1]
    w = zeros(eltype(y), K, K)

    w[1,1] = 1
    w[1,2:end] = z[1,2:end]
    
    for i in 2:K
        w[i,i] = sqrt(prod(1 .- z[1:(i-1),i].^2))
        for j in (i+1):K
            w[i,j] = z[i,j]*sqrt(prod(1 .- z[1:(i-1),j].^2))
        end
    end

    UpperTriangular(w) ##w'*w
end

## p_y(y) \propto |J|p_w(w) \propto |J'|p_x(x)
function logpdf_free(y::Vector, K::Int, eta::Float64)
    w = free_to_corr(y, K)
    d = LKJcorrChol(K, eta)
    logpdf(d, w) + log_jacobian_det_free_to_chol(y, K)
end
