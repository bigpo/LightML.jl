mutable struct SVM{T <: Real, V <: Real}
    X::Matrix{T}
    y::Vector{V}
    C::T
    tol::T
    max_iter::Int
    kernel::String
    degree::Int
    gamma::T
    alpha::Vector{T}
    b::T
    sv_indx::Vector{Int}
    K::Matrix{T}
end

function svm(X::Matrix{T},
             y::Vector{V};
             C::T = one(T),
             kernel::String = "linear",
             max_iter::Int = 100,
             tol::T = one(T) / 10^3,
             degree::Int = 2,
             gamma::T = one(1) / 10,
             alpha::Vector = zeros(T, 10),
             b::T = zero(T)) where {T <: Real, V <: Real}
    n = size(X,1)
    alpha = zeros(T, n)
    K = zeros(T, n, n)
    sv_indx = collect(1:n)
    return SVM(X, y, C, tol, max_iter, kernel, degree, gamma, alpha, b, sv_indx, K)
end

function predict(model::SVM, x::Array)
    n = size(x, 1)
    res = zeros(eltype(x), n)
    if n == 1
        res[1] = predict_row(x, model)
    else
        for i = 1:n
            res[i] = predict_row(x[i, :], model)
        end
    end
    return res
end

function train!(model::SVM)
    n_sample = size(model.X, 1)
    model.K = zeros(eltype(model.X), n_sample, n_sample)
    for i in 1:n_sample
        model.K[:, i] .= kernel_c(model.X, model.X[i, :], model)
    end
    # start training

    iters = 0
    while iters < model.max_iter
        iters += 1
       # println("Processing $(iters)/$(model.max_iter)")
        alpha_prev = copy(model.alpha)
        for j = 1:n_sample
            i = rand(1:n_sample)
            eta = 2 * model.K[i, j] - model.K[i, i] - model.K[j, j]
            if eta >= 0
                continue
            end
            L, H = count_bounds(i, j, model)

            # Error for current examples
            e_i, e_j = error_(i, model), error_(j, model)

            # Save old alphas
            alpha_io, alpha_jo = model.alpha[i], model.alpha[j]

            # Update alpha
            model.alpha[j] -= (model.y[j] * (e_i - e_j)) / eta
            model.alpha[j] = clamp(model.alpha[j], L, H)

            model.alpha[i] = model.alpha[i] + model.y[i] * model.y[j] * (alpha_jo - model.alpha[j])

            # Find intercept
            b1 = model.b - e_i - model.y[i] * (model.alpha[i] - alpha_jo) * model.K[i, i] -
                 model.y[j] * (model.alpha[j] - alpha_jo) * model.K[i, j]
            b2 = model.b - e_j - model.y[j] * (model.alpha[j] - alpha_jo) * model.K[j, j] -
                 model.y[i] * (model.alpha[i] - alpha_io) * model.K[i, j]
            if 0 < model.alpha[i] < model.C
                model.b = b1
            elseif 0 < model.alpha[j] < model.C
                model.b = b2
            else
                model.b = 1//2 * (b1 + b2)
            end

            # Check convergence
            diff = LinearAlgebra.norm(model.alpha - alpha_prev)
            if diff < model.tol
                break
            end
        end
    end
    #println("Convergence has reached after $(iters). for $(model.kernel)")
    # Save support vectors index
    model.sv_indx = findall(!iszero, model.alpha .> 0)
    return nothing
end

function kernel_c(X::Matrix,
                y::Vector,
                model::SVM)
    if model.kernel == "linear"
        return X * y
    elseif model.kernel == "poly"
        return (X * y).^model.degree
    elseif model.kernel == "rbf"
        n = size(X, 1)
        res = zeros(n)
        for i = 1:n
            res[i] = MathConstants.e^(-model.gamma*sum(abs2, X[i,:]-y))
        end
        return res
    end
end

function count_bounds(i,j,model)
    if model.y[i] != model.y[j]
        L = max(0, model.alpha[j] - model.alpha[i])
        H = min(model.C, model.C - model.alpha[i] + model.alpha[j])
    else
        L = max(0, model.alpha[i] + model.alpha[j] - model.C)
        H = min(model.C, model.alpha[i] + model.alpha[j])
    end
    return L, H
end

function predict_row(x,model)
    res = kernel_c(model.X,x,model)
    return sign(res' * (model.alpha .* model.y) + model.b)[1]
end

function error_(i,model)
    return predict_row(model.X[i,:],model) - model.y[i]
end

function plot_test_svm()
    (X_test, predictions) = test_svm()
    pca_model = PCA()
    train!(pca_model, X_test)
    return plot_in_2d(pca_model, X_test, predictions, "svm")
end

function test_svm()
    X_train, X_test, y_train, y_test = make_cla(n_features = 14)
    predictions = 0
    for kernel in ["linear", "rbf"]
        model = svm(X_train, y_train, max_iter=500, kernel=kernel, C=0.6)
        train!(model)
        predictions = predict(model,X_test)
        println("Classification accuracy $(kernel): $(accuracy(y_test, predictions))")
    end
    return (X_test, predictions)
end
