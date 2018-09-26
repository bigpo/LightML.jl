mutable struct LSC
    n_clusters::Int64
    n_landmarks::Int64
    method::Symbol
    non_zero_landmarks::Int64
    bandwidth::Float64
    cluster_result::Vector
end

function LSC(;
             n_clusters::Int64 = 2,
             n_landmarks::Int64 = 150,
             method::Symbol = :Kmeans,
             non_zero_landmarks::Int64 = 4,
             bandwidth::Float64 = 0.4,
             cluster_result::Vector = zeros(10))

    return LSC(n_clusters, n_landmarks, method, non_zero_landmarks,
               bandwidth, cluster_result)

end

function gaussianKernel(distance, bandwidth)
    exp(-distance / (2 * bandwidth^2));
end

function get_landmarks(X, p;method=:Kmeans)
    if(method == :Random)
        numberOfPoints = size(X, 2);
        landmarks = X[:, Random.randperm(numberOfPoints)[1:p]];
        return landmarks;
    end

    if(method == :Kmeans)
        kmeansResult = kmeans(X, p)
        return kmeansResult.centers;
    end
    throw(ArgumentError("method can only be :Kmeans or :Random"));
end

function compose_sparse_Z_hat_matrix(X, landmarks, bandwidth, r)
    distances = pairwise(Distances.Euclidean(), landmarks, X);
    similarities = map(x -> gaussianKernel(x, bandwidth), distances);
    ZHat = zeros(size(similarities));

    for i in 1:(size(similarities, 2))
        topLandMarksIndices = partialsortperm(similarities[:, i], 1:r, rev=true);
        topLandMarksCoefficients = similarities[topLandMarksIndices, i];
        topLandMarksCoefficients = topLandMarksCoefficients / sum(topLandMarksCoefficients);
        ZHat[topLandMarksIndices, i] = topLandMarksCoefficients;
    end
    return LinearAlgebra.Matrix(LinearAlgebra.Diagonal((sum(ZHat, dims=2)[:])))^(-1/2) * ZHat;

end

function train!(model::LSC, X::Matrix)
    if size(X, 1) > size(X, 2)
        X = copy(transpose(X))
    end
    landmarks = get_landmarks(X, model.n_landmarks, method = model.method)
    Z_hat = compose_sparse_Z_hat_matrix(X, landmarks, model.bandwidth,
                                        model.non_zero_landmarks)
    svd_result = LinearAlgebra.svd(transpose(Z_hat))
    temp = copy(transpose(svd_result.U[:, 1:model.n_clusters]))
    model.cluster_result = kmeans(temp, model.n_clusters).assignments
end

function plot_in_2d(model::LSC, X::Matrix)
    if size(X, 1) > size(X, 2)
        X = transpose(X)
    end
    Gadfly.plot(x = X[1, :], y = X[2, :], color = model.cluster_result)
end
