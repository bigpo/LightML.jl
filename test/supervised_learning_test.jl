@testset "SVM" begin
    X_train, X_test, y_train, y_test = LightML.make_cla(n_features = 14)
    for (kernel, acc) in [("linear", 0.725) , ("rbf", 2.6375 / 3)]
        model = LightML.svm(X_train, y_train, max_iter=500, kernel=kernel, C=0.6)
        train!(model)
        predictions = predict(model,X_test)
        @test isapprox(acc, LightML.accuracy(y_test, predictions))
    end
end
