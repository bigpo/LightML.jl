@testset "ClassificationTree" begin
    X_train, X_test, y_train, y_test = LightML.make_iris()
    y_train = LightML.one_hot(y_train)
    y_test = LightML.one_hot(y_test)
    model = LightML.ClassificationTree()
    train!(model,X_train, y_train)
    predictions = predict(model,X_test)
    y_test = LightML.unhot(y_test)
    predictions = LightML.unhot(predictions)
    @test LightML.accuracy(y_test, predictions) <= 0.97
end
