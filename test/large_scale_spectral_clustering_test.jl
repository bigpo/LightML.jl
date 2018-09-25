import DelimitedFiles

@testset "Read data" begin
    path_ = "../resources/data"
    data_name = ["smiley", "spirals", "shapes", "cassini1"]
    datasets = joinpath.(@__DIR__, path_, string.(data_name, ".csv"))
    for i in 1:4
        data = DelimitedFiles.readdlm(datasets[i])
        if i == 1
            @test isapprox(data[1, :], [-0.817135877712173 1.09291948845948 1]')
        end
    end
end
