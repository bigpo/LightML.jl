import DelimitedFiles

@testset "Read data" begin
    data_name = ["smiley", "spirals", "shapes", "cassini1"]
    datasets = joinpath.(LightML.data_dir, string.(data_name, ".csv"))
    for i in 1:4
        data = DelimitedFiles.readdlm(datasets[i])
        if i == 1
            @test isapprox(data[1, :], [-0.817135877712173 1.09291948845948 1]')
        end
    end
end
