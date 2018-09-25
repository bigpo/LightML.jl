using LightML
import LightML: LSC
using DataFrames
import DelimitedFiles
using Gadfly

"""
    LSC_example(; plot=true, printdata=true)

Large-scale spectral clustering example.
"""
function LSC_example(; plot=true, printdata=true)
    data_name = ["smiley", "spirals", "shapes", "cassini1"]
    datasets = joinpath.(LightML.data_dir, string.(data_name, ".csv"))
    clusters = [4, 2, 4, 3]
    n_landmarks = [50, 150, 50, 50]
    bandwidth = [0.4, 0.04, 0.04, 0.4]
    count = 0
    df = DataFrame()
    for i in 1:4
        data = DelimitedFiles.readdlm(datasets[i])
        data = transpose(convert(Array{Float64, 2}, data[:, 1:2]))
        model = LSC(n_clusters = clusters[i], n_landmarks = n_landmarks[i], bandwidth = bandwidth[i])
        train!(model, copy(data))
        dataframe = DataFrame(x=data[1, :], y=data[2, :], group=model.cluster_result, datasets=data_name[i])
        if count == 0
            df = dataframe
            count += 1
        else
            df = vcat(df, dataframe)
        end
        println("Progress: $(i/4*100)%....")
    end
    if  printdata
        println("$df")
    end
    if plot
       println("computing finished, drawing the plot......")
        set_default_plot_size(25cm, 14cm)
        Gadfly.plot(df, x="x", y = "y", color = "group", xgroup = "datasets", Geom.subplot_grid(Geom.point),
                    Guide.title("Large Scale Spectral Clustering"))
    end
end
