"""
    load_examples()

Load `LightML` examples into the `Main` module.
"""
function load_examples()
    example_code = ["large_scale_spectral_clustering_example.jl"]
    for file in example_code
        Base.include(Main, joinpath(examples_dir, file))
    end
    return nothing
end
