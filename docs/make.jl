using Documenter: Documenter
using FlowTransformBindings: FlowTransformBindings

Documenter.makedocs(;
    modules = [FlowTransformBindings],
    sitename = "FlowTransformBindings.jl",
    authors = "Jordan Benjamin",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://jbphyswx.github.io/FlowTransformBindings.jl",
        edit_link = "main",
    ),
    checkdocs = :public,
    pages = [
        "Home" => "index.md",
        "Nonuniform FFTs" => "nufft.md",
        "Packed layout" => "layout.md",
        "Least squares" => "lsmr.md",
        "FastTransforms from tasks" => "fasttransforms.md",
    ],
)

Documenter.deploydocs(;
    repo = "github.com/jbphyswx/FlowTransformBindings.jl",
    devbranch = "main",
)
