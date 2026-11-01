# Build the documentation locally with
#
#     julia --project=docs -e 'using Pkg; Pkg.instantiate()'   # once
#     julia --project=docs docs/make.jl
#
# The result is in docs/build. The Gset tutorial reads its graphs from the
# working directory, as a reader would after downloading them. The build
# downloads them from Ye's page into docs/gset (not tracked) the first time,
# so the first build on a machine needs network access, and runs the examples
# there. On GitHub Actions, deploydocs publishes the result to the gh-pages
# branch; run locally, it does nothing.

using Documenter
using Downloads
using MFAOA

const GSET_URL = "https://web.stanford.edu/~yyye/yyye/Gset"
const GSET_DIR = joinpath(@__DIR__, "gset")
const GSET_GRAPHS = ("G1", "G11", "G22", "G43", "G55")    # the graphs the tutorial reads

"""
    fetch_gset(dir, names)

Download each Gset graph in `names` into `dir` unless it is already there.
"""
function fetch_gset(dir, names)
    mkpath(dir)
    for name in names
        path = joinpath(dir, name)
        isfile(path) || Downloads.download("$GSET_URL/$name", path)
    end
end

fetch_gset(GSET_DIR, GSET_GRAPHS)

makedocs(;
    sitename = "MFAOA.jl",
    modules = [MFAOA],
    # Internal helpers have docstrings too; only the exported names (and
    # MFAOA.Optimizer, listed on the API page) are public.
    checkdocs = :exports,
    repo = Documenter.Remotes.GitHub("Quicopt", "MFAOA.jl"),
    workdir = GSET_DIR,
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true",
                             canonical = "https://quicopt.github.io/MFAOA.jl/stable/",
                             edit_link = "main"),
    pages = [
        "Home" => "index.md",
        "Tutorial: Max-Cut on Gset" => "tutorials/maxcut_gset.md",
        "Manual" => [
            "manual/problems.md",
            "manual/jump.md",
            "manual/schedules.md",
            "manual/symmetry.md",
            "manual/performance.md",
        ],
        "API reference" => "api.md",
    ],
)

deploydocs(; repo = "github.com/Quicopt/MFAOA.jl.git", devbranch = "main")
