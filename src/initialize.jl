// # FILE: .\src\initialize.jl
"""
INITIALIZE.JL - Initialization Wrapper

Loads the configuration and physics scaling modules.
"""

include("config_loader.jl")
include("physics_scaling.jl")

println("[Init] Module loaded (ROBUST VERSION)")