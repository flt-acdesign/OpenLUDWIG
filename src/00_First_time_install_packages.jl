// # FILE: .\src\00_First_time_install_packages.jl
import Pkg

function install_dependencies()
    println("==========================================================")
    println("      LBM SOLVER | DEPENDENCY INSTALLER                   ")
    println("==========================================================")

    
    dependencies = [
        "KernelAbstractions", # For backend-agnostic kernel writing
        "CUDA",               # For GPU acceleration
        "Adapt",              # For data transfer between CPU/GPU
        "StaticArrays",       # For high-performance small arrays (SVector)
        "YAML",               # For reading configuration files
        "WriteVTK",           # For exporting results to ParaView
        "Atomix"              # For atomic array operations in kernels
    ]

    println("[Installer] Activating project environment...")
    
    # This creates a Project.toml and Manifest.toml if they don't exist
    #Pkg.activate(".") 

    println("[Installer] The following packages will be installed:")
    for dep in dependencies
        println("  - $dep")
    end
    println("----------------------------------------------------------")

    try
        
        Pkg.add(dependencies)
        
        println("\n[Installer] Packages added successfully.")
        println("[Installer] Precompiling (this may take a few minutes)...")
        
        
        Pkg.precompile()
        
        println("\n[Success] Environment is ready!")
        println("----------------------------------------------------------")
        println("To run the solver:")
        println("1. Open a terminal in this folder")
        println("2. Run: julia --project=. src/main.jl")
        
    catch e
        println("\n[Error] Installation failed!")
        println(e)
    end
end


install_dependencies()