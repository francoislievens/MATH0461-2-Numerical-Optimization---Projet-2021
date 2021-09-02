
using ImageView
using Images
using ImageIO
using Gurobi
using JuMP
using CSV, Tables

include("utilities.jl")
include("txt_parser.jl")

println("* ======================================================= *")
println("*                                                         *")
println("*              L-1 norm image approximer                  *")
println("*               with noisy measurements                   *")
println("*             by using the robust variant B               *")
println("*             Loop for a given value of eps               *")
println("*               Over all measurements set                 *")
println("*                                                         *")
println("*                Chose eps value between 0 and 1...       *")
println("*                                                         *")
println("* ======================================================= *")

# Choice of the value of epsilon
eps = 0
entry = readline()
eps = parse(Float64, entry)
println("Choised epsilon value: ", eps)

# Noisy or not
noisy = "N"
println("Do you want to use noisy measurements?")
println("Yes = Y, No = N...")
entry = readline()
if entry == "Y"
    noisy = "Y"
end

# Size of the input signal
N = 6084

# Load sparsity basis
psi = unpickler("data/basis_matrix.pickle")


for M in [608, 1014, 1521, 3042]

    println("* ======================================================= *")
    println("*          Start for measurement  ", M, "                         *")
    println("* ======================================================= *")
    println("Model building...")

    # Load signal
    if noisy == "Y"
        m = unpickler(string("Data/noisy_measurements_M", M, ".pickle"))
    else
        m = unpickler(string("Data/uncorrupted_measurements_M", M, ".pickle"))
    end

    # Load Sampling matrix
    phi = unpickler(string("Data/measurement_matrix_M", M, ".pickle"))
    mt = phi*psi

    println("... Done.")



    println("Model building...")

    # Instanciate the model
    model = Model(Gurobi.Optimizer)

    # Add variables:
    @variable(model, x[1:N])        # The sparse vector that we want to approximate
    @variable(model, t[1:N])        # New variable for the epigraph tricks

    # Add Objective function
    @objective(model, Min, sum(t))

    # Add constraints:
    @constraint(model, [eps; mt*x - m] in SecondOrderCone())
    @constraint(model, linearA[n=1:N], x[n] <= t[n])
    @constraint(model, linearB[n=1:N], -t[n] <= x[n])

    println("... Done.")
    println("* Optimizing.... *")
    optimize!(model)
    println("... Done.")

    # Get cell image
    println("Export the image...")
    r = psi*value.(x)
    cell = reshape(r, (78,78))

    # Save the image and export x^hat
    x_hat = value.(x)

    if noisy == "Y"
        save(string("Outputs/L1_rob_B_noisy_M", M, "_eps_", eps, ".png"), clamp!(cell, 0.0, 1.0))
        CSV.write(string("csv/L1_rob_B_noisy_M", M, "_eps_", eps, ".csv"), Tables.table(x_hat), writeheader=false)
        println("... Done.")
        println("Image exported in the file Outputs/L1_rob_A_noisy_M", M, "_eps_",eps, ".png")
    else
        save(string("Outputs/L1_rob_B_uncorr_M", M, "_eps_", eps, ".png"), clamp!(cell, 0.0, 1.0))
        CSV.write(string("csv/L1_rob_B_uncorr_M", M, "_eps_", eps, ".csv"), Tables.table(x_hat), writeheader=false)
        println("... Done.")
        println("Image exported in the file Outputs/L1_rob_B_uncorr_M", M, "_eps_",eps, ".png")
    end
end

println("End of the program")





