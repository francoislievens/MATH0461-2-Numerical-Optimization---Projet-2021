
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
println("*            with uncorupted measurements                 *")
println("*                                                         *")
println("* Makes your choice:                                      *")
println("*     1: M = 608                                          *")
println("*     2: M = 1014                                         *")
println("*     3: M = 1521                                         *")
println("*     4: M = 3042                                         *")
println("*                                                         *")
println("* ======================================================= *")

entry = readline()

# Init parameters
M = 0
if entry == "1"
    M = 608
elseif entry == "2"
    M = 1014
elseif entry == "3"
    M = 1521
else
    M = 3042
end

println(" Data Loading for M", M, " uncorrupted signal...")

# Size of the input signal
N = 6084

# Load sparsity basis
psi = unpickler("data/basis_matrix.pickle")

# Load signal
m = unpickler(string("Data/uncorrupted_measurements_M", M, ".pickle"))

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
@constraint(model, initial[c=1:M], sum(mt[c,i]*x[i] for i in 1:N) == m[c])
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

# Export x
x_hat = value.(x)
CSV.write(string("csv/L1_Uncorrupt_M", M, ".csv"), Tables.table(x_hat), writeheader=false)

# Save the image
save(string("Outputs/L1_Uncorrupt_M", M, ".png"), clamp!(cell, 0.0, 1.0))

println("... Done.")
println("Image exported in the file Outputs/L1_Uncorrupt_M", M, ".png")

println("End of the program")




