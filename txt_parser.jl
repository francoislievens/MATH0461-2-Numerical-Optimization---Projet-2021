using DelimitedFiles

function txt_parser(M, N, path)

    array_string = readdlm(path, '\n');
    data_matrix = Array{Float64, 2}(undef, M, N)
    tmp = Array{String, 1}(undef, N)
    @inbounds for i = 1:length(array_string)
        if typeof(array_string[i]) == Float64
            data_matrix[i,:] .= array_string[i]
        elseif typeof(array_string) == String
            tmp .= split(array_string[i], " ")
            data_matrix[i,:] .= parse.(Float64, tmp)
        end
    end
    return data_matrix
end
