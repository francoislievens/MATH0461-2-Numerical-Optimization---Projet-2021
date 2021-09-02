import pickle
import numpy

path_in = "ADD_YOUR_PATH_HERE"
path_out = "ADD_YOUR_PATH_HERE"
f_in = open(path_in, "rb")
f_out = open(path_out, "wb")
array = pickle.load(f_in)
numpy.savetxt(f_out, array)
f_in.close()
f_out.close()
