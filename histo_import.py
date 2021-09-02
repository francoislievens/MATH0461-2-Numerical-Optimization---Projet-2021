import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Get list of files
lst = os.listdir("csv/")

for i in range(len(lst)):
    itm = lst[i]
    print("------------------------------")
    print(" Building histogram {}/{}".format(i, len(lst)))
    print(" file: {}".format(itm))

    df = pd.read_csv("csv/{}".format(itm))
    data = df.to_numpy()
    avg = np.average(data)
    print("AVG: {}".format(avg))


    # Plot the histogram
    plt.hist(data.flatten(), bins=100, range=(-1, 1))
    plt.yscale("log")

    # Save the plot
    name = itm.replace(".png", "")
    plt.savefig("OutputsHistogramsDirect/{}_hist.png".format(name))
    plt.close()

