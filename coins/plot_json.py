import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.mlab import PCA


filenames = glob.glob("data/*.json")

gridSizeSeconds = 60
tStart = 1504450000
tEnd =   1504811600
grid = np.linspace(tStart, tEnd, num=np.int((tEnd-tStart)/gridSizeSeconds))

allData = np.array([grid])

x = 0
for coin in filenames:

    print("Loading " + coin)
    json_data=open(coin).read()
    data = json.loads(json_data)
    np_array = np.array(data).transpose()

    u, indices = np.unique(np_array[0], return_index=True)
    uTimes = np_array[0][indices] * 0.001
    uVals = np_array[1][indices]

    vals=uVals * (1.0 / np.max(uVals))
    gridVals=griddata(uTimes, vals, grid, method='linear')

    allData = np.concatenate((allData, [gridVals]), axis=0)

    # plt.plot(grid, gridVals, label=coin)

    # x = x + 1
    # if(x > 10):
    #     break

allData = allData.transpose()
allData = allData[:,1:]

eigenvectors, eigenvalues, V = np.linalg.svd(allData.T, full_matrices=False)
projected_data = np.dot(allData, eigenvectors)

plt.plot(grid, projected_data, label='Projection')

# plt.legend()
# plt.axis([tStart, tEnd, -2.5, 1])
plt.show()
