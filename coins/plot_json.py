import glob
import json
import time
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

filenames = glob.glob("data/*.json")

gridSizeSeconds = 60
tStart = time.mktime(date(2017, 9, 1).timetuple())
tEnd =   time.mktime(date(2017, 9, 7).timetuple())
grid = np.linspace(tStart, tEnd, num=np.int((tEnd-tStart)/gridSizeSeconds))


f, axarr = plt.subplots(2, sharex=True)

# extractDayTime = lambda t: ((t%86400)- 43200)/43200
# extractDayTime = lambda t: random.randrange(-4, 4)
# gridVals = np.array([extractDayTime(t) for t in grid])
allData = np.array([grid])

x = 0
for coin in filenames:

    print("Loading " + coin)
    json_data=open(coin).read()
    data = json.loads(json_data)
    np_array = np.array(data).transpose()

    u, indices = np.unique(np_array[0], return_index=True)
    uTimes = np_array[0][indices] * 0.001 # Divide timestamp by 1000 to convert from ms to sec
    uVals = np_array[1][indices]

    gridVals=griddata(uTimes, uVals, grid, method='linear')

    # Normalize the values to the all-time maximum
    gridVals=gridVals * (1.0 / np.max(uVals))
    gridVals-=gridVals.mean(axis=0)

    allData = np.concatenate((allData, [gridVals]), axis=0)

    axarr[0].plot(grid, gridVals, label=coin)

    # x = x + 1
    # if(x > 10):
    #     break

allData = allData.transpose()

# Remove the first row as it contains the linear time
allData = allData[:,1:]

eigenvectors, eigenvalues, V = np.linalg.svd(allData.T, full_matrices=False)
projected_data = np.dot(allData, eigenvectors)

axarr[1].plot(grid, projected_data, label='Projection')

# plt.legend()
# plt.axis([tStart, tEnd, -2.5, 1])
plt.show()
