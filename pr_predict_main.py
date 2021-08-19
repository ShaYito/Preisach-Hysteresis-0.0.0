import openpyxl
import pr
import numpy as np
import matplotlib.pyplot as plt

# x -> ycal, y -> xinv

if __name__ == '__main__':
    sheet = openpyxl.load_workbook("TestData\\test55.xlsx").active
    lst = [column for column in sheet.values]
    nxy = np.array(lst)

    sample = 5
    n = nxy[:, 0][::sample]
    x = nxy[:, 1][::sample]  # input
    y = nxy[:, 2][::sample]  # output

    M = 60  # discretization

    syt = pr.Preisach()

    x = syt.normalize(x)
    y = syt.normalize(y)

    xcal = syt.forward(y, M)

    cut = int(len(x) / 100)
    n = n[cut:]
    x = x[cut:]
    y = y[cut:]
    xcal = xcal[cut:]

    xcal = syt.normalize(xcal)

    x_err = abs((xcal - x))
    print(np.corrcoef(y, xcal))

    plt.figure()
    plt.subplot(211)
    plt.title('backward model')
    plt.plot(x, label='x_measure')
    plt.plot(xcal, label='x_predict')
    plt.legend()
    plt.subplot(212)
    plt.title('error: |x_predict-x_measure|')
    plt.plot(x_err)

    plt.show()
