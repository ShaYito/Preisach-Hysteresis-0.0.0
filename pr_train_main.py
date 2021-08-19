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

    plt.figure()
    plt.subplot(211)
    plt.plot(n, x)
    plt.title('input x(k) ~ time k')
    plt.subplot(212)
    plt.plot(n, y)
    plt.title('output y(k) ~ time k')

    M = 60  # discretization
    syt = pr.Preisach()
    x = syt.normalize(x)
    y = syt.normalize(y)

    print(len(x))

    # plt.figure()
    # plt.title('')
    # plt.plot(n, x, label='x')
    # plt.plot(n, y, label='y')
    # plt.legend()

    # Train the model with M, x, y
    syt.forward_train(M, y, x)

    plt.show()
