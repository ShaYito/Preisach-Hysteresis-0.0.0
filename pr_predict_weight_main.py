import openpyxl
import pr
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    gl = np.load('TestData\\mat_stream_L.npy', encoding="latin1")
    gr = np.load('TestData\\mat_stream_R.npy', encoding="latin1")
    tl = np.load('TestData\\time_stamp_L.npy', encoding="latin1")[:, 1]
    tr = np.load('TestData\\time_stamp_R.npy', encoding="latin1")[:, 1]

    print(len(gl), len(gr), len(tl), len(tr))
    t = min(len(tl), len(tr))
    n = tl[:t]
    n = (n - n[0]) / 1000
    gl = gl[:t]
    gr = gr[:t]

    sample = 5
    n = n[::sample]
    gl = gl[::sample]
    gr = gr[::sample]

    sp = np.shape(gl)
    gmax = max(max(gl.flatten()), max(gr.flatten()))  # gmax = 2.78e-05

    conductance = np.zeros(sp[0])
    for k in range(0, sp[0]):
        conductance[k] = sum(sum(gl[k])) + sum(sum(gr[k]))

    plt.figure()
    plt.plot(n, conductance)

    M = 36  # discretization
    syt = pr.Preisach()

    r = 1
    s = 1
    yl = np.zeros(sp[0])
    yr = np.zeros(sp[0])
    xcl = np.zeros(sp)
    xcr = np.zeros(sp)
    cnt = 0
    for i in range(int(sp[1] / r)):
        for j in range(int(sp[2] / s)):
            cnt = cnt + 1
            print(cnt)
            yl = gl[:, r * i, s * j]
            if max(yl) == 0:
                continue
            else:
                yl = syt.normalize(yl)
                xcl[:, r * i, s * j] = syt.forward(yl, M)
                xcl[:, r * i, s * j] = syt.normalize(xcl[:, r * i, s * j])
    cnt = 0
    for i in range(int(sp[1] / r)):
        for j in range(int(sp[2] / s)):
            cnt = cnt + 1
            print(cnt)
            yr = gr[:, r * i, s * j]
            if max(yr) == 0:
                continue
            else:
                yr = syt.normalize(yr)
                xcr[:, r * i, s * j] = syt.forward(yr, M)
                xcr[:, r * i, s * j] = syt.normalize(xcr[:, r * i, s * j])

    weight = np.zeros(sp[0])
    for k in range(0, sp[0]):
        weight[k] = sum(xcl[k].flatten()) + sum(xcr[k].flatten())

    plt.figure()
    plt.plot(n, weight)
    plt.show()
