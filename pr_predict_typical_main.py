import openpyxl
import pr
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    gl = np.load('TestData\\typical\\mat_stream_L.npy', encoding="latin1")
    gr = np.load('TestData\\typical\\mat_stream_R.npy', encoding="latin1")
    tl = np.load('TestData\\typical\\time_stamp_L.npy', encoding="latin1")[:, 1]
    tr = np.load('TestData\\typical\\time_stamp_R.npy', encoding="latin1")[:, 1]

    print(len(gl), len(gr), len(tl), len(tr))
    t = min(len(tl), len(tr))
    n = tl[:t]
    n = (n - n[0]) / 1000
    gl = gl[:t]
    gr = gr[:t]

    sample = 1
    begin = 1200
    end = 1800
    n = n[::sample][begin:end]
    gl = gl[::sample][begin:end]
    gr = gr[::sample][begin:end]

    sp = np.shape(gl)
    gmax = max(max(gl.flatten()), max(gr.flatten()))  # gmax = 2.78e-05

    cl = np.zeros(sp[0])
    cr = np.zeros(sp[0])
    for k in range(0, sp[0]):
        cl[k] = sum(sum(gl[k]))
        cr[k] = sum(sum(gr[k]))

    plt.figure()
    plt.plot(n, cl)
    plt.plot(n, cr)

    # M = 36  # discretization
    # syt = pr.Preisach()
    #
    # r = 5
    # s = 4
    # yl = np.zeros(sp[0])
    # yr = np.zeros(sp[0])
    # xcl = np.zeros(sp)
    # xcr = np.zeros(sp)
    # cnt = 0
    # for i in range(int(sp[1] / r)):
    #     for j in range(int(sp[2] / s)):
    #         cnt = cnt + 1
    #         print(cnt)
    #         yl = gl[:, r * i, s * j]
    #         if max(yl) == 0:
    #             continue
    #         else:
    #             yl = syt.normalize(yl)
    #             xcl[:, r * i, s * j] = syt.forward(yl, M)
    #             xcl[:, r * i, s * j] = syt.normalize(xcl[:, r * i, s * j])
    # cnt = 0
    # for i in range(int(sp[1] / r)):
    #     for j in range(int(sp[2] / s)):
    #         cnt = cnt + 1
    #         print(cnt)
    #         yr = gr[:, r * i, s * j]
    #         if max(yr) == 0:
    #             continue
    #         else:
    #             yr = syt.normalize(yr)
    #             xcr[:, r * i, s * j] = syt.forward(yr, M)
    #             xcr[:, r * i, s * j] = syt.normalize(xcr[:, r * i, s * j])
    #
    # wl = np.zeros(sp[0])
    # wr = np.zeros(sp[0])
    # for k in range(0, sp[0]):
    #     wl[k] = sum(xcl[k].flatten())
    #     wr[k] = sum(xcr[k].flatten())
    #
    #
    # plt.figure()
    # plt.plot(n, wl, label='left')
    # plt.plot(n, wr, label='right')


    plt.show()
