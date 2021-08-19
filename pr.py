import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.io import savemat

class Preisach(object):
    def __init__(self):
        pass

    def normalize(self, y):
        return (y-(max(y)+min(y))/2) / (max(y)-min(y))
        # return ((y-min(y)) / 2) / (max(y)-min(y))

    def plane_plot(self, pl):
        plt.imshow(pl)
        plt.xlabel('β')
        plt.ylabel('α')
        return

    def plane_initialization_half(self, m):
        # discretize the alpha-beta plane into m*m squares, labeled as [i][j], with i,j = {0,1,...,m-1}
        # choose one of configurations: half, negative or positive (for symmetry, 'half' is recommended)
        # ne: number of dominating extrema, i.e. turning points of dividing line (at time k). In ref, ne is m(k).
        # ei: the alpha coordinate of extrema; ej: the beta coordinate of extrema
        # sk: the sign of extrema (maximum: sk=+1, minimum: sk=-1)

        pl = np.zeros((m, m), dtype=int)
        for i in range(m):  # range(n): [0,1,...,n-1]
            for j in range(m - i):
                if i > j:
                    pl[i][j] = 1  # Gamma+ half-plane
                else:
                    pl[i][j] = -1  # Gamma- half-plane

        return pl

    # (obsolete) initialize vectors needed for "fast_calculation"
    def dividing_line_initialization_half(self, m):
        ne = m
        ei = np.ceil(np.arange(1, m + 1) / 2).astype(np.uint8)  # for m=60, ei=[1,1,2,2,3,3,...,29,29,30,30]
        ej = np.floor(np.arange(1, m + 1) / 2).astype(np.uint8)  # for m=60, ej=[0,1,1,2,2,3,...,28,29,29,30]
        sk = np.zeros(ne)
        for x in range(ne):
            sk[x] = math.pow(-1, x)  # sk=[1,-1,1,-1,...,1,-1]
        return ne, ei, ej, sk

    def plane_initialization_negative(self, pl, m):
        # initialize the plane to be all-negative
        pl = np.zeros((m, m), dtype=int)
        for i in range(m):
            for j in range(m - i):
                pl[i][j] = -1  # Gamma- plane
        return pl

    def plane_initialization_input(self, m):
        pass

    def plane_update(self, pl, m, x, up, alt):
        if up:
            for i in range(math.ceil((-x + 0.5) * m), m):
                for j in range(m - i):
                    if i == m or j == m:
                        break
                    if pl[i][j] == 1:
                        alt[i][j] = 0
                    else:
                        pl[i][j] = 1
                        alt[i][j] = 1  # to indicate the area that alters its sign
        else:
            # print((x + 0.5) * m)
            for j in range(math.ceil((x + 0.5) * m), m):
                for i in range(m - j):
                    # print(i, j)
                    if i == m or j == m:
                        break
                    if pl[i][j] == -1:
                        alt[i][j] = 0
                    else:
                        pl[i][j] = -1
                        alt[i][j] = 1
        plf = pl.flatten()
        plf = np.delete(plf, np.where(plf == 0))
        return alt, pl, plf

    # Method 1: Updating the alpha-beta plane and get the output y(k)
    def forward_predict(self, m, x, u):
        # enable figure()-related codes in this section
        # to see how alpha-beta plane changes
        plane = self.plane_initialization_half(m)
        nn = len(x)
        ycal = np.zeros(nn)
        ycal[0] = 0
        # aim = np.zeros(nn)
        # err = np.zeros(nn)
        # plt.figure()
        cnt = 0
        for k in range(1, nn):
            if x[k] - x[k - 1] > 0:
                up = True
            else:
                up = False
            altered_area, plane, plf = self.plane_update(plane, m, x[k], up, np.zeros((m, m)))
            # for i in range(m):
            #     for j in range(m - i):
            #         aim[k] = aim[k] + plane[i][j] * u[i][j]
            ycal[k] = ycal[k - 1] + self.everett(x[k], x[k - 1], m, u, altered_area)
            # err[k] = (ycal1[k] - aim[k]) ** 2  # equation (6.17)
            # plt.cla()
            # plt.subplot(121)
            # self.plane_plot(plane)
            # cnt = cnt + 1
            # plt.title(str(cnt))
            # plt.subplot(122)
            # self.plane_plot(altered_area)
            # plt.pause(0.001)
            # count = count + 1
        # plt.ioff()
        # op1 = sum(err)  # equation (6.17)
        # if op > op1:
        #     op = op1
        #     ycal = ycal1
        return ycal

    # Train the model: write plane pa and output y at each moment into csv
    # (then, run "density_identification.m" to identify density u)
    def forward_train(self, m, x, y):
        plane = self.plane_initialization_half(m)
        # ne, ei, ej, sk = self.dividing_line_initialization_half(m)
        nn = len(x)
        pa = np.zeros((nn, int((m + 1) * m / 2)))
        for k in range(1, nn):
            if x[k] - x[k - 1] > 0:
                up = True
            else:
                up = False
            altered_area, plane, plf = self.plane_update(plane, m, x[k], up, np.zeros((m, m)))
            pa[k, :] = plf
        savemat('AYM\\textile.mat', {'A': pa, 'Y': y, 'm': m})
        return

    # (obsolete) an analytical function for weight distribution
    def u_dat(self, m, par):
        amp, h1, sg1, h2, sg2, ita = par
        u = np.zeros((m, m))
        for i in range(m):
            for j in range(m - i):
                a = i / m - 0.5
                b = j / m - 0.5
                u[i][j] = amp / (1 + np.power(((a + b + h1) * sg1) ** 2 + ((a - b - h2) * sg2) ** 2, ita))
        return u

    # (obsolete) loop to change six parameters of u_dat
    def u_dat_parameter_loop(self, x, m):
        loop = 10
        op = float('inf')
        ycal = np.zeros(len(x))
        sg2 = 0
        par = (0, 0, 0, 0, 0, 0)
        for i in range(loop):
            amp = 1.3
            h1 = -2
            sg1 = 4
            h2 = 0
            sg2 = sg2 + 0.5
            ita = 20
            par = (amp, h1, sg1, h2, sg2, ita)
            u = self.u_dat(m, par)
            op, ycal = self.forward_calculation(m, x, u, op, ycal)
        # print(par)
        return ycal, par

    # Use the density u identified in MATLAB
    def u(self):
        return np.loadtxt(open('density.csv', 'rb'), delimiter=",", skiprows=0)

    def forward(self, x, m):

        # Choice 1: use analytical function as density u
        # ycal, par = self.u_dat_parameter_loop(x, m)

        # Choice 2: use discretized density u calculated by MATLAB function svd & quadprog
        u = self.u()
        ycal = self.forward_predict(m, x, u)
        return ycal

    def plotxy(self, n, x, y):
        x = x / (2 * max(abs(x)))
        y = y / (2 * max(abs(y)))
        plt.figure()
        plt.subplot(121)
        plt.title('x(k) & y(k)')
        plt.plot(n, x, label='x(k)')
        plt.scatter(n, y, label='y(k)')
        plt.legend(loc='lower left')
        plt.subplot(122)
        plt.plot(x, y)
        return

    # (obsolete)
    def everett(self, p, q, m, weight, alt):
        double_integral = 0  # the double sum (or double integral)
        for i in range(m):
            for j in range(m - i):
                double_integral = double_integral + weight[j][i] * alt[j][i]
        dy = np.sign(p - q) * double_integral  # p = x[k], q = x[k-1]
        # print(f'syt: dy={dy}')
        return dy

    # (obsolete)
    def fast_loop(self, p, q, m, ep, ne, ei, ej, sk, up):
        # reference: p212 of <Piezoelectric Sensors and Actuators> by Stefan Johann Rupitsch, 2019
        flag = False
        if p > q:  # if input x(k) increases
            pm = int((-p + 0.5) * m)  # mapping 0.5>=p>=-0.5 to 0<=i<=m-1
            for r in range(0, ne):
                if pm <= ei[r]:  # if Ture, some extrema are wiped out
                    ei = ei[:r]
                    ej = ej[:r]
                    sk = sk[:r]
                    # print(str(ne-r) + ' extrema wiped out by increasing input from ', q, ' to', p)
                    ne = r
                    flag = True
                    break
                if sk[-1] == -1 and pm < m - ej[-1]:
                    flag = True
                    break
            if flag:
                ei = np.append(ei, [pm])
                if ne >= 1:
                    ej = np.append(ej, [ej[-1]])
                else:
                    ej = np.append(ej, [0])
                sk = np.append(sk, [1])
                ne = ne + 1  # ne is the length of ei, ej
                # print('increase: ei: ', ei, '; ej: ', ej)

        elif p < q:  # if input x(k) decreases
            pm = int((p + 0.5) * m)  # mapping 0.5>=p>=-0.5 to 0<=i<=m-1
            for r in range(0, ne):
                if pm <= ej[r]:
                    ei = ei[:r]
                    ej = ej[:r]
                    sk = sk[:r]
                    # print(str(ne-r) + ' extrema wiped out by decreasing input from ', q, ' to', p)
                    ne = r
                    flag = True
                    break
                if sk[-1] == 1 and pm < m - ei[-1]:
                    flag = True
                    break
            if flag:
                ej = np.append(ej, [pm])
                if ne >= 1:
                    ei = np.append(ei, [ei[-1]])
                else:
                    ei = np.append(ei, [0])
                sk = np.append(sk, [-1])
                ne = ne + 1
                # print('decrease: ei: ', ei, '; ej: ', ej)

        else:  # p==q
            print('p==q happens')

        y = - ep[0][0] * sk[0]
        count = 0
        for n in range(0, ne):
            y = y + ep[ei[n]][ej[n]] * sk[n]
            count = count + 1
        return y, ne, ei, ej, sk, up

    # (obsolete) Calculate the epsilon matrix in advance
    def epsilon(self, m):
        u = self.u()

        plt.figure()
        plt.imshow(u)
        plt.xlabel('β')
        plt.ylabel('α')
        ep = np.zeros((m, m))
        for i in range(m):
            for j in range(m - i):
                for r in range(i, m):
                    for s in range(j, m - i):
                        ep[i][j] = ep[i][j] + u[r][s]
        return ep

    # (obsolete) Method 2: fast calculation of y(k) (unfinished: some of configurations are not considered in 'fast_cycle')
    def forward_fast(self, x, m):
        nn = len(x)
        plane, ne, ei, ej, sk = self.plane_initialization_half(m)
        ep = self.epsilon(m)
        ycal2 = np.zeros(len(x))
        ycal2[0] = 0
        up = True
        for k in range(1, nn):
            ycal2[k], ne, ei, ej, sk, up = self.fast_loop(x[k], x[k - 1], m, ep, ne, ei, ej, sk, up)
        ycal2 = ycal2 / (2 * max(abs(ycal2)))
        return ycal2

    # (obsolete) Inverse model: calculate input x(k) with output y(k)
    def back_loop(self, yp, yq, m, q, ep, ne, ei, ej, sk, dy_error):
        # reference: p246 of <Piezoelectric Sensors and Actuators> by Stefan Johann Rupitsch, 2019
        # unfinished
        dy = yp - yq
        if dy == 0 or abs(dy) < abs(dy_error):
            p = q
        else:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            dy1 = dy + ep[ei[-1]][ej[-1]] * sk[-1]  # dy1: dy'
            # first: coarse search
            low = 0
            dy2 = 0
            n = -1
            for v in range(ne - 1, 0, -1):  # go backwards(v=ne-1,ne-2,...,n,...,2,1) to find an index n
                low = low + ep[ei[v]][ej[v]] * sk[v]
                if abs(low) < abs(dy1):
                    # high = low + ep[ei[v - 1]][ej[v - 1]] * sk[v - 1]
                    high = low - ep[ei[v]][ej[v]] * sk[v]
                    if abs(high) <= abs(low):
                        print(ep[ei[v]][ej[v]] * sk[v], 'error:|high|<=|low|, n is not found')
                    if abs(dy1) < abs(high):
                        n = v
                        # print(f'n is found, n={n}')
                        dy2 = dy1 + low  # dy2: dy"
                        ei = ei[:n]  # delete entries ne-1,...,n of ei,ej,sk (note: ne-1>=n, v==n)
                        ej = ej[:n]
                        sk = sk[:n]
                        # print(f'{ne-n} extrema are deleted')
                        ne = n
                        break  # the index n is found
            if dy2 == 0:
                print('error: dy"==0')
            if n == -1:
                print('n not found')
                # exit(-3)
            # second: detailed search: along the column j=j_ne-1 (for dy>0) or i=i_ne-1 (for dy<0)
            ir = js = n - 1
            # print(f'ne={ne}')
            if dy > 0:
                mm = abs(dy2 - ep[n][ej[-1]])
                for i in range(n - 1, ei[-1], -1):
                    mi = abs(dy2 - ep[i][ej[-1]])
                    if mm > mi:
                        mm = mi
                        ir = i  # index ir gives the minimum of |dy2-ep[ei[ir]][ej[jm]]|
                # print(f'ir={ir}')
                # calculate the predicted input
                p = (m-ir)/(m-1) - 0.5

            elif dy < 0:
                mm = abs(dy2 - ep[n][ej[-1]])
                for j in range(n - 1, ej[-1], -1):
                    mj = abs(dy2 - ep[j][ej[-1]])
                    if mm > mj:
                        mm = mj
                        js = j  # index js gives the minimum of |dy2-ep[ei[im]][ej[js]]|
                # print(f'js={js}')
                p = 0.5 - (m - js) / (m - 1)

            dy_error = ep[ei[ir]][ej[js]] * sk[-1] + dy2
            print(dy, dy_error)

        return p, ne, ei, ej, sk, dy_error

    # (obsolete)
    def backward(self, y, m, x):
        nn = len(y)
        ep = self.epsilon(m)
        plane = self.plane_initialization_half(m)
        ne, ei, ej, sk = self.dividing_line_initialization_half(m)
        xinv = np.zeros(len(y))
        xinv[0] = 0
        dy_error = 0
        for k in range(1, nn):
            xinv[k], ne, ei, ej, sk, dy_error = self.back_loop(y[k], y[k - 1], m, x[k - 1], ep, ne, ei, ej, sk, dy_error)
        return xinv

