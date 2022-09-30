import math

import numpy as np
import random as rnd
import matplotlib.pyplot as plt

class RestrictedBoltzmann:

    def __init__(self, hidden_neurons, k, learning_rate):
        self.N = 3
        self.M = hidden_neurons
        self.ln = learning_rate
        self.k = k


    def _initWeightsAndTheta(self):
        self._weights = np.zeros((self.M, self.N))
        self._thetaV = np.zeros(self.N)
        self._thetaH = np.zeros(self.M)
        return

    def _sample_patterns(self, n):
        patterns = []
        for i in range(0, n):
            val = rnd.randint(0,3)
            str = ""
            if val == 0:
                str = "000"
            elif val == 1:
                str = "101"
            elif val == 2:
                str = "011"
            elif val == 3:
                str = "110"

            patterns.append(str)

        return patterns

    def _initVisible(self, pattern):
        self._visible = np.zeros(self.N)
        self._visible_0 = np.zeros(self.N)
        self._hidden = np.zeros(self.M)

        self._b_h = np.zeros(self.M)
        self._b_v = np.zeros(self.N)
        self._b_h_0 = np.zeros(self.M)
        self._b_v_0 = np.zeros(self.N)

        for i in range(0, self.N):
            self._visible[i] = pattern[i]
            self._visible_0[i] = pattern[i]

    def _updateHidden(self, isFirst):
        for h in range(0, self.M):
            b_h = 0
            for i in range(0, self.N):

                b_h += self._weights[h][i] * self._visible[i]

            self._b_h[h] = b_h - self._thetaH[h]

            if isFirst:
                self._b_h_0[h] = self._b_h[h]

            # Update with certain probability.
            prob = 1/(1 + math.exp(-2 * b_h))
            if rnd.random() < prob:
                self._hidden[h] = 1
            else:
                self._hidden[h] = -1

    def _updateVisible(self, isFirst):
        for v in range(0, self.N):
            b_v = 0
            for i in range(0, self.M):
                b_v += self._weights[i][v] * self._hidden[i]

            self._b_v[v] = b_v - self._thetaV[v]

            if isFirst:
                self._b_v_0[v] = self._b_v[v]

            # Update with certain probability.
            prob = 1/(1 + math.exp(-2 * b_v))
            if rnd.random() < prob:
                self._visible[v] = 1
            else:
                self._visible[v] = -1

    def _adjustNetwork(self):

        for m in range(0, self.M):
            for n in range(0, self.N):
                dw = self.ln * (math.tanh(self._b_h_0[m])*self._visible_0[n] - math.tanh(self._b_h[m])*self._visible[n])
                self._weights[m][n] += dw

        for n in range(0, self.N):
            self._thetaV[n] -= self.ln * (self._visible_0[n] - self._visible[n])

        for m in range(0, self.M):
            self._thetaH[m] -= self.ln * (math.tanh(self._b_h_0[m]) - math.tanh(self._b_h[m]))


    def train_cd_k_alg(self):
        # init weights, threshold.
        self._initWeightsAndTheta()
        for v in range(0, self.N):
            p_0 = 4
            patterns = self._sample_patterns(p_0)
            for p in range(0, p_0):
                self._initVisible(patterns[p])
                self._updateHidden(True)

                # Calculate model distribution
                for t in range(0, self.k):
                    self._updateVisible(t == 0)
                    self._updateHidden(False)
                    self._adjustNetwork()


            # Update wights and thresholds
            for _ in range(0, p_0):
                self._adjustNetwork()


    def _visibleToBin(self, value):
        if value < 0:
            return_val = 0
        else:
            return_val = 1

        return return_val


    def sample_model_dist(self, n):
        # 000 101 011 110
        pattern_counts = [0 ,0 ,0 ,0]
        for _ in range(0, n):
            for _ in range(0, self.k):
                self._updateVisible(False)
                self._updateHidden(False)

            val0 = int(self._visibleToBin(self._visible[0]))
            val1 = int(self._visibleToBin(self._visible[1]))
            val2 = int(self._visibleToBin(self._visible[2]))

            pattern = str(val0) + str(val1) + str(val2)

            if pattern == "000":
                pattern_counts[0] += 1
            elif pattern == "101":
                pattern_counts[1] += 1
            elif pattern == "011":
                pattern_counts[2] += 1
            elif pattern == "110":
                pattern_counts[3] += 1

        return pattern_counts


def calcRatio(arr):
    total = 0
    ratios = []
    for i in range(0, len(arr)):
        total += arr[i]

    for i in range(0, len(arr)):
        ratios.append(arr[i]/total)

    return ratios

def calc_Kullback_Leibler(p_model):
    kl = 0
    for i in range(0, len(p_model)):
        kl += 1/4 * math.log((1/4) / (p_model[i]))

    return kl

def calc_kullback_upper_bounds(M):
    return_bound = []
    for i in range(0, len(M)):
        m = M[i]
        if m < pow(2, 2) - 1:
            return_bound.append(3 - math.log(m + 1, 2) - (m + 1)/pow(2, math.log(m+1, 2)))
        else:
            return_bound.append(0)

    return return_bound

M = [1,2,4,8]
k = 100
ln = 0.5

all_kl = []

for i in range(0, len(M)):
    network = RestrictedBoltzmann(M[i], k, ln)
    network.train_cd_k_alg()

    counts = network.sample_model_dist(1000)
    ratios = calcRatio(counts)
    kl = calc_Kullback_Leibler(ratios)
    all_kl.append(kl)

    print("Calculate for M = " + str(M[i]))
    print("Output counts: (000 101 011 110) " + str(counts))
    print("Ratios: " + str(ratios))
    print("")

print("Kullback-Leibler values: " + str(all_kl))



plt.plot(M, all_kl, 'bo')
plt.plot(M, calc_kullback_upper_bounds(M))
plt.title("The Kullback-Leibler divergence versus number of hidden neurons")
plt.xlabel("M")
plt.ylabel("D_KL")
plt.show()