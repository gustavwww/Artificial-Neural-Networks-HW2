import math

import pandas as pd
import numpy as np
import random as rnd

class Network:

    def __init__(self, M, learning_rate):
        self.M = M
        self.ln = learning_rate
        self.w_1 = 0.1*(np.random.normal(0, 30, size=(M, 2)))
        self.w_2 = 0.1*(np.random.normal(0, 30, size=M))
        self.thetaV = np.zeros(M)
        self.thetaO = 0
        self.v = np.zeros(M)
        self.b_v = np.zeros(M)


    def forwardIterate(self, input):
        # Update hidden neurons
        for j in range(0, self.M):
            b_v = 0
            for k in range(0, 2):
                b_v += self.w_1[j][k] * input[k]
            b_v -= self.thetaV[j]
            self.b_v[j] = b_v
            self.v[j] = math.tanh(b_v)

        # Calculate output
        b_o = 0
        for j in range(0, self.M):
            b_o += self.w_2[j] * self.v[j]
        b_o -= self.thetaO

        output = math.tanh(b_o)
        return output, b_o

    def validate(self, data, t):
        nr_patterns = len(data)
        C = 0
        for p in range(0, nr_patterns):
            output, b_0 = self.forwardIterate(data[p])

            if output < 0:
                sgn_o = -1
            else:
                sgn_o = 1

            C += abs(sgn_o - t[p])

        C *= 1/(2*nr_patterns)
        return C


    def trainNetwork(self, data, t, k):

        for iter in range(0, k):
            print(iter)
            p = rnd.randint(0, len(data)-1)
            output, b_o = self.forwardIterate(data[p])

            # Calc update for w_2 and pattern p
            error_w2 = (t[p] - output) * 1/pow(math.cosh(b_o), 2)
            self.thetaO += -self.ln * error_w2

            for i in range(0, self.M):
                dw_2 = self.ln * error_w2 * self.v[i]
                self.w_2[i] += dw_2

            # Calc update for w_1 and pattern p
            for i in range(0, self.M):
                error_w1_i = error_w2 * self.w_2[i] * 1/pow(math.cosh(self.b_v[i]), 2)
                self.thetaV[i] += -self.ln * error_w1_i

                for j in range(0, 2):
                    dw_1 = self.ln * error_w1_i * data[p][j]
                    self.w_1[i][j] += dw_1


training = pd.read_csv("files/training_set.csv")
validation = pd.read_csv("files/validation_set.csv")

data_training = training.iloc[:, [0,1]].to_numpy()
t_training = training.iloc[:, 2].to_numpy()

data_validation = validation.iloc[:, [0,1]].to_numpy()
t_validation = validation.iloc[:, 2].to_numpy()

M = 10
learning_rate = 0.03

network = Network(M, learning_rate)

C = 1
while C >= 0.12:
    network.trainNetwork(data_training, t_training, 10000)
    C = network.validate(data_validation, t_validation)
    print("Epoch finished: C = " + str(C))

pd.DataFrame(network.w_1).to_csv("files/w1.csv")
pd.DataFrame(network.w_2).to_csv("files/w2.csv")
pd.DataFrame(network.thetaV).to_csv("files/t1.csv")
pd.DataFrame([network.thetaO]).to_csv("files/t2.csv")

print(C)