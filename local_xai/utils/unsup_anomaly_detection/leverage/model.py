import numpy as np


class Leverage():
    def __init__(self):
        self.name= 'Leverage'

    def detect(self, encoded_features, trace_lens):
        encoded_features = encoded_features.reshape((encoded_features.shape[0], 
                                                     np.prod(encoded_features.shape[1:])))
        XE = np.matrix(encoded_features)

        HE = XE * np.linalg.pinv(XE.T * XE) * XE.T
        l = np.diagonal(HE)

        N = trace_lens 

        Z = (N - np.mean(N)) / np.std(N)

        sigZ = 1 / (1 + np.exp(-Z))

        if np.max(N) > 2.2822 / 0.3422:
            cNmax = -2.2822 + np.power(np.max(N), 0.3422)
        else:
            cNmax = 0

        w = np.power((1 - sigZ), cNmax)

        trace_level_abnormal_scores = w * l

        return trace_level_abnormal_scores, None, None