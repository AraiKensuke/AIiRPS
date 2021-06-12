import random
import numpy as np
import itertools

class MarkovChain():
    def __init__(self, order, decay):
        self.decay = decay
        self.matrix = self.create_matrix(order)
    @staticmethod
    def create_matrix(order):
        def create_keys(order):
            keys = ['1', '2', '3']
            for i in range(order*2-1):
                keyLen = len(keys)
                for i in itertools.product(keys, ''.join(keys)):
                    keys.append(''.join(i))
                keys = keys[keyLen:]
            return keys
        #these are RR, RP, RS, PR, ..., PS
        keys = create_keys(order)
        #R,P,S for each key
        matrix = {}
        for key in keys:
            matrix[key] = {'1': {'prob' : 1/3,
                                 'n_obs' : 0
                                },
                           '2': {'prob' : 1/3,
                                 'n_obs' : 0
                                },
                           '3': {'prob' : 1/3,
                                 'n_obs' : 0
                                }
                            }
        return matrix
    def update_matrix(self, pair, p_input):
        #change amount of observations remembered for each key depending on decay
        for i in self.matrix[pair]:
            self.matrix[pair][i]['n_obs'] = self.decay * self.matrix[pair][i]['n_obs']
        self.matrix[pair][p_input]['n_obs'] += 1
        n_total = 0
        for i in self.matrix[pair]:
            n_total += self.matrix[pair][i]['n_obs']
        #probabilities: player plays x hand y percent of the time given machine/player choice pair
        for i in self.matrix[pair]:
            self.matrix[pair][i]['prob'] = self.matrix[pair][i]['n_obs'] / n_total
    def predict(self, pair):
        #draw a uniform sample and choose corresponding hand
        probs = self.matrix[pair]
        ### Method: MAX
        # maxP = 0.0
        # minP = 1.1
        # maxKey = ''
        # for i in probs:
        #     if probs[i]['prob'] > maxP: 
        #         maxP = probs[i]['prob']
        #         maxKey = i
        #     if probs[i]['prob'] < minP: minP = probs[i]['prob']
        # if maxP == minP:
        #     return random.choice(['1', '2', '3'])
        #     #return '1'
        # else:
        #     return maxKey
        ### Method: SOFTMAX
        # sample = np.random.uniform()
        # r_prob = probs['1']['prob']
        # p_prob = probs['2']['prob']
        # if (sample < r_prob): return '1'
        # if (sample >= r_prob) and (sample < r_prob + p_prob): return '2'
        # if (sample >= r_prob + p_prob): return '3'
        ### Method: SQUARED
        C = (probs['1']['prob'])**2 + (probs['2']['prob'])**2 + (probs['3']['prob'])**2
        r_prob = ((probs['1']['prob'])**2)/C
        p_prob = ((probs['2']['prob'])**2)/C
        s_prob = ((probs['3']['prob'])**2)/C
        print("Probabilities for round (pR, pP, pS):")
        print([r_prob, p_prob, s_prob])
        print()
        sample = np.random.uniform()
        if (sample < r_prob): return '1'
        if (sample >= r_prob) and (sample < r_prob + p_prob): return '2'
        if (sample >= r_prob + p_prob): return '3'
