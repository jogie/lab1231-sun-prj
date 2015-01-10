import numpy as np

def KLD(prob_p, prob_q):
    KLD_result = 0
    for i in range(len(prob_q)):
        KLD_result += prob_p[i] * (np.log2(prob_p[i] / prob_q[i]))
    return KLD_result

p = [float(1.0/8), float(1.0/8), float(1.0/4), float(1.0/2)]
q = [float(1.0/4), float(1.0/4), float(1.0/4), float(1.0/4)]
print KLD(p, q)