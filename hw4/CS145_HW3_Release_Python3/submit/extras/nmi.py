import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score

predictions = [[10,12,14,16,18], [1,2,7,8,15,17], [3,4,5,9,13], [6,11,19,20]]
truths = [[3,4,5,13,17], [10,12,14,16,18], [1,2,7,8,15], [6,9,11,19,20]]

p_labels = [2, 2, 3, 3, 3, 4, 2, 2, 3, 1, 4, 1, 3, 1, 2, 1, 2, 1, 4, 4]
w_labels = [3, 3, 1, 1, 1, 4, 3, 3, 4, 2, 4, 2, 1, 2, 3, 2, 1, 2, 4, 4]

def I(predictions, truths):
    result = 0
    for k in range(len(predictions)):
        for j in range(len(truths)):
            print(list(set(predictions[k]) & set(truths[j])))
            a = len(list(set(predictions[k]) & set(truths[j])))
            b = (20 * a) / (len(predictions[k]) * len(truths[j]))
            if b == 0:
                result += (a/20)
            else:
                result += (a/20) * np.log(b)
    return result

def H(W):
    result = 0
    for j in range(len(W)):
        a = len(W[j])/20
        result -=  a * np.log(a)
    return result

i = I(predictions, truths)
hC = H(predictions)
hW = H(truths)
nmi = i/(np.sqrt(hC * hW))
skl_nmi = normalized_mutual_info_score(p_labels, w_labels)

print("\nI: ", i)
print("hC: ", hC)
print("hW: ", hW)
print("My NMI: ", nmi)
# Compare with sklearn's NMI calculation
print("sklearn NMI: ", skl_nmi)
