predictions = [2, 2, 3, 3, 3, 4, 2, 2, 3, 1, 4, 1, 3, 1, 2, 1, 2, 1, 4, 4]
truths = [3, 3, 1, 1, 1, 4, 3, 3, 4, 2, 4, 2, 1, 2, 3, 2, 1, 2, 4, 4]

assert(len(predictions) == len(truths))

tp = 0
fp = 0
tn = 0
fn = 0

# For every tuple
for i in range(len(predictions)):
    for j in range(len(truths)):
        if j <= i:
            continue
        else:
            # Same class, same cluster
            if truths[i] == truths[j] and predictions[i] == predictions[j]:
                tp += 1
            # Different class, same cluster
            elif truths[i] != truths[j] and predictions[i] == predictions[j]:
                fp += 1
            # Same class, different cluster
            elif truths[i] == truths[j] and predictions[i] != predictions[j]:
                fn += 1
            # Different class, different cluster
            elif truths[i] != truths[j] and predictions[i] != predictions[j]:
                tn += 1

print("TP: ", tp)
print("FP: ", fp)
print("TN: ", tn)
print("FN: ", fn)
print("Total: ", tp+fp+tn+fn)