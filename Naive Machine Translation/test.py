import numpy as np
from main import nearest_neighbors, get_matrices, en_fr_test, fr_embedding_subset, en_embedding_subset, R_train

def test_vocabulary(X,Y,R, nearest_neighbor = nearest_neighbors):
    pred = np.dot(X,R)
    num_correct = 0
    for i in range(len(pred)):
        pred_idx = nearest_neighbor(pred[i], Y)
        if pred_idx == i:
            num_correct += 1
    accuracy = num_correct / len(pred)
    return accuracy

X_val, Y_val = get_matrices(en_fr_test, fr_embedding_subset, en_embedding_subset)
accuracy = test_vocabulary(X_val, Y_val, R_train)
print(f'accuracy on test set is {accuracy:.3f}')