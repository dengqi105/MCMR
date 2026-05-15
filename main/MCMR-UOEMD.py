import numpy as np
import os
from scipy.io import loadmat
import time
from Models.MCMR import MCMR

# %% parameters setting
print("****" * 10, "parameters setting", "****" * 10)
fs = 42000  # sampling frequency
sensor_index = [0, 1, 2, 3, 4]
K = 5
L = 2800  # sampele length
kernel_size = 45

T = 10
step = int(L / 2)
nums = int(fs * T / step) - 1  # sample number for each class

print("K:", K)
print("L:", L)
print("S:", len(sensor_index))
print("kernel_size:", kernel_size)
print("****" * 25)

# %% LOAT DATA #time-varing condition
path = "./dataset/1_Unloaded_Condition" #2_Loaded
fault_model = ["H_H", "R_U", "R_M", "S_W", "V_U", "B_R", "K_A", "F_B"]

datas = []
for i in range(len(fault_model)):
    dataset = loadmat(os.path.join(path, "{}_1_0.mat".format(fault_model[i])))["data"][:T * fs, sensor_index]#15&15to45Hz
    print("dataset shape:", dataset.shape)
    samples = []
    for j in range(nums):
        sample = dataset[j * step:j * step + L]
        samples.append(sample)
    samples = np.stack(samples)
    datas.append(samples)
datas = np.row_stack(datas).swapaxes(1, 2)
all_data = datas

# %%
label = [0, 1, 2, 3, 4, 5, 6, 7]


def get_label(label):
    labels = []
    for i in range(len(label)):
        labels = labels + [i] * nums
    labels = np.array(labels)
    return labels


labels = get_label(label)

corrs = ["Correntropy", "Euclidean", "Covariance", "Correlation", "Minkowski", "Cosine"]  # correlation methods
d_methods = ["AFD", "EWT", "VMD"]  # dataset agumentation methods
clf_names = ["RR", "NB", "LR", "LDA", "LSVM", "GMSVM"]  # classifiers

corr = corrs[1] # For this dataset,we use Euclidean to avoid tuning kernel_size
d_method = d_methods[0]
clf = clf_names[0]

print("correlation method:", corr)
print("dataset agumentation method:", d_method)
print("classifier:", clf)

# %% LOAD MODEL
start_time = time.time()
Extractor = SFCE(fs=fs, K=K, kernel_size=kernel_size, clf=clf, corr=corr, d_method=d_method, z_score=True)
corr_f, corr_f1, corr_f2 = Extractor.transform(all_data)
end_time = time.time()
print("time:", end_time - start_time)

# %% TRAIN AND TEST

train_number = 2 # for each class
repeat_num = 20

print("train_number:", train_number)

Acc = np.zeros(repeat_num)
Acc1 = np.zeros(repeat_num)
Acc2 = np.zeros(repeat_num)
for i in range(repeat_num):
    train_index = np.random.choice(nums, train_number, replace=False).astype(int)
    train_id = []
    for j in train_index:
        for k in range(len(label)):
            train_id.append(nums * k + j)

    test_id = list(set(list(np.arange(all_data.shape[0]))) - set(train_id))

    train_label = labels[train_id]
    test_label = labels[test_id]

    train_f, test_f = corr_f[train_id, :], corr_f
    train_f1, test_f1 = corr_f1[train_id, :], corr_f1
    train_f2, test_f2 = corr_f2[train_id, :], corr_f2

    predicted_label = Extractor.fit_predict(train_f, train_label, test_f)
    predicted_label1 = Extractor.fit_predict(train_f1, train_label, test_f1)
    predicted_label2 = Extractor.fit_predict(train_f2, train_label, test_f2)

    Acc[i] = np.sum(test_label == predicted_label[test_id]) / len(test_label) * 100
    Acc1[i] = np.sum(test_label == predicted_label1[test_id]) / len(test_label) * 100
    Acc2[i] = np.sum(test_label == predicted_label2[test_id]) / len(test_label) * 100

print('Acc:{}'.format(np.round(np.mean(Acc), 2)), 'Std:{}'.format(np.round(np.std(Acc), 2)))
print('Inner_Acc:{}'.format(np.round(np.mean(Acc1), 2)), 'Std:{}'.format(np.round(np.std(Acc1), 2)))
print('Intra_Acc:{}'.format(np.round(np.mean(Acc2), 2)), 'Std:{}'.format(np.round(np.std(Acc2), 2)))
