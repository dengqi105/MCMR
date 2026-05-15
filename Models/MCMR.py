# -*- coding:utf-8 -*-
# @FileName  :MCMR.py
# @Time      :2026/5/15/11:34
# @Author    :dengqi

import numpy as np
from scipy.fftpack import dct, idct, fft, ifft
from scipy.linalg import logm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV,RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
# from vmdpy import VMD
# import ewtpy

estimators = {'RR': RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)), 'NB': GaussianNB(),
              "LR": LogisticRegression(), "LDA": LinearDiscriminantAnalysis(),
              'LSVM': SVC(kernel='linear'), 'GMSVM': SVC(kernel='precomputed'),
              }



class MCMR:
    def __init__(self, fs, K, kernel_size, d_method="AFD", corr="Correntropy", clf="RR", z_score=True):
        self.fs = fs
        self.K = K
        self.corr = corr
        self.kernel_size = kernel_size
        self.clf = clf
        self.d_method = d_method
        self.z_score = z_score

    def get_corr(self, X, Y, corr, kernel_size):
        if corr == "Covariance":
            corr_value = np.cov(X, Y)[0, 0]
        if corr == "Minkowski":
            corr_value = distance.minkowski(X, Y)
        if corr == "Correlation":
            corr_value = distance.correlation(X, Y)
        if corr == "Euclidean":
            corr_value = distance.euclidean(X, Y)
        if corr == "Cosine":
            corr_value = distance.cosine(X, Y)
        if corr == "Correntropy":
            corr_value = np.exp(-np.sqrt(np.sum((X - Y) ** 2)) / kernel_size)
        return corr_value

    def get_imfs(self, X, d_method):
        K, corr, kernel_size, fs = self.K, self.corr, self.kernel_size, self.fs
        if d_method == "AFD":
            FC = [0]
            [FC.append(fs / (2 ** (K - i))) for i in range(K)]
            imfs = self.AFD(X, fs, fc=FC)
        if d_method == "VMD":
            imfs, imfs_hat, omega = VMD(X, alpha=40000, tau=0., K=K, DC=0, init=1, tol=1e-7)
            imfs = imfs.T
        if d_method == "EWT":
            imfs, mfb, boundaries = ewtpy.EWT1D(X.squeeze(), N=K)
        return imfs

    def AFD(self, X, fs, fc, filter_type='dct'):
        N = X.shape[0]
        fc = np.sort(fc)
        if filter_type == 'dct':
            dct_type = 2
            K = np.round(2 * N * fc / fs).astype(int)
            no_of_subbands = K.shape[0] - 1
            Hk = np.zeros((N, 1, no_of_subbands))
            for i in range(no_of_subbands):
                Hk[K[i]: K[i + 1], :, i] = 1
            Xk = dct(X, type=dct_type, n=N, axis=0, norm='ortho', overwrite_x=False)
            Yk = np.einsum('ij,ijk->ijk', Xk, Hk)
            Y = idct(Yk, type=dct_type, n=N, axis=0, norm='ortho', overwrite_x=False)
            FIMFs = np.squeeze(Y, axis=1)
        if filter_type == 'dft':
            append_ratio = 0.02
            pad_rows = int(N * append_ratio)
            X_padded = np.pad(X, pad_width=((pad_rows, pad_rows), (0, 0)), mode="symmetric")
            appended_length = X_padded.shape[0]
            L = appended_length
            N_fft = 2 * np.ceil(L / 2).astype(int)
            K = np.round(N_fft * fc / fs).astype(int)
            no_of_subbands = K.shape[0] - 1
            Hk = np.zeros((N_fft, 1, no_of_subbands))
            for i in range(no_of_subbands):
                Hk[K[i]: K[i + 1], :, i] = 1
                Hk[N_fft - K[i + 1]: N_fft - K[i], :, i] = 1
            Xk = 1 / L * fft(X_padded, n=N_fft, axis=0)
            Yk = np.einsum('ij,ijk->ijk', Xk, Hk)
            Y = L * ifft(Yk, n=N_fft, axis=0, overwrite_x=False)
            Y = np.real(Y)
            Y = Y[pad_rows + 1: appended_length - pad_rows + 1, :, :]
            FIMFs = np.squeeze(Y, axis=1)
        return FIMFs


    def corr_matrix(self, data, corr, kernel_size):
        dim = data.shape[0]
        corr_M = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(i + 1):
                corr_M[i, j] = self.get_corr(data[i, :], data[j, :], corr, kernel_size)
        corr_M = (corr_M + corr_M.T) - np.eye(np.shape(corr_M)[0]) * np.tile(np.diag(corr_M), (np.shape(corr_M)[1], 1))
        return corr_M

    def get_corr_matrix(self, sensor_seg, K, corr, kernel_size):
        tol = 1e-3
        shape = np.shape(sensor_seg)
        all_samples = np.zeros((shape[0], K, K), dtype=np.float32)
        for i in range(shape[0]):
            K_segments = sensor_seg[i, :, :]
            corr_M = self.corr_matrix(K_segments, corr, kernel_size)
            eye = np.eye(np.shape(corr_M)[0])
            all_samples[i, :, :] = logm(corr_M + tol * eye * np.trace(corr_M))
        return all_samples

    def Z_score(self, all_data):
        pre_data = np.zeros(all_data.shape)
        for i in range(all_data.shape[1]):
            pre_data[:, i, :] = StandardScaler().fit_transform(all_data[:, i, :].T).T
        return pre_data

    def transform(self, X):
        K, corr, kernel_size, fs, d_method = self.K, self.corr, self.kernel_size, self.fs, self.d_method
        E, S, L = X.shape

        if self.z_score:
            X = self.Z_score(X)

        all = np.zeros((E, S, K + 1, L), dtype=np.float32)
        for e in range(E):
            for s in range(S):
                sensor_seg = X[e, s:s + 1, :].transpose()
                p = np.zeros((L, K + 1), dtype=np.float32)
                p[:, :K] = self.get_imfs(sensor_seg, d_method=d_method)
                p[:, K:K + 1] = sensor_seg
                all[e, s, :, :] = p.transpose()
        K += 1
        SVCMs = np.zeros((E, K * K, S))
        SACMs = np.zeros((E, S * S, K))
        for s in range(S):
            innner_data = all[:, s, :, :]
            all_M = self.get_corr_matrix(innner_data, K, corr=corr, kernel_size=kernel_size)
            corr_feature = np.reshape(all_M, (E, K * K))
            SVCMs[:, :, s] = corr_feature
        intra_features = np.reshape(SVCMs, (E, K * K * S), order="F")
        for k in range(K):
            intra_data = all[:, :, k, :]
            all_M = self.get_corr_matrix(intra_data, S, corr=corr, kernel_size=kernel_size)
            corr_feature = np.reshape(all_M, (E, S * S))
            SACMs[:, :, k] = corr_feature
        inner_features = np.reshape(SACMs, (E, S * S * K), order="F")

        features = np.hstack((inner_features, intra_features))
        return features, inner_features, intra_features

    def fit_predict(self, X_train, y_train, X_test):
        estimator = estimators[self.clf]
        if self.clf == "GMSVM":
            G_Tr = np.dot(X_train, X_train.T).T
            G_Te = np.dot(X_train, X_test.T).T
            estimator.fit(G_Tr, y_train)
            predicted_label = estimator.predict(G_Te)
        else:
            estimator.fit(X_train, y_train)
            predicted_label = estimator.predict(X_test)
        return predicted_label

