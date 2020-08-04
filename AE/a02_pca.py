import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
X = dataset.data
Y = dataset.target

print("X.shape :", X.shape) # (442, 10)
print("Y.shape :", Y.shape) # (442,)

pca = PCA(n_components = 5)
X_pca = pca.fit_transform(X)
pca_evr = pca.explained_variance_ratio_
print(pca_evr) # [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856] # 압축한 컬럼들의 중요도 비율
print(sum(pca_evr)) # 0.8340156689459766 # 1이 아니다? # n_components = 10이면 1, 5이면 1이 안 됨 # 80% 정도로 압축이 되었다(나머지는 압축과정에서 손실 된 것)

