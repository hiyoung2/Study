import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
X = dataset.data
Y = dataset.target

print("X.shape :", X.shape) # (442, 10)
print("Y.shape :", Y.shape) # (442,)

# pca = PCA(n_components = 5)
# X_pca = pca.fit_transform(X)
# pca_evr = pca.explained_variance_ratio_
# print(pca_evr) 
# print(sum(pca_evr)) 

pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_) 
print(cumsum)
# [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
#  0.94794364 0.99131196 0.99914395 1.        ]
# 이 수치가 완벽한 것은 아니다, 고려해서 반영은 가능하다

np.argmax(cumsum >= 0.94)
print(cumsum >= 0.94) # [False False False False False False  True  True  True  True]

a = np.argmax(cumsum >= 0.94) + 1 # 0부터 세므로
print(a) # 7 # n_components에 쓰기 위해서는 0부터 시작하는 인덱스 개념 때문에 + 1을 해줘야 한다