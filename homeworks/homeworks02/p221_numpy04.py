# 종합 문제
# NumPy 지식 이용, 두 이미지의 차이를 계산
# 0 ~ 5 사이의 정수를 색으로 가정
# 이미지는 2차원 데이터라서 행렬로 나타낼 수 있으며, NumPy 배열로 처리 가능
# 크기가 동일한 이미지의 차이는 같은 위치에 있는 요소 칸의 차이를 성분으로 하는 행렬로 표시 할 수 있다

# 난수로 지정한 크기의 이미지를 생성하는 함수 make_image()를 완성하라
# 전달된 행렬의 일부분을 난수로 변경하는 함수 change_matrix()를 완성
# 생성된 image1과 image2의 각 요소의 차이의 절댓값을 계산, image3에 대입

import numpy as np

np.random.seed(0)

def make_image(m, n) :
    image = np.random.randint(0, 6, (m, n))
    return image

def change_little(matrix) :
    shape = matrix.shape
    for i in range(shape[0]) :
        for j in range(shape[1]) :
            if np.random.randint(0, 2) == 1 :
                matrix[i][j] = np.random.randint(0, 6, 1)
    return matrix


image1 = make_image(3, 3)
print(image1)
print()

image2 = change_little(np.copy(image1))
print(image2)
print()

image3 = image2 - image1
print(image3)
print()

image3 = np.abs(image3)
print(image3)