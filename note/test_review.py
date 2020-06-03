'''
06월 02일 삼성전자 주가 예측 테스트 대략적인 정리

1 . 데이터 read : index, header, comma, encoding 처리

2. head, tail 등으로 데이터 확인

3. 결측치 NAN 제거, 보충
- dropna
- fillna(method = '') : bfill, ffill, mean() 등등
- iloc, loc 중요

4. 내림차순 또는 오름차순으로 데이터 정렬하기
- ascending = [True] : 오름차순 or [False] : 내림차순

5. 문자열로 이뤄진 데이터를 정수형으로 바꿔주기(for문 사용)

6. npy 파일로 저장하기
- df=df.values : 인덱스나 헤더를 제외한 실제 수치들로만 데이터로 만든 후 저장해야 한다
- np.save('/위치/파일명, arr = df)

7. 필요한 것들을 import (코드 상단에 적어주는 버릇을 들이자)

8. 데이터를 자르는데 사용할 Split 함수 정의

9. 데이터 load, 불러오기

10. Data shape 확인하기
- 이 때, Column = 1 인 데이터는 미리 Vector 형태로 reshape 해주자, 다음 단계들에서 shape 문제들을 다루는 데 좀 편해진다

11. split 함수로 시계열 데이터에 맞게끔 시간 순서를 반영한 데이터로 만들어준다

12. split 함수로 만들어진 datset 중 x, y data로 나눠야 하는 것을 slicing 을 통해서 만들어준다

13. scaler를 사용해서 데이터 전처리를 한다

14. pca는 scaler를 사용한 후에 쓰도록 하자
- pca를 쓰는 데이터는 split 함수를 마지막에 적용시켜야 한다
- split 후에 pca를 하면 중복값들이 포함되기 때문에?

15. shape를 확인하고 ensemble로 짜기 위해 입력 data들의 '행'을 꼭 맞춰줘야 한다
- 주식 같은 경우에는 가장 오래된 날짜들의 데이터를 제거하면 좋다
- 어떤 레이어를 쓸 지에 따라 shape를 잘 맞춰준다

16. train_test_split을 해 준다

17. compile, fit, evaluate, predict 과정 실행
- predict 과정에서 shape 문제가 발생할 수 있다
- predict 에 이용할 데이터의 shape를 확인해주고 reshape를 통해 해결하자

'''