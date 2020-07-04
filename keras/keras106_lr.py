# lr이 너무 낮으면 gradient vanishing , 소실 문제가 발생
# lr이 높으면 gradient exploding
# 방지하는 방법을 알아보자 


weight = 0.5
input = 0.5
goal_prediction = 0.8

lr = 0.001
# 1
# 0.1
# 0.01
# 0.0001

# 0.5를 넣어서 0.8을 찾아가는 과정
for iteration in range(1101) :
    prediction = input * weight # y = wx
                                # 0.5 * 0.5 = 0.25 = prediction
    error = (prediction - goal_prediction) ** 2 # loss
                                                # error = (0.25 - 0.8)의 제곱 = 0.3025 

    print("Error :", + str(error) + "\tPrediction :" + str(prediction))

    # goal_prediction을 찾아가다가 너무 높게 잡아서 지나치는 기준
    up_prediction = input * (weight + lr)
    up_error = (goal_prediction - up_prediction) ** 2

    # goal_prediction을 찾아가다가 너무 낮게 잡아서 지나치는 기준
    down_prediction = input *(weight - lr)
    down_error = (goal_prediction - down_prediction) ** 2

    # 그 기준이 너무 크다면 weight에서 -lr 하여 다시 goal_prediction을 찾아가라
    if (down_error < up_error) :
        weight = weight - lr

    # 그 기준이 너무 작다면 weight에서 +lr 하여 다시 goal_prediction을 찾아가라
    if (down_error > up_error) :
        weight = weight + lr