# 14.1 CSV

# 14.1.1 Pandas로 CSV 읽기
# 이 장에서는 CSV(comma-separated values)라는 데이터 형식을 취급
# CSV가 저런 뜻이었다니!!
# CSV는 데이터를 쉼표로 구분하여 저장한 데이터이며, 사용이 편리하므로 데이터 분석 등에 자주 사용된다

# pandas를 사용하여 CSV 파일을 읽고, 이를 DataFrame으로 만들어보자

import pandas as pd
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = None)

df.columns = ["", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", 
              "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", 
              "Proanthocyanins", "Color intensity", "Hue", "0D280/0D315 of diluted wines",
              "Proline" ]

print(df)

# 문제
# 다음의 웹사이트에서 붓꽃 데이터를 CSV 파일형식으로 불러들이고, Pandas의 DataFrame형으로 출력
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")

df.columns = ["sepal length", "sepal width", "petal length", "petal widts", "class"]

print(df)

# 14.1.2 CSV 라이브러리로 CSV 만들기
# 파이썬 3에 기본 탑재된 CSV 라이브러리로 CSV 데이터를 만들어보자
# 과거 10회의 올림픽 데이터를 표로 만들어보세요

import csv

# with 문을 사용해서 파일을 처리한다
with open("csv0.csv", "w") as csvfile :
    # writer() 메서드의 인수로 csvfile과 개행코드(\n)을 지정한다
    writer = csv.writer(csvfile, lineterminator = "\n")

    # writerow(리스트)로 행을 추가
    writer.writerow(["city", "year", "season"])
    writer.writerow(["Nagano", 1998, "winter"])
    writer.writerow(["Sydney", 2000, "summer"])
    writer.writerow(["Salt Lake City", 2002, "winter"])
    writer.writerow(["Athens", 2004, "summer"])
    writer.writerow(["Torino", 2006, "winter"])
    writer.writerow(["Beijing", 2008, "summer"])
    writer.writerow(["Vancouver", 2010, "winter"])
    writer.writerow(["London", 2012, "summer"])
    writer.writerow(["Sochi", 2014, "winter"])
    writer.writerow(["Rio de Janeiro", 2016, "summer"])


# 문제
# CSV 파일을 독자 스스로 만들어 보세요
with open ("lotte_mainplayer.csv", "w") as csvfile :
    writer = csv.writer(csvfile, lineterminator = "\n")
    
    writer.writerow(["name", "age", "number"])
    writer.writerow(["Ahseop", 32, 31])
    writer.writerow(["Junwoo", 34, 8])
    writer.writerow(["Daeho", 37, 10])


# 14.1.3 Pandas로 CSV 만들기
# 굳이 CSV 라이브러리를 사용하지 않아도 Pandas로 CSV 데이터를 만들 수 있다
# Pandas의 DataFrame 자료형을 CSV 파일로 만들 때는 이 방법이 더 편리하다
# DataFrame 데이터로 일림픽 개최 도시, 연도, 계절을 정리하여 CSV 파일을 만들어보자

data = {"city" : ["Nagano", "Sydney", "Salt Lake City", "Athens", "Torino",
                "Beijing", "Vancouver", "London", "Sochi", "Rio de Janeiro"],
        "year" : [1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016],
        "season" : ["winter", "summer", "winter", "summer", "winter", "summer", 
        "winter", "summer", "winter", "summer"]}

df = pd.DataFrame(data)
df.to_csv("csv1.csv")

print(df)
print()

data = {"OS" : ["Machintosh", "Windows", "Linux"], 
        "release" : [1984, 1985, 1991],
        "country" : ["US", "US", ""]}

df = pd.DataFrame(data)
df.to_csv("0Slist.csv")

print(df)
print()



