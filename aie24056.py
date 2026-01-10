"""String=input("Enter the string: ")
print(String)
count1=0
count2=0
for s in String:
    if s in ['a','e','i','o','u']:
        count1+=1
    else:
        count2+=1
print("count of vowels: ",count1)
print("count of constants: ",count2)

        
List1=[1,2,3,4,5,6,7,8,9,0]
List2=[11,12,13,14,15,1,2,3]
count=0
for i in List1:
    for j in List2:
        if i==j:
            count+=1
print("no.of common elements:",count)

r = int(input("Enter number of rows: "))
c = int(input("Enter number of columns: "))
matrix = []
print("Enter matrix elements:")
for i in range(r):
    row = list(map(int, input().split()))
    matrix.append(row)
print("Transpose of the matrix:")
for i in range(c):
    for j in range(r):
        print(matrix[j][i], end=" ")
    print()

r1 = int(input("Enter rows of matrix A: "))
c1 = int(input("Enter columns of matrix A: "))

A = []
print("Enter elements of matrix A:")
for i in range(r1):
    A.append(list(map(int, input().split())))

r2 = int(input("Enter rows of matrix B: "))
c2 = int(input("Enter columns of matrix B: "))

B = []
print("Enter elements of matrix B:")
for i in range(r2):
    B.append(list(map(int, input().split())))

if c1 != r2:
    print("Matrices cannot be multiplied")
else:
    result = []
    for i in range(r1):
        row = []
        for j in range(c2):
            row.append(0)
        result.append(row)

    for i in range(r1):
        for j in range(c2):
            for k in range(c1):
                result[i][j] = result[i][j] + A[i][k] * B[k][j]

    print("Resultant matrix:")
    for row in result:
        print(row)"""

import random
numbers = []
for i in range(100):
    numbers.append(random.randint(100, 150))
total = 0
for i in numbers:
    total = total + i
mean = total / 100
numbers.sort()
median = (numbers[49] + numbers[50]) / 2
mode = numbers[0]
max_count = 0
for i in numbers:
    count = 0
    for j in numbers:
        if i == j:
            count = count + 1
    if count > max_count:
        max_count = count
        mode = i
print("Numbers",numbers)
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)


        
        
