import numpy as np

s1 = [1,1,1,2,2,2,2]
s2 = [3,3,3,1,1,1,1]
s3 = []

for i in range(len(s1)):
    s3.append(max(s1[i], s2[i]))

print(s3)