import numpy as np

def I(x, y):
    tot = x + y
    A = (-x/tot)*np.log2(x/tot)
    B = (-y/tot)*np.log2(y/tot)
    return A + B

# print(I(1, 8))
# result = ((1/9) * I(0, 1)) + ((8/9) * I(1,7))
# result = ((6/9) * I(1,5))
# print(result)
# print(I(2,34))
# res2 = ((-9/11)*np.log2(9/11)) + ((-2/11)*np.log2(2/11))
# print(res2)
norm = np.array([-1.3381, 0.3890])
test = (1 - np.dot(norm, np.array([0.91, 0.32])))/3 + (1 - np.dot(norm, np.array([0.41, 2.04])))/3 + (-1 - np.dot(norm, np.array([2.05, 1.54])))/3
# print(test)

test2 = np.dot(norm, np.array([-1, 2])) + 1.3308



x2 = np.array([0.91, 0.32])
x6 = np.array([0.41, 2.04])
x18 = np.array([2.05, 1.54])

w = np.multiply(0.5084, x2) + np.multiply(0.4625, x6) + np.multiply(-0.9709, x18)
print('w= ', w)

test3 = (1 - np.dot(w, x2))
test4 = (1 - np.dot(w, x6))
test5 = (-1 - np.dot(w, x18))
b = test3/3 + test4/3 + test5/3

print(test3)
print(test4)
print(test5)
print('b= ', b)

newX = np.array([-1,2])
print('class= ', np.dot(w, newX) + b)