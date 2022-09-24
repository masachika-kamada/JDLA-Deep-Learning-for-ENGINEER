import numpy as np


# def is10(num_in):
#     num = num_in
#     d = num // 500
#     num = num % 500
#     c = num // 100
#     num = num % 100
#     l = num // 100
#     num = num % 100
#     x = num // 100
#     num = num % 100
#     v = num // 100
#     num = num % 100
#     i = num // 100
#     num = num % 100
#     # if num_in // 100 == 9:
#     #     d = 1
#     #     c = 1
#     if i == 4:
#         i = 1
#         v += 1
#     if v == 4:
#         v = 1
#         x += 1
#     if x == 4:
#         x = 1
#         l += 1
#     if l == 4:
#         l = 1
#         c += 1
#     if c == 4:
#         c = 1
#         d += 1
#     dst = d + c + l + x + v + i
#     if dst == 10:
#         return True
#     else:
#         return False

def is10(num):
    i100 = num // 100
    num = num % 100
    i10 = num // 10
    num = num % 10
    i1 = num
    l = [i1, i10, i100, 0]
    for i in range(3):
        if l[i] == 9:
            l[i] = 1
            l[i + 1] += 1
        elif l[i] == 4:
            l[i] = 2
        elif l[i] > 4:
            l[i] -= 4
    dst = 0
    for item in l:
        dst += item
    if dst == 10:
        return True
    else:
        return False


sum = 0
for i in range(1, 1000):
    if is10(i):
        print(i)
        sum += i
print(sum)
