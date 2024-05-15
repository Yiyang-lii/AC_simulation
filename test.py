import random

# 產生包含數字1、2、3的列表
numbers = ['C', 'L', 'G']

# 洗牌列表
random.shuffle(numbers)

# 顯示隨機順序的數字
for index, number in enumerate(numbers):
    print(f"{index + 1}: {number}")