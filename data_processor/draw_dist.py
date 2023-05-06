import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
def read_csv(filename):
    # 创建一个空数组，用于存储所有列的数据
    data = []

    # 打开CSV文件
    with open(filename, mode='r') as csv_file:
        # 创建一个CSV读取器
        reader = csv.reader(csv_file)

        # 遍历CSV文件的每一行
        for row in reader:
            # 遍历每一列，并将其添加到data数组中对应的数组中
            for i in range(len(row)):
                # print(len(row))
                # 如果data数组中还没有对应的数组，则创建一个新的数组并添加到data数组中
                if len(data) <= i:
                    data.append([])

                # 将当前元素添加到对应的数组中
                data[i].append(row[i])

    return data

# 调用函数，读取CSV文件并将每一列封装为一个数组
data = read_csv('test77.csv')

# 打印数据
print(data)

fig, ax = plt.subplots()
for i in range(len(data)):
    name = data[i][0]
    ax.plot(data[i][1:], label=name)
ax.legend()
tick_spacing = 10
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
plt.show()
    
