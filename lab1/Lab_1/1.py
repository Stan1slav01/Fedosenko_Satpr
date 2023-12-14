import numpy as np

matrix = np.array([
    [3, 7, 2, 9],
    [8, 3, 6, 7],
    [4, 8, 3, 5],
    [9, 6, 5, 4]
])
weight = np.array([8, 9, 6, 7])
crit_value = np.dot(matrix, weight)
print(crit_value)
position = np.argmax(crit_value)
print(position)
print(f"ОПР найкраще укласти договір з адвокатом за номером : {position+1} і функцією корисності {crit_value[position]}")
