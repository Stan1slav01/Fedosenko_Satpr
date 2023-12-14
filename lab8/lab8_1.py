import numpy as np


def nw_angle(supply_arr, demand_arr):
    rows = len(supply_arr)
    cols = len(demand_arr)
    result = np.zeros((rows, cols))

    i = 0
    j = 0

    while i < rows and j < cols:
        quantity = min(supply_arr[i], demand_arr[j])
        result[i, j] = quantity
        supply_arr[i] -= quantity
        demand_arr[j] -= quantity

        if supply_arr[i] == 0:
            i += 1

        if demand_arr[j] == 0:
            j += 1

    return result


def main():
    supply_arr = np.array([200,250,200])
    demand_arr = np.array([190,	100,120,110,130])
    cost_matrix = np.array([
        [28,27,	18,	27,	24],
        [18,26,	27,	32,	21],
        [27,33,	23,	31,	34],
        ])
            

    nw_result = nw_angle(supply_arr, demand_arr)
    result = np.sum(nw_result * cost_matrix)
    print(nw_result)
    print(result)


if __name__ == '__main__':
    main()
