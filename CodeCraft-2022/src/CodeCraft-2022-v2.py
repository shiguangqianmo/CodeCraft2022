import configparser
import numpy as np
import os
import time

start = time.time()


root_dir = os.path.dirname(os.path.abspath(__file__))[0: -3]
def read_csv():
    data_dir = root_dir + 'data'    # 官方测试数据
    # data_dir = root_dir + 'pressure_data'    # 压力测试数据
    # data_dir = root_dir + 'simulated_data'  #
    # data_dir = '/data'
    config = configparser.ConfigParser()
    config.read(data_dir + '/config.ini')
    qos_constraint = int(config.get('config', 'qos_constraint'))     # qos约束
    print('qos_constraint:', ' ', qos_constraint)

    # 读取并转换QoS约束
    qos_csv = np.genfromtxt(data_dir + '/qos.csv', delimiter=',')  # 读取qos csv文件，不同行表示不同边缘节点，不同列表示不同客户节点,100*10
    QoS = np.delete(qos_csv, 0, axis=0)     # 删除第0行
    QoS = np.delete(QoS, 0, axis=1)         # 删除第0列
    QoS = np.where(QoS < qos_constraint, 1, 0)     # QoS约束 N * M
    with open(data_dir + '/qos.csv') as f:
        data = np.loadtxt(data_dir + '/qos.csv', dtype=str, delimiter=',')
    customer_ID = data[0, 1:]     # 客户ID
    site_ID = data[1:, 0]       # 节点ID

    # 读取客户节点的需求
    demand_csv = np.genfromtxt(data_dir + '/demand.csv', delimiter=',')    # 读取客户节点需求csv文件，不同行表示不同时刻，不同列表示不同客户节点,100*10
    Demand = np.delete(demand_csv, 0, axis=0)
    Demand = np.delete(Demand, 0, axis=1)

    # 读取边缘节点的带宽上限
    site_bandwidth_csv = np.genfromtxt(data_dir + '/site_bandwidth.csv', delimiter=',')    # 读取边缘节点csv文件，不同行表示不同边缘节点, 100*1
    Node = np.delete(site_bandwidth_csv, 0, axis=0)
    Node = np.delete(Node, 0, axis=1)

    return QoS, Demand, Node, customer_ID, site_ID


def save_output(gbest):
    gbest = gbest.astype('int32')
    if os.path.exists(root_dir + 'output') == False:
        os.makedirs(root_dir+'output')
    comma_flag = 0
    with open(root_dir + 'output/solution.txt', mode='w') as output_file:
        for t in range(0, T):
            for m in range(0, M):
                output_file.write(customer_ID[m] + ':')
                for n in range(0, N):
                    if gbest[t, m, n] != 0:     # 不为0才写
                        if comma_flag:
                            output_file.write(',<' + site_ID[n] + ',' + str(gbest[t, m, n]) + '>')
                        else:
                            output_file.write('<' + site_ID[n] + ',' + str(gbest[t, m, n]) + '>')
                            comma_flag = 1
                comma_flag = 0
                output_file.write('\n')
    print('The schedule has been written to output/solution.txt')


# 计算该时刻，该用户是否还有可以贪心的节点
def greedy_node_index(cur_t, cur_client):
    if_greedy = False       # 是否可以贪心
    node_index = None       # 可贪心节点的索引
    band_cur_t = remain_bandwidth[cur_t, :]  # t时刻所有节点剩余带宽 N
    avali_node_band = np.multiply(band_cur_t, QoS[:, cur_client])  # t时刻该用户可用节点的剩余带宽
    sort_index = np.argsort(-avali_node_band)  # 剩余带宽从大到小排序的索引
    for cur_node_index in sort_index:  # 对节点遍历
        cur_node_band = avali_node_band[cur_node_index]  # 当前考虑的节点的带宽
        if cur_node_band > 0:  # 如果当前节点还有剩余可用带宽
            # 判断该节点可否被贪心
            # 即判断节点的两个条件是否都满足 1. 该节点是否已经用过五个时刻； 2. 该节点是否已经被自己用过（情况2不存在）
            sch_node = Schedule[:, :, cur_node_index]  # 该节点在所有时刻的分配方案 T * M
            used_t = np.where(np.sum(sch_node, axis=1) > 0, 1, 0)  # 用过的时刻 T
            if (np.sum(used_t) < greedy_quantity):  # 如果用过的时刻数小于贪心数
                if_greedy = True  # 该节点可被贪心
                node_index = cur_node_index  # 准备使用的节点索引
                break
        else:  # 后面节点剩余带宽全是0
            break
    return if_greedy, node_index


# 贪心时的分配方案
def greedy_schedule(cur_t, node_index):
    client_index = np.where(QoS[node_index, :] > 0)  # 该节点所连客户的索引
    total_demand = np.sum(Demand[cur_t, client_index[0]])  # 该节点所连客户的总需求
    if total_demand <= remain_bandwidth[cur_t, node_index]:  # 如果该节点的带宽可以满足所有用户需求
        Schedule[cur_t, client_index, node_index] += Demand[cur_t, client_index]  # 按照客户需求分配带宽
        remain_bandwidth[cur_t, node_index] -= total_demand  # 更新该节点此刻的剩余带宽
        Demand[cur_t, client_index] = 0  # 更新客户需求——此时刻客户需求已全满足
    else:  # 否则按照用户需求大小比例分配带宽（向下取整）
        cur_sch = np.floor(
            Demand[cur_t, client_index] / total_demand * remain_bandwidth[cur_t, node_index])   # 1 * x
        cur_sch[0, -1] = remain_bandwidth[cur_t, node_index] - np.sum(cur_sch[0, 0: -1])  # 最后一个客户分配的需求谨慎考虑
        num = int(cur_sch[0, -1] - Demand[cur_t, client_index[0][-1]])
        if num > 0:  # 如果分配的需求比真实需求大
            sort_index = np.argsort(-cur_sch)
            cur_sch[0, sort_index[0, 0: num]] += 1
            # cur_sch[0, 0: num] += 1  # 多出的需求分配给其他节点
            cur_sch[0, -1] = 0
        Schedule[cur_t, client_index, node_index] += cur_sch  # 分配方案更新
        remain_bandwidth[cur_t, node_index] = 0  # 该节点此刻带宽已用完
        Demand[cur_t, client_index] -= cur_sch      # 更新客户需求


# 按可用节点剩余带宽比例进行分配——普通策略
def common_schedule(cur_t, cur_client):
    aval_node_band = np.multiply(remain_bandwidth[cur_t, :], QoS[:, cur_client])  # 此时该用户每个节点上可用的带宽 N
    cur_sch = np.ceil(aval_node_band / np.sum(aval_node_band) * Demand[cur_t, cur_client])  # 将客户需求按照剩余可用带宽比进行分配(向上取整） N
    aval_node_index = np.where(cur_sch > 0)  # 有分配带宽的节点索引

    cur_sch[aval_node_index[0][-1]] = Demand[cur_t, cur_client] - np.sum(
        cur_sch[aval_node_index[0][0: -1]])  # 更新最后这个节点分配的需求
    if (cur_sch[aval_node_index[0][-1]]) < 0:  # 如果最后这个节点分配的值为负数
        sort_index = np.argsort(-cur_sch)
        num = -1 * int(cur_sch[aval_node_index[0][-1]])  # 有num个节点需要-1
        cur_sch[sort_index[0: num]] -= 1  # 最大的num个-1
        cur_sch[aval_node_index[0][-1]] = 0  # 最后这个节点的值为0
    Schedule[cur_t, cur_client, :] += cur_sch  # 更新分配策略
    Demand[cur_t, cur_client] = 0  # 更新客户需求
    remain_bandwidth[cur_t, :] -= cur_sch  # 更新剩余带宽


# 全局变量
QoS, Demand, Node, customer_ID, site_ID = read_csv()
# QoS: N * M
# Demand: T * M
# Node: T * 1
(T, M) = Demand.shape
(N, _) = Node.shape     # T=100为时刻数，M=10为用户节点数，N=100为边缘节点数
print(T, M, N)
greedy_quantity = T - np.ceil(T * 0.95)     # 贪心时刻数量
Schedule = np.zeros((T, M, N))      # 分配方案 T * M * N
remain_bandwidth = np.repeat(np.transpose(Node), T, axis=0)  # T个时刻所有节点剩余带宽 T * N

if __name__ == "__main__":
    (max_demand_t, max_demand_client) = np.unravel_index(Demand.argmax(), Demand.shape)    # 最大需求的时刻和用户 tuple(某时刻, 某客户)
    max_demand = Demand[max_demand_t, max_demand_client]     # 最大需求的值

    while abs(max_demand - 0) > 0:   # 如果还没分配完
        # print("t: ", max_demand_t)
        # print("client: ", max_demand_client)
        # 判断该时刻，该用户是否还有贪心节点
        if_greedy, node_index = greedy_node_index(max_demand_t, max_demand_client)

        if if_greedy == True:  # 还有节点能被贪
            greedy_schedule(max_demand_t, node_index)  # 采用贪婪分配策略
        else:  # 没有能贪的节点了
            common_schedule(max_demand_t, max_demand_client)  # 客户需求按照可用节点的剩余带宽比进行分配

        # 计算新的最大需求
        (max_demand_t, max_demand_client) = np.unravel_index(Demand.argmax(), Demand.shape)  # 最大需求的时刻和用户 tuple(某时刻, 某客户)
        max_demand = Demand[max_demand_t, max_demand_client]  # 最大需求的值

    save_output(Schedule)

    end = time.time()
    print(end - start)







