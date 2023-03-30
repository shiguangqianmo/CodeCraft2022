import configparser
import numpy as np
import os
import time

stime = time.time()

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
    with open(data_dir + '/qos.csv') as f:
        data = np.loadtxt(data_dir + '/qos.csv', dtype=str, delimiter=',')
    QoS = np.array(data[1:, 1:], dtype=int)     # 取数字
    QoS = np.where(QoS < qos_constraint, 1, 0)     # QoS约束 N * M
    customer_ID = data[0, 1:]           # 客户ID M
    site_ID = data[1:, 0]               # 节点ID  N
    col_index = customer_ID.argsort()
    row_index = site_ID.argsort()
    customer_ID = customer_ID[col_index]
    site_ID = site_ID[row_index]
    QoS = QoS[row_index, :]
    QoS = QoS[:, col_index]


    # 读取客户节点的需求
    with open(data_dir + '/demand.csv') as f:
        data = np.loadtxt(data_dir + '/demand.csv', dtype=str, delimiter=',')
    Demand = np.array(data[1:, 1:], dtype='float64')
    c_id = data[0, 1:]     # 客户ID
    col_index = c_id.argsort()
    Demand = Demand[:, col_index]

    # 读取边缘节点的带宽上限
    with open(data_dir + '/site_bandwidth.csv') as f:
        data = np.loadtxt(data_dir + '/site_bandwidth.csv', dtype=str, delimiter=',')
    Node = np.array(data[1:, 1:], dtype='float64')
    s_id = data[1:, 0]  # 节点ID
    col_index = s_id.argsort()
    Node = Node[col_index, :]

    return QoS, Demand, Node, customer_ID, site_ID


def save_output(gbest):
    gbest = gbest.astype('int32')
    if os.path.exists(root_dir + 'output') == False:
        os.makedirs(root_dir + 'output')
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


def first_distribution():
    node_sort_index = np.squeeze(np.argsort(-1 * Node, axis=0))  # 记录降序排序后的节点在初始矩阵中的索引

    for n in node_sort_index:
        fill_block = QoS[n]     # 正在进行分配的边缘节点所对应的客户节点
        ans = np.where(fill_block == 1)    # 返回此边缘节点对应的所有客户节点的索引
        if ans[0].shape[0] == 0:
            continue
        greedy_need = np.multiply(Demand, fill_block)  # 求所对应所有客户节点在时间序列上的和 T * M
        greedy_total = greedy_need.sum(axis=1)     #在时间序列上所有客户节点的需求和 T
        greedy_ans = np.argsort(-1 * greedy_total)     #对所有可用客户节点的需求和在时间序列上排序 T

        for k in range(0, dealT):
            if greedy_total[greedy_ans[k]] < remain_bandwidth[greedy_ans[k], n]:     #比较对应客户节点
                Demand[greedy_ans[k], :] -= greedy_need[greedy_ans[k], :]                            #remain_bandwidth[greedy_ans[k], fill_dot]    #更新T*M需求表
                remain_bandwidth[greedy_ans[k], n] -= greedy_total[greedy_ans[k]]
                Schedule[greedy_ans[k], :, n] += greedy_need[greedy_ans[k], :]   #方案写入
            else:
                first_distribute = np.floor(greedy_need[greedy_ans[k], :] / greedy_total[greedy_ans[k]] * remain_bandwidth[greedy_ans[k], n])  # 按各个客户节点需求瓜分不够多的边缘节点带宽
                first_distribute[ans[0][-1]] = remain_bandwidth[greedy_ans[k], n] - np.sum(first_distribute[ans[0][0: -1]])    #防止溢出
                num = int(first_distribute[ans[0][-1]] - Demand[greedy_ans[k], ans[0][-1]])
                if num > 0:
                    sort_index = np.argsort(-first_distribute[0: -1])
                    first_distribute[sort_index[0: num]] += 1
                    first_distribute[ans[0][-1]] = Demand[greedy_ans[k], ans[0][-1]]
                remain_bandwidth[greedy_ans[k], n] = 0     #带宽置零
                Demand[greedy_ans[k], :] -= first_distribute     #更新需求
                Schedule[greedy_ans[k], :, n] += first_distribute    #方案写入


def second_distribution():
    for t in range(0, T):
        if np.max(Demand[t, :]) > 0:
            avail_dis1 = np.where(Demand[t, :] > 0)
            avail_dis = np.array(avail_dis1)
            for m in avail_dis[0]:
                total_n_m_matrix = np.multiply(remain_bandwidth[t, :], QoS[:, m])    #对此时刻客户节点对应的边缘节点按顺序列出
                total_n_m_sum = total_n_m_matrix.sum()
                common_distribute = np.ceil((total_n_m_matrix / total_n_m_sum) * Demand[t, m])
                use_node_index = np.where(total_n_m_matrix > 0)
                common_distribute[use_node_index[0][-1]] = Demand[t, m] - np.sum(common_distribute[use_node_index[0][0: -1]])
                if (common_distribute[use_node_index[0][-1]]) < 0:
                    sort_index = np.argsort(-common_distribute)
                    num = -1 * int(common_distribute[use_node_index[0][-1]])
                    common_distribute[sort_index[0: num]] -= 1
                    common_distribute[use_node_index[0][-1]] = 0
                Schedule[t, m, :] += common_distribute
                Demand[t, m] = 0
                remain_bandwidth[t, :] -= common_distribute

# 在当前分配方案下
# 1. 计算每个节点前95%时刻内的aval_space——可优化空间
#       aval_space需要根据节点所连的每个客户进行计算=min(分配值，该客户所连的其他节点的aval_band)
# 2. 计算每个节点前95%时刻的 上限=负载-aval_space
# 3. 计算每个节点95-上限（目标优化空间）
# 4. 返回优化空间最大的节点的索引、上限、所有节点的95分位点、要优化的时刻索引
def pre_calc():
    Load_band = np.sum(Schedule, axis=1)        # 计算当前分配方案 每个时刻每个节点的负载,T * N
    load_sort = np.sort(Load_band, axis=0)      # 对节点负载在时间上进行排序 T * N
    quantile_95 = load_sort[T - dealT - 1, :]   # 计算所有节点的95分位点, N
    Aval_band = np.repeat(np.expand_dims(quantile_95, axis=0), T, axis=0) - Load_band   # 95分位数-节点负载    T * N
    Aval_band[Aval_band < 0] = 0    # 负数置为0

    Aval_space = np.zeros((T, N))     # 可优化空间——每时刻每个节点的可优化空间  T * N
    for t in range(0, T):       # 遍历时间
        for n in range(0, N):   # 遍历节点
            client_index = np.where(QoS[n, :] > 0)[0]   # 所连客户索引 x
            client_aval_band = np.multiply(np.repeat(np.expand_dims(Aval_band[t, :], axis=0), \
                                    client_index.shape[0], axis=0), np.transpose(QoS[:, client_index]))   # 每个关联客户的aval_band, x * N
            client_aval_band[:, n] = 0  # 本身节点的aval_band为0
            reallocation_cap = np.sum(client_aval_band, axis=1)
            sch_n = Schedule[t, client_index, n]    # 在这些客户上的分配方案
            aval_space = np.append(np.expand_dims(reallocation_cap, axis=0), np.expand_dims(sch_n, axis=0), axis=0)
            aval_space = np.min(aval_space, axis=0)
            aval_space_detail[t, n] = tuple(aval_space)
            Aval_space[t, n] = np.sum(aval_space)   # t时刻n节点的可优化空间

    Up_limit = np.zeros(N)     # 每个节点的优化上限
    for n in range(0, N):
        load_band_n = Load_band[:, n]   # n节点的负载情况
        load_band_n[load_band_n > quantile_95[n]] = 0   # 大于95带宽的负载置为0
        up_limit_n = load_band_n - Aval_space[:, n]     # n节点各时刻的上限
        Up_limit[n] = np.max(up_limit_n)    # n节点的优化上限

    Optim_space = quantile_95 - Up_limit    # 每个节点的优化空间
    obj_node_index = np.argmax(Optim_space)     # 目标优化节点的索引
    t_index = np.intersect1d(np.where(Load_band[:, obj_node_index] > Up_limit[obj_node_index])[0], np.where(Load_band[:, obj_node_index] < quantile_95[obj_node_index])[0]) # 要优化的时刻索引
    return obj_node_index, Up_limit[obj_node_index], quantile_95, t_index



def optimize2():
    _iter = 0

    while(_iter < max_iter):
        cur_node_index, up_limit, quantile_95, t_index = pre_calc()     # 当前考虑的节点索引，优化上限，所有节点的95带宽
        # 将t_index时刻，cur_node的带宽压至上限
        client_index = np.where(QoS[cur_node_index, :] > 0)[0]  # 当前节点关联的客户索引
        for t in t_index:
            optim_space = np.sum(Schedule[t, :, cur_node_index]) - up_limit     # 这个时刻这个节点可优化空间（当前负载-上限）
            # aval_space = np.array(aval_space_detail[t, cur_node_index])   # 各关联客户节点的aval_space
            # client_band = aval_space / np.sum(aval_space) * optim_space     # 各客户要更改的band
            # aval_client_index = np.where(client_band > 0)[0]
            # client_band[aval_client_index[-1]] = optim_space - np.sum(aval_space[0: -1])
            # num = int(client_band[aval_client_index[-1]] - aval_space[aval_client_index[-1]])    # 需要承担的 > 能承担的
            # if num > 0:
            #     client_band[aval_client_index[0: num]] += 1
            #     client_band[aval_client_index[-1]] = aval_space[aval_client_index[-1]]

            sch_t = Schedule[t, :, :]   # t时刻分配方案
            for m in client_index:
                aval_band = quantile_95 - np.sum(sch_t, axis=0)
                aval_band[aval_band < 0] = 0
                aval_band[cur_node_index] = 0
                aval_band = np.multiply(aval_band, np.where(np.transpose(QoS[:, m]) > 0, 1, 0))
                m_sch = aval_band / np.sum(aval_band) * optim_space
                node_index = np.where(m_sch > 0)[0]
                m_sch[node_index[-1]] = optim_space - np.sum(m_sch[node_index[0: -1]])
                num = int(m_sch[node_index[-1]] - aval_band[node_index[-1]])
                if num > 0:
                    m_sch[node_index[0: num]] += 1
                    m_sch[node_index[-1]] = aval_band[node_index[-1]]
                Schedule[t, m, :] += m_sch
                Schedule[t, m, cur_node_index] -= np.sum(m_sch)
        _iter += 1


# 全局变量
QoS, Demand, Node, customer_ID, site_ID = read_csv()
(T, M) = Demand.shape
(N, _) = Node.shape     # T=100为时刻数，M=10为用户节点数，N=100为边缘节点数
dealT = int(np.floor(0.05 * T))  # 需要考虑的时间序列个数
Schedule = np.zeros((T, M, N))
remain_bandwidth = np.repeat(np.transpose(Node), T, axis=0)  # T个时刻所有节点剩余带宽 T * N
to_be_optimized_node = np.ones(N)    # 待优化的节点，N，为0则优化过，为1则还未被优化
print(T, M, N)
aval_space_detail = np.zeros((T, N), dtype='O')
max_iter = N


if __name__ == "__main__":
    first_distribution()
    second_distribution()
    optimize2()
    # optimize()
    # 把gbest写入output.txt
    save_output(Schedule)

    etime = time.time()
    print(etime - stime)

