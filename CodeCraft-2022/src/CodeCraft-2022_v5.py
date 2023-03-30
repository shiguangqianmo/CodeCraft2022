import configparser
import numpy as np
import os
import time

stime = time.time()

root_dir = os.path.dirname(os.path.abspath(__file__))[0: -3]
def read_csv():
    # data_dir = root_dir + 'data'    # 官方测试数据
    # data_dir = root_dir + 'pressure_data'    # 压力测试数据
    data_dir = root_dir + 'simulated_data'  #
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

# 计算当前分配方案下每个节点每个时刻的负载
# 返回优化空间最大的节点索引和95分位点
def calc_load():
    Load_band = np.sum(Schedule, axis=1)        # 计算当前分配方案 每个时刻每个节点的负载,T * N
    load_sort = np.sort(Load_band, axis=0)      # 对节点负载在时间上进行排序 T * N
    quantile_95 = load_sort[T - dealT - 1, :]   # 所有节点的95分位点, N
    Optim_space = np.zeros(N)   # 每个节点可优化空间

    for n in range(0, N):
        cur_node_sort = np.unique(load_sort[0: T - dealT, n])  # 去重
        if cur_node_sort.shape[0] == 1:     # 如果这个节点的0到95分位数的值相同（95分位数为0），将其视为优化过的节点
            to_be_optimized_node[n] = 0
        else:   # 否则还有优化空间
            Optim_space[n] = cur_node_sort[-1] - cur_node_sort[-2]  # 计算可优化空间

    Optim_space = np.multiply(Optim_space, to_be_optimized_node)    # 优化过的点的优化空间置为0
    max_space_nodeIndex = np.argmax(Optim_space)   # 优化空间最大的节点索引
    # print(load_sort[:, max_space_nodeIndex])
    ass_t = np.where(Load_band[:, max_space_nodeIndex] == quantile_95[max_space_nodeIndex])[0]   # 可能有多个时刻的值都为95带宽（只取前95%内时刻）

    return max_space_nodeIndex, ass_t, quantile_95, Optim_space[max_space_nodeIndex]


# 对第二次分配进行优化
def optimize():
    _iter = 0
    while(np.sum(to_be_optimized_node) > 0):  # 还有没有优化完的节点
        cur_node_index, ass_t, quantile_95, optim_space = calc_load()   # 计算当前要优化的节点索引，对应的时刻（可能多个），所有节点的95分位值，优化空间
        if ass_t.shape[0] > T/4:
            to_be_optimized_node[cur_node_index] = 0  # 标志位置为0
            continue
        optim_space = np.ones(ass_t.shape[0]) * optim_space     # x个时刻的可优化空间
        sch_t = Schedule[ass_t, :, :]  # ass_t时刻的分配方案 x * M * N
        x_aval_band = np.repeat(np.expand_dims(quantile_95, axis=0), ass_t.shape[0], axis=0) - np.sum(sch_t, axis=1)  # 其他节点可加的带宽，x * N，可能为负
        x_aval_band[x_aval_band < 0] = 0  # 负数置为0
        x_aval_band[:, cur_node_index] = 0  # 本身节点置为0

        aval_client_index = np.where(QoS[cur_node_index, :] > 0)  # 当前节点关联的客户索引
        for i in range(0, ass_t.shape[0]):     # i为第几个时刻
            for m in aval_client_index[0]:  # m为客户索引
                # if ass_t[i] == 5:
                #     if m == 1:
                #         if cur_node_index == 32:
                #             if quantile_95[cur_node_index] == 1044:
                #                 print(ass_t[i])
                aval_band = np.multiply(x_aval_band[i, :], np.transpose(QoS[:, m]))   # 当前客户所连的节点的可加的带宽，N
                reallocation_cap = np.sum(aval_band)  # 该客户节点可进行重新分配的能力
                aval_space = min(sch_t[i, m, cur_node_index], reallocation_cap)  # 可优化空间，取小者
                if aval_space > 0:  # 如果该客户可以进行重新分配
                    if optim_space[i] == 0:
                        break   # 这个时刻已经优化完成，直接考虑下一个时刻
                    if optim_space[i] >= aval_space:   # 如果待优化空间 >= 可优化空间
                        m_sch = aval_band / reallocation_cap * aval_space  # 分配可优化空间 N
                        aval_node_index = np.where(m_sch > 0)  # 可以进行分配的节点索引
                        m_sch = np.floor(m_sch)  # 向下取整
                        m_sch[aval_node_index[0][-1]] = aval_space - np.sum(m_sch[aval_node_index[0][0: -1]])  # 注意最后一个值
                        num = int(m_sch[aval_node_index[0][-1]] - aval_band[aval_node_index[0][-1]])
                        if num > 0:
                            m_sch[aval_node_index[0][0: num]] += 1
                            m_sch[aval_node_index[0][-1]] -= num  # N
                        # 该时刻，其他可分配节点的负载增大，对应aval_band减小
                        aval_band -= m_sch
                        Schedule[ass_t[i], m, :] += m_sch  # 分配方案更新 N
                        Schedule[ass_t[i], m, cur_node_index] -= aval_space
                        optim_space[i] -= aval_space  # 可优化空间减少
                    else:   # 待优化空间 < 可优化空间
                        m_sch = aval_band / reallocation_cap * optim_space[i]
                        aval_node_index = np.where(m_sch > 0)  # 可以进行分配的节点索引
                        m_sch = np.floor(m_sch)
                        m_sch[aval_node_index[0][-1]] = optim_space[i] - np.sum(m_sch[aval_node_index[0][0: -1]])  # 注意最后一个值
                        num = int(m_sch[aval_node_index[0][-1]] - aval_band[aval_node_index[0][-1]])  # 该节点需要承担的 - 能承担的
                        if num > 0:
                            m_sch[aval_node_index[0][0: num]] += 1  # N
                            m_sch[aval_node_index[0][-1]] -= num  # N
                        Schedule[ass_t[i], m, :] += m_sch
                        Schedule[ass_t[i], m, cur_node_index] -= np.sum(m_sch)
                        optim_space[i] = 0
                        break
            if optim_space[i] > 0:  # 如果遍历完所有客户节点后，可优化空间还是>0，则该节点无法继续被优化
                to_be_optimized_node[cur_node_index] = 0    # 标志位置为0
        _iter += 1


# 全局变量
QoS, Demand, Node, customer_ID, site_ID = read_csv()
(T, M) = Demand.shape
(N, _) = Node.shape     # T=100为时刻数，M=10为用户节点数，N=100为边缘节点数
fill_dot = 0    #初始化边缘节点位置
dealT = int(np.floor(0.05 * T))  # 需要考虑的时间序列个数
Schedule = np.zeros((T, M, N))
remain_bandwidth = np.repeat(np.transpose(Node), T, axis=0)  # T个时刻所有节点剩余带宽 T * N
to_be_optimized_node = np.ones(N)    # 待优化的节点，N，为0则优化过，为1则还未被优化
# if T > 5000:
#     iter_max = 100
# else:
#     iter_max = 1e6
print(T, M, N)


if __name__ == "__main__":
    first_distribution()
    second_distribution()
    optimize()
    # 把gbest写入output.txt
    save_output(Schedule)

    etime = time.time()
    print(etime - stime)

