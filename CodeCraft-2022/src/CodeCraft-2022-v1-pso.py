import configparser
import numpy as np
import os
import time

start = time.time()


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


# 计算某时刻带宽余量
def remain_bandwidth(cur_t_sch):
    used_band = np.sum(cur_t_sch, axis=0)
    remained_band = np.transpose(Node) - used_band
    return remained_band


# 初始化第一个粒子
def init_first_pop():
    schedule_first_pop = np.zeros((T, M, N))    # 初始化第一个粒子的位置
    speed_first_pop = (np.random.rand(T, M, N) - 0.5) * v_init  # 初始化第一个粒子的速度
    for t in range(0, T):   # 当前时刻
        cur_t_sch = schedule_first_pop[t, :, :]     # M*N
        for m in range(0, M):   # 当前客户
            remained_band = remain_bandwidth(cur_t_sch)     # 当前时刻剩余带宽 1 * N
            avail_node_band = np.multiply(np.transpose(QoS[:, m]), remained_band)   # 当前可用节点的剩余带宽 1*N
            band_per_node = avail_node_band / np.sum(avail_node_band)      # 每个可用带宽分配占比    1 * N
            sch_cur_client = np.ceil(Demand[t, m] * band_per_node)     # 当前客户需求分配情况，四舍五入取整 1 * N
            avail_node_index = np.where(QoS[:, m] > 0)      # 可使用节点索引
            sch_cur_client[0, avail_node_index[0][-1]] = Demand[t, m] - np.sum(sch_cur_client[0, avail_node_index[0][0: -1]])   # 更新最后一个可使用节点的值使满足客户需求
            if sch_cur_client[0, avail_node_index[0][-1]] < 0:      # 如果最后可使用节点的值为负数
                sort_index = np.argsort(-sch_cur_client)
                num = -1 * int(sch_cur_client[0, avail_node_index[0][-1]])      # 有num个节点需要-1
                sch_cur_client[0, sort_index[0, 0: num]] = sch_cur_client[0, sort_index[0, 0: num]] - 1
                sch_cur_client[0, avail_node_index[0][-1]] = 0  # 最后一个可用节点的值置为0
            cur_t_sch[m, :] = sch_cur_client    # 当前时刻分配方案更新
        schedule_first_pop[t, :, :] = cur_t_sch     # 各时刻分配方案写入
        speed_first_pop[t, :, :] = np.multiply(speed_first_pop[t, :, :], np.transpose(QoS))     # 不满足Qos处速度置零 M * N
    return schedule_first_pop, speed_first_pop


# 初始化种群的位置（分配方案）和速度
def initpop():
    # T=100为时刻数，M=10为用户节点数，N=100为边缘节点数
    Schedule = np.zeros((popsize, T, M, N))  # 初始化粒子位置，种群数*时刻数*客户节点数*边缘节点数
    Speed = (np.random.rand(popsize, T, M, N) - 0.5) * v_init   # 初始化粒子速度-50~50
    print(Schedule.shape)
    print(Speed.shape)
    Schedule[0, :, :, :], Speed[0, :, :, :] = init_first_pop()      # 第一个粒子初始化

    # 剩余粒子初始化
    for pop in range(1, popsize):  # pop为当前粒子
        for t in range(0, T):  # t为当前时刻
            for m in range(0, M):  # m为当前客户
                percent_per_node = np.random.rand(1, N)      # 给每个节点一个0~1的随机数
                avail_node_band = np.multiply(np.transpose(QoS[:, m]), percent_per_node)     # 保留可用节点上的随机数
                band_per_node = avail_node_band / np.sum(avail_node_band)  # 每个可用带宽分配占比    1 * N
                sch_cur_client = np.ceil(Demand[t, m] * band_per_node)  # 当前客户需求分配情况，四舍五入取整 1 * N
                avail_node_index = np.where(QoS[:, m] > 0)  # 可使用节点索引
                sch_cur_client[0, avail_node_index[0][-1]] = Demand[t, m] - np.sum(sch_cur_client[0, avail_node_index[0][0: -1]])  # 更新最后一个可使用节点的值使满足客户需求
                if sch_cur_client[0, avail_node_index[0][-1]] < 0:  # 如果最后可使用节点的值为负数
                    sort_index = np.argsort(-sch_cur_client)
                    num = -1 * int(sch_cur_client[0, avail_node_index[0][-1]])  # 有num个节点需要-1
                    sch_cur_client[0, sort_index[0, 0: num]] = sch_cur_client[0, sort_index[0, 0: num]] - 1
                    sch_cur_client[0, avail_node_index[0][-1]] = 0  # 最后一个可用节点的值置为0
                Schedule[pop, t, m, :] = sch_cur_client
            Speed[pop, t, :, :] = np.multiply(Speed[pop, t, :, :], np.transpose(QoS))
    return Schedule, Speed


# 计算适应度函数（成本）
# 并根据适应度函数值更新pbest（粒子最优位置）和gbest（全局最优位置）
def calculate(Schedule, fitness_last=None, pbest_last=None):
    if fitness_last is None:    # 第一次计算适应度值
        fitness_last = np.ones(popsize) * float("inf")  # 适应度值初始化为无穷 1*popsize
    if pbest_last is None:      # 第一次计算pbest
        pbest_last = Schedule       # 所有粒子均用初始分配方案初始化 (popsize, T, M, N)
    pbest = pbest_last  # 用上次的pbest初始化此次pbest
    fitness = fitness_last

    for pop in range(0, popsize):   # 计算每个粒子此次的适应度值
        cur_pop_shc = Schedule[pop, :, :, :]  # 当前粒子T个时刻所有分配方案
        cost = np.sum(cur_pop_shc, axis=1)  # T个时刻N个节点的带宽使用量 T*N
        # 判断每个时刻每个节点是否均满足带宽约束
        bandwidth_limit = np.repeat(np.transpose(Node), T, axis=0)  # T * N
        not_satisfied = np.where(cost <= bandwidth_limit, 0, 1)     # 不满足时置为1
        if np.sum(not_satisfied) < 1e-6:    # 当前分配方案有效，则计算当前粒子的fitness
            cost = np.sort(cost, axis=0)  # 在时间维度上排序
            index = int(np.ceil(T * 0.95)) - 1  # 索引向上取整，索引从0开始
            cur_fitness = np.sum(cost[index, :])  # 当前fitness,一个值
            if cur_fitness <= fitness_last[pop]:    # 当前fitness更优
                pbest[pop, :, :, :] = cur_pop_shc   # 更新pbest
                fitness[pop] = cur_fitness  # 更新当前粒子的fitness
        # else: 当前分配方案无效，则此次的pbest还是为上次的pbest

    # 更新gbest
    fitness_min_index = np.argmin(fitness)
    gbest = pbest[fitness_min_index, :, :, :]
    return fitness, pbest, gbest


# 更新位置（分配方案）和速度
def param_update(Schedule_last, Speed_last, pbest, gbest, g):
    Schedule = Schedule_last
    Speed = Speed_last
    w_t = (w_init - w_end) * (maxgen - g) / maxgen + w_end
    for pop in range(0, popsize):   # 对每个粒子
        last_pop_sch = Schedule_last[pop, :, :, :]   # 上一次粒子位置
        last_pop_speed = Speed_last[pop, :, :, :]    # 上一次粒子速度
        cur_pop_speed = w_t * last_pop_speed + \
                    c1 * np.random.rand() * (pbest[pop, :, :, :] - last_pop_sch) + \
                    c2 * np.random.rand() * (gbest - last_pop_sch)   # 速度更新
        cur_pop_speed = np.where(cur_pop_speed < -1*v_limit, -1 * v_limit, 0) + \
                        np.multiply(np.where((-1*v_limit <= cur_pop_speed) & (cur_pop_speed <= v_limit), 1, 0), cur_pop_speed) + \
                        np.where(cur_pop_speed > v_limit, v_limit, 0)       # 限制速度
        cur_pop_sch = last_pop_sch + cur_pop_speed       # 位置更新

        # 使位置合法（分配方案满足需求约束）
        for t in range(0, T):   # 对每个时刻
            for m in range(0, M):   # 对每个客户
                cur_client_sch = cur_pop_sch[t, m, :]   # 当前客户的分配方案
                cur_client_sch = np.multiply(cur_client_sch, np.where(cur_client_sch >= 0, 1, 0))  # 只保留大于0的元素
                cur_demand = np.sum(cur_client_sch)  # 当前分配方案产生的总需求
                if np.abs(cur_demand - Demand[t, m]) > 1e-6:  # 如果当前分配产生的总需求与真实需求不一致
                    cur_client_sch = cur_client_sch / cur_demand * Demand[t, m]  # 按原比例调整需求分配   # N
                    cur_client_sch = np.ceil(cur_client_sch)  # 向上取整
                    avil_index = np.where(cur_client_sch > 0)  # 有需求分配的节点索引
                    cur_client_sch[avil_index[0][-1]] = Demand[t, m] - np.sum(cur_client_sch[avil_index[0][0: -1]])
                    if cur_client_sch[avil_index[0][-1]] < 0:
                        sort_index = np.argsort(-cur_client_sch)
                        num = -1 * int(cur_client_sch[avil_index[0][-1]])  # 有num个节点需要-1
                        cur_client_sch[sort_index[0: num]] = cur_client_sch[sort_index[0: num]] - 1
                        cur_client_sch[avil_index[0][-1]] = 0  # 最后一个可用节点的值置为0
                    cur_pop_sch[t, m, :] = cur_client_sch   # 调整后的分配方案
        Schedule[pop, :, :, :] = cur_pop_sch
        Speed[pop, :, :, :] = cur_pop_speed
    return Schedule, Speed


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


# 全局变量
maxgen = 10  # 最大迭代次数
popsize = 100  # 种群规模
v_init = 100  # 初始化的粒子速度最大值-50 ~ +50
w_init = 0.9
w_end = 0.4
# w = 0.9   # 惯性因子
c1 = 2
c2 = 2   # 学习因子
v_limit = 200    # 速度最大值不能超过v_limit
QoS, Demand, Node, customer_ID, site_ID = read_csv()
(T, M) = Demand.shape
(N, _) = Node.shape     # T=100为时刻数，M=10为用户节点数，N=100为边缘节点数
print(T, M, N)


if __name__ == "__main__":
    # Schedule, Speed = initpop()    # 种群初始化 (popsize, T, M, N)
    # print('Schedule', Schedule.shape)
    # print('Speed', Speed.shape)
    #
    # fitness, pbest, gbest = calculate(Schedule)    # 计算种群初始化后的fitness, pbest, gbest
    # print(fitness)
    # print("The best fitness: ", np.min(fitness))
    # print(pbest.shape)
    # print(gbest.shape)
    #
    # fitness_list = np.ones(maxgen) * float("inf")
    # for i in range(0, maxgen):
    #     Schedule, Speed = param_update(Schedule, Speed, pbest, gbest, g=i)   # 更新Schedule和Speed
    #     fitness, pbest, gbest = calculate(Schedule, fitness, pbest)
    #     fitness_list[i] = np.min(fitness)
    #     print("iter: ", i, '  best_fitness: ', np.min(fitness))
    #     # print(fitness)
    # print(fitness_list)

    (schedule_first_pop, speed_first_pop) = init_first_pop()
    gbest = schedule_first_pop
    # 把gbest写入output.txt
    save_output(gbest)
    end = time.time()
    print(end - start)

    # (schedule_first_pop, speed_first_pop) = init_first_pop()
    # print(schedule_first_pop)
    # print(speed_first_pop)
















