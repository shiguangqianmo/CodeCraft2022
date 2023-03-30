import configparser
import numpy as np


def read_csv():
    config = configparser.ConfigParser()
    config.read('data/config.ini')
    qos_constraint = int(config.get('config', 'qos_constraint'))     # qos约束
    print('qos_constraint:', ' ', qos_constraint)

    # 读取并转换QoS约束
    qos_csv = np.genfromtxt('data/qos.csv', delimiter=',')  # 读取qos csv文件，不同行表示不同边缘节点，不同列表示不同客户节点,100*10
    QoS = np.delete(qos_csv, 0, axis=0)     # 删除第0行
    QoS = np.delete(QoS, 0, axis=1)         # 删除第0列
    QoS = np.where(QoS < qos_constraint, 1, 0)     # QoS约束
    with open('data/qos.csv') as f:
        data = np.loadtxt('data/qos.csv', dtype=str, delimiter=',')
    customer_ID = data[0, 1:]     # 客户ID
    site_ID = data[1:, 0]       # 节点ID
    # print(customer_ID)
    # print(site_ID)
    # print(type(site_ID[10]))
    # print(QoS)

    # 读取客户节点的需求
    demand_csv = np.genfromtxt('data/demand.csv', delimiter=',')    # 读取客户节点需求csv文件，不同行表示不同时刻，不同列表示不同客户节点,100*10
    Demand = np.delete(demand_csv, 0, axis=0)
    Demand = np.delete(Demand, 0, axis=1)

    # 读取边缘节点的带宽上限
    site_bandwidth_csv = np.genfromtxt('data/site_bandwidth.csv', delimiter=',')    # 读取边缘节点csv文件，不同行表示不同边缘节点, 100*1
    Node = np.delete(site_bandwidth_csv, 0, axis=0)
    Node = np.delete(Node, 0, axis=1)

    return QoS, Demand, Node, customer_ID, site_ID


# 初始化种群的位置（分配方案）和速度
def initpop():
    # T=100为时刻数，M=10为用户节点数，N=100为边缘节点数
    Schedule = np.zeros((popsize, T, M, N))  # 初始化粒子位置，种群数*时刻数*客户节点数*边缘节点数
    Speed = (np.random.rand(popsize, T, M, N) - 0.5) * v_init   # 初始化粒子速度-500~500
    print(Schedule.shape)
    print(Speed.shape)

    for pop in range(0, popsize):  # pop为当前粒子
        for t in range(0, T):  # t为当前时刻
            for m in range(0, M):  # m为当前客户
                avail_n = QoS.sum(axis=0)[m]  # 当前客户可用的边缘节点数
                percent = np.random.rand(avail_n)
                ratio = 1 / sum(percent)
                percent = ratio * percent  # 每个可用节点分配给当前用户的需求比例
                sch_cur_client = np.floor(Demand[t, m] * percent)  # 每个可用节点分配给当前用户的带宽
                sch_cur_client[avail_n - 1] = Demand[t, m] - sum(sch_cur_client[0: avail_n - 1])
                i = 0
                for n in range(0, N):  # n 为当前边缘节点
                    if QoS[n, m] - 0 > 1e-6:    # 当前节点和客户满足QoS约束
                        Schedule[pop, t, m, n] = sch_cur_client[i]  # 分配带宽
                        i += 1
                    else:   # 不满足约束，速度置零
                        Speed[pop, t, m, n] = 0
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
def param_update(Schedule_last, Speed_last, pbest, gbest):
    Schedule = Schedule_last
    Speed = Speed_last
    for pop in range(0, popsize):   # 对每个粒子
        last_pop_sch = Schedule_last[pop, :, :, :]   # 上一次粒子位置
        last_pop_speed = Speed_last[pop, :, :, :]    # 上一次粒子速度
        cur_pop_speed = w * last_pop_speed + \
                    c1 * np.random.rand() * (pbest[pop, :, :, :] - last_pop_sch) + \
                    c2 * np.random.rand() * (gbest - last_pop_sch)   # 速度更新
        cur_pop_sch = last_pop_sch + cur_pop_speed       # 位置更新

        # 使位置合法（分配方案满足需求约束）
        for t in range(0, T):   # 对每个时刻
            for m in range(0, M):   # 对每个客户
                cur_client_sch = cur_pop_sch[t, m, :]   # 当前客户的分配方案
                cur_client_sch = np.multiply(cur_client_sch, np.where(cur_client_sch >= 0, 1, 0))  # 只保留大于0的元素
                cur_demand = np.sum(cur_client_sch)  # 当前分配方案产生的总需求
                if np.abs(cur_demand - Demand[t, m]) > 1e-6:  # 如果当前分配产生的总需求与真实需求不一致
                    cur_client_sch = cur_client_sch / cur_demand * Demand[t, m]  # 按原比例调整需求分配
                    cur_client_sch = np.floor(cur_client_sch)  # 向下取整（向上取整需求会溢出）
                    avil_index = np.where(cur_client_sch > 0)  # 有需求分配的节点索引
                    cur_client_sch[avil_index[0][-1]] = Demand[t, m] - np.sum(cur_client_sch[avil_index[0][0: -1]])
                    cur_pop_sch[t, m, :] = cur_client_sch   # 调整后的分配方案
        Schedule[pop, :, :, :] = cur_pop_sch
        Speed[pop, :, :, :] = cur_pop_speed
    return Schedule, Speed


def save_output(gbest):
    gbest = gbest.astype('int32')
    comma_flag = 0
    with open('output/solution.txt', mode='w') as output_file:
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
maxgen = 5  # 最大迭代次数
popsize = 50  # 种群规模
v_init = 100  # 初始化的粒子速度最大值-50 ~ +50
w = 0.9   # 惯性因子
c1 = 2
c2 = 2   # 学习因子
v_limit = 100    # 速度最大值不能超过v_limit
QoS, Demand, Node, customer_ID, site_ID = read_csv()
(T, M) = Demand.shape
(N, _) = Node.shape     # T=100为时刻数，M=10为用户节点数，N=100为边缘节点数
print(T, M, N)


if __name__ == "__main__":
    Schedule, Speed = initpop()    # 种群初始化 (popsize, T, M, N)
    print('Schedule', Schedule.shape)
    print('Speed', Speed.shape)

    fitness, pbest, gbest = calculate(Schedule)    # 计算种群初始化后的fitness, pbest, gbest
    print(fitness)
    print("The best fitness: ", np.min(fitness))
    print(pbest.shape)
    print(gbest.shape)

    fitness_list = np.ones(maxgen) * float("inf")
    for i in range(0, maxgen):
        Schedule, Speed = param_update(Schedule, Speed, pbest, gbest)   # 更新Schedule和Speed
        fitness, pbest, gbest = calculate(Schedule, fitness, pbest)
        fitness_list[i] = np.min(fitness)
        print("iter: ", i, '  best_fitness: ', np.min(fitness))
    print(fitness_list)

    # 把gbest写入output.txt
    save_output(gbest)


















