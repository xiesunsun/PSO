def jMultiTaskPSO(feat, label, opts):
    # 参数初始化
    lb = 0  # 下界
    ub = 1  # 上界
    thres = 0.5  # 阈值，用于将连续值转换为二进制
    c1 = 2  # 认知因子
    c2 = 2  # 社会因子
    c3 = 2  # 群体社会因子
    w = 0.9  # 惯性权重
    Vmax = (ub - lb) / 2  # 最大速度

    # 从opts字典中获取参数，如果没有指定则使用默认值
    N = opts.get("N", 20)  # 粒子数量
    max_Iter = opts.get("T", 100)  # 最大迭代次数
    c1 = opts.get("c1", c1)
    c2 = opts.get("c2", c2)
    c3 = opts.get("c3", c3)
    w = opts.get("w", w)
    Vmax = opts.get("Vmax", Vmax)
    thres = opts.get("thres", thres)

    # 目标函数
    fun = jFitnessFunction
    # 特征维度
    dim = feat.shape[1]
    # 特征权重矩阵
    weight = np.zeros(dim)

    # 初始化粒子群
    X = np.random.uniform(lb, ub, (N, dim))  # 粒子位置
    V = np.zeros((N, dim))  # 粒子速度
    fit = np.zeros(N)  # 粒子适应度
    fitG = np.inf  # 全局最优适应度
    
    # 子种群相关初始化
    numSub = 2  # 子种群数量
    fitSub = np.ones(numSub) * np.inf  # 子种群最优适应度
    Xsub = np.zeros((numSub, dim))  # 子种群最优位置
    subSize = int(N / numSub)  # 每个子种群的大小

    # 初始化每个粒子的适应度和最优位置
    j = 0
    for i in range(N):
        fit[i] = fun(feat, label, X[i, :] > thres, opts)
        # 更新子种群最优
        if fit[i] < fitSub[j]:
            Xsub[j, :] = X[i, :]
            fitSub[j] = fit[i]
        # 更新子种群索引
        if (i + 1) % subSize == 0:
            j += 1
        # 更新全局最优
        if fit[i] < fitG:
            Xgb = X[i, :]
            fitG = fit[i]

    # 个体历史最优初始化
    Xpb = X.copy()
    fitP = fit.copy()

    # 用于记录迭代过程中的最优适应度和选择的特征数量
    curve = np.zeros(max_Iter + 1) # why would add 1
    curve[0] = fitG    # curve
    fnum = np.zeros(max_Iter + 1)
    fnum[0] = np.sum(Xpb[0, :] > thres)

    # 主循环
    t = 1
    while t <= max_Iter: #这里会进行相应的100次的循环
        k = 0
        for i in range(N):
            if k == 0:  # 子种群1：标准PSO更新 sub_1
                for d in range(dim):
                    r1, r2, r3 = np.random.rand(3)
                    # 速度更新
                    VB = (w * V[i, d] +
                          c1 * r1 * (Xpb[i, d] - X[i, d]) +
                          c2 * r2 * (Xgb[d] - X[i, d]) +
                          c3 * r3 * (Xsub[1, d] - X[i, d]))
                    # 速度限制
                    VB = np.clip(VB, -Vmax, Vmax)
                    V[i, d] = VB
                    # 位置更新
                    X[i, d] = X[i, d] + V[i, d]
                # 边界处理
                X[i, :] = np.clip(X[i, :], lb, ub)
            else:  # 子种群2：基于特征权重的更新
                index = weight < 0
                valued_features = weight.copy()
                valued_features[index] = 0 #这一步把所有权重小于零的维度都给其 置为0了
                sum_values = np.sum(valued_features) #得到相应的选择尺寸的维度
                for d in range(dim):
                    p = np.random.rand()
                    if valued_features[d] / sum_values > p: #sub_2好像是有一定的随机性进行
                        X[i, d] = 1
                    else:
                        X[i, d] = 0

            # 计算新位置的适应度
            fit[i] = fun(feat, label, X[i, :] > thres, opts)
            
            # 更新个体最优
            if fit[i] < fitP[i]:
                increase_acc = fitP[i] - fit[i]
                Xpb[i, :] = X[i, :]
                fitP[i] = fit[i]
                # 更新特征权重
                change = np.logical_xor(X[i, :] > thres, Xpb[i, :] > thres)
                weight[change & (X[i, :] > thres)] += increase_acc
                weight[change & (X[i, :] <= thres)] -= increase_acc
            else:
                decrease_acc = fit[i] - fitP[i]
                # 更新特征权重
                change = np.logical_xor(Xpb[i, :] > thres, X[i, :] > thres)
                weight[change & (Xpb[i, :] > thres)] -= decrease_acc
                weight[change & (Xpb[i, :] <= thres)] += decrease_acc

            # 更新子种群最优
            if fit[i] < fitSub[k]:
                Xsub[k, :] = X[i, :]
                fitSub[k] = fit[i]
            # 更新子种群索引
            if (i + 1) % subSize == 0: #是偶数族群 则下一轮切换到奇数族群
                k += 1
            # 更新全局最优
            if fitP[i] < fitG:
                Xgb = Xpb[i, :]
                fitG = fitP[i]

        # 记录当前迭代的最优结果
        curve[t] = fitG
        fnum[t] = np.sum(Xgb > thres)
        print(f"Iteration {t} Best (MEL)= {curve[t]}")
        t += 1

    # 选择最终的特征子集
    Pos = np.arange(dim)
    Sf = Pos[Xgb > thres]
    sFeat = feat[:, Sf]

    # 返回结果
    results = {
        "curve": curve,  # 迭代过程中的最优适应度
        "fnum": fnum,    # 迭代过程中选择的特征数量
        "fitG": fitG,    # 最终的最优适应度
        "nf": len(Sf),   # 最终选择的特征数量
        "sf": Sf,        # 最终选择的特征索引
        "ff": sFeat,     # 最终选择的特征子集
    }
    return results