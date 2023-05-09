import numpy as np

class KM_matcher:

    def __init__(self, Bipartite_Graph):

        self.Bipartite_Graph = Bipartite_Graph

        # 左右结点数量记录
        self.left = self.Bipartite_Graph.shape[0]  # 以左边为主
        # print(self.Bipartite_Graph)
        # print(self.Bipartite_Graph[0])
        self.right_true = self.Bipartite_Graph.shape[1]
        self.right = self.Bipartite_Graph.shape[1] + self.left

        self.reshape_graph()

        # 初始化顶标
        self.label_left = np.max(self.Bipartite_Graph, axis=1)  # 设置左边顶标为权重最大值（每行的最大值）

        self.label_right = np.zeros(self.right)  # 右边集合的顶标设置为0

        # 初始化辅助变量——是否已匹配
        self.visit_left = np.zeros(self.left, dtype=bool)
        self.visit_right = np.zeros(self.right, dtype=bool)

        # 初始化右边的匹配结果.如果已匹配就会对应匹配结果
        self.match_right = np.empty(self.right) * np.nan

        # 用inc记录需要减去的权值d，不断取最小值故初始化为较大值。权值都为负数，应该不用很大也行
        self.inc = 1000*1000*1000

        self.fail_boy = list()  # 每次匹配重新创建一个二分图匹配对象，所以这个也不用手动重置了

    def reshape_graph(self):
        new = np.ones((self.left, self.left)) * 0
        self.Bipartite_Graph = np.column_stack((self.Bipartite_Graph, new))
        # print(self.Bipartite_Graph)

    def match(self, boy):
        # print("---------------我是boy%d----------------------" % boy)
        boy = int(boy)
        # 记录下这个boy已经被寻找
        self.visit_left[boy] = True
        for girl in range(self.right):
            # 如果这个女生还没访问过
            if not self.visit_right[girl] and self.Bipartite_Graph[boy][girl] >= 0:  # 女孩仍未匹配并且男女之间存在匹配的可能性(不可匹配的点设置为负数，取反后变正数,故正数不可取)
                gap = self.label_left[boy] + self.label_right[girl] - self.Bipartite_Graph[boy][girl]  # gap也不会取到不能匹配的那条边
                if gap == 0:   # 差值为0，是可行的替换。所以可以直接尝试替换。后面不行再去将这个一起减去gap。这个列表是记录希望匹配的
                    self.visit_right[girl] = True
                    # 女生未被匹配，或虽然已被匹配，但是已匹配对象(男生)有其他可选备胎。这里那些是否已访问的列表不需要重置，因为不改变前面的尝试匹配
                    if np.isnan(self.match_right[girl]) or self.match(self.match_right[girl]):
                        self.match_right[girl] = boy
                        # print(self.match_right)
                        # 递归匹配，匹配成功
                        return 1

                # 找到权值最小的差距
                elif self.inc > gap:
                    self.inc = gap  # 等于0的gap不会存在这，所以只要存在可能匹配的情况，gap就不会等于原来的inc

        return 0

    def Kuh_Munkras(self):

        self.match_right = np.empty(self.right) * np.nan
        # 寻找最优匹配
        for man in range(self.left):
            while True:
                self.inc = 1000*1000  # the minimum gap
                self.reset()  # 每次寻找过的路径，所有要重置一下

                # 可找到可行匹配
                if self.match(man):
                    break
                # 不能找到可行匹配
                # (1)将所有在增广路中的boy方点的label全部减去最小常数
                # (2)将所有在增广路中的girl方点的label全部加上最小常数
                for k in range(self.left):
                    if self.visit_left[k]:
                        self.label_left[k] -= self.inc
                for n in range(self.right):
                    if self.visit_right[n]:
                        self.label_right[n] += self.inc

        return self.fail_boy

    def calculateSum(self):
        sum = 0
        boys_girls = []
        self.fail_boy = [i for i in range(self.left)]
        for i in range(self.right_true):
            if not np.isnan(self.match_right[i]):
                sum += self.Bipartite_Graph[int(self.match_right[i])][i]
                boy_girl = (int(self.match_right[i]), i)
                boys_girls.append(boy_girl)
                self.fail_boy.remove(int(self.match_right[i]))
        # print("最短路径：", sum)

        return boys_girls, self.fail_boy

    def getResult(self):
        return self.match_right

    def reset(self):
        self.visit_left = np.zeros(self.left, dtype=bool)
        self.visit_right = np.zeros(self.right, dtype=bool)

def main():
    graph = [[8,6,-1,-1],
            [-1,3,9,-1],
            [9,8,-1,-1],
            [-1,-1,2,-1],
            [6, 5, 4, 3]]
    #print(graph)
    km = KM_matcher(np.array(graph))

    km.Kuh_Munkras()  # 匹配

    boys_girls, fail_boys = km.calculateSum()  # 匹配结果元组,以及失败的男孩们

    print("匹配男女列表", boys_girls)
    print("失败男孩列表", fail_boys)


if __name__ == "__main__":
    main()
