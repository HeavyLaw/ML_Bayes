import numpy as np
import matplotlib as plt


class NaiveBayes:
    def __init__(self, features, label):
        self.features = features
        self.label = label
        self.features_type: list = []
        for x in range(features.shape[1]):
            self.features_type.append(np.unique(features[:, x]).tolist())
        self.label_type = np.unique(label).tolist()
        self.features_metric = []
        self.label_metric = []

    @staticmethod
    def cal_prior_probability(alist):
        prior_prob_list = []
        c_type = np.unique(alist)
        alist = alist.tolist()
        for c in c_type:  # 计算类先验概率
            prior_prob_list.append(alist.count(c) / len(alist))
        return np.array(prior_prob_list)

    def cal_conditional_probability(self, features, label):
        conditional_prob_list = []  # 条件概率
        c_type = np.unique(label)

        for c in c_type:
            features_split = []  # 保存按类分割后的样本数据
            index = np.where(label == c)  # 找到对应每一类样本的下标
            for i in index:
                features_split = features[i].tolist()  # 在这里特别注意，如果直接使用list的append()函数会导致features_split的数据类型出现问题

            # 计算每类各个属性对应的取值的条件概率矩阵 p(x|y)
            for x in range(features.shape[1]):
                x_prob = self.cal_prior_probability(np.array(features_split)[:, x])  # 有时分割后的样本数据无法覆盖某个属性，
                padding = len(self.features_type[x]) - len(x_prob)  # 返回的列表将小于属性的范围
                x_prob_padding = np.pad(x_prob, (0, padding))  # 因此需要用0补足
                conditional_prob_list.append(x_prob_padding)
        return np.array(conditional_prob_list)

    def train(self):
        self.label_metric = self.cal_prior_probability(self.label)
        self.features_metric = self.cal_conditional_probability(self.features, self.label)
        return self.label_metric, self.features_metric  # 用类先验概率和条件概率矩阵作为模型参数

    def test(self, test_data):
        max_prob = 0
        max_prob_label = None
        for i, c in enumerate(self.label_metric):  # 进行最大似然估计
            cum_feature_prob = 1
            for j, x in enumerate(test_data):  # 计算预测其为y_i这一类的各属性累积条件概率
                index = self.features_type[j].index(str(x))
                cum_feature_prob = cum_feature_prob * self.features_metric[j+i*len(test_data)][index]
            if c * cum_feature_prob > max_prob:  # 选择估计值最大的类标记
                max_prob = c * cum_feature_prob
                max_prob_label = self.label[i]
        return max_prob_label


X_train = np.array([[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'], [2, 'S'],
           [2, 'M'], [2, 'M'], [2, 'L'], [2, 'L'], [3, 'L'], [3, 'M'], [3, 'M'],
           [3, 'L'], [3, 'L']])
y_train = np.array([-1, -1, 1, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, -1])

a = NaiveBayes(X_train, y_train)
a.train()
print(a.test([2, 'S']))







