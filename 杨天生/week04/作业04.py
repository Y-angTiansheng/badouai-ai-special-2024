#椒盐噪声
def fun1(src, percentage):
    noise_num = int(percentage * src.shape[0] * src.shape[1])
    rand_coords = np.random.randint(0, src.shape[0] * src.shape[1], noise_num)
    rand_coords = np.unravel_index(rand_coords, src.shape)
    noise_values = np.random.choice([0, 255], size=noise_num)
    flat_indices = np.ravel_multi_index(rand_coords, src.shape)
    src.flat[flat_indices] = noise_values
    return src

#高斯噪声
def GaussianNoise(src, mean, sigma, percentage):
    NoiseNum = int(percentage * src.shape[0] * src.shape[1])
    noise = np.random.normal(mean, sigma, (NoiseNum, 2)).astype(np.int32)
    noise[:, 0] = np.clip(noise[:, 0], 0, src.shape[0] - 1)
    noise[:, 1] = np.clip(noise[:, 1], 0, src.shape[1] - 1)
    for i in range(NoiseNum):
        randX, randY = noise[i]
        src[randX, randY] = np.clip(src[randX, randY] + np.random.normal(mean, sigma), 0, 255)
    return src

#pca
class CPCA:
    def __init__(self, X, K, verbose=False):
        self.X = X
        self.K = K
        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()
        if verbose:
            self.print_details()
    def _centralized(self):
        mean = np.mean(self.X, axis=0)
        centrX = self.X - mean
        return centrX
    def _cov(self):
        ns = self.centrX.shape[0]
        C = np.cov(self.centrX, rowvar=False) / (ns - 1)
        return C
    def _U(self):
        a, b = np.linalg.eig(self.C)
        ind = np.argsort(-a)
        U = b[:, ind[:self.K]]
        return U
    def _Z(self):
        Z = np.dot(self.X, self.U)
        return Z
    def print_details(self):
        print('样本矩阵X:\n', self.X)
        print('样本集的特征均值:\n', np.mean(self.X, axis=0))
        print('样本矩阵X的中心化centrX:\n', self.centrX)
        print('样本矩阵X的协方差矩阵C:\n', self.C)
        print('样本集的协方差矩阵C的特征值:\n', np.linalg.eigvals(self.C))
        print('%d阶降维转换矩阵U:\n' % self.K, self.U)
        print('样本矩阵X的降维矩阵Z:\n', self.Z)
if __name__ == '__main__':
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = X.shape[1] - 1
    pca = CPCA(X, K, verbose=True)
