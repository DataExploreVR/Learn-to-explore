from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class EmbedPCA():

    def __init__(self, raw_data, target_dim=2):
        self.SS = StandardScaler()
        self.SS.fit(raw_data)
        self.PCA = PCA(n_components=target_dim)
        self.PCA.fit(self.SS.transform(raw_data))

    def transform(self, data):
        return self.PCA.transform(self.SS.transform(data))
