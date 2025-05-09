import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from sklearn.manifold import TSNE
import scipy.stats

from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer


class Analysis(object):

    def __init__(self,
                 index,
                ):
        self.index = index
        self.data_path = f'feature/{self.index}/{self.index}.csv'
        self.cluster_path = f'feature/{self.index}/cluster_{self.index}.csv'

        self.data = pd.read_csv(self.data_path, index_col=0)
        self.data = scipy.stats.zscore(self.data)


    def xmeans(self):
        xm_c = kmeans_plusplus_initializer(self.data, 5).initialize()
        xm_i = xmeans(data=self.data, initial_centers=xm_c, kmax=20, ccore=True)
        xm_i.process()
        print(f'Number of clusters: {len(xm_i._xmeans__clusters)}')

        z_xm = np.ones(self.data.shape[0])
        for k in range(len(xm_i._xmeans__clusters)):
            z_xm[xm_i._xmeans__clusters[k]] = k
        self.cluster = pd.DataFrame({}, index=self.data.index)
        self.cluster['cluster'] = z_xm
        self.cluster.to_csv(self.cluster_path)

        #plot
        tsne = TSNE(perplexity=50)
        data_2d = pd.DataFrame(tsne.fit_transform(self.data), index=self.data.index, columns=['x', 'y'])
        fig, ax = plt.subplots(figsize=(5, 5))
        plt.scatter(data_2d.iloc[:, 0], data_2d.iloc[:, 1], c=[plt.cm.tab10(int(i)) for i in self.cluster['cluster']], s=5)
        
        # Create legend for clusters
        unique_clusters = sorted(set(self.cluster['cluster']))
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor=plt.cm.tab10(int(i)), 
                  markersize=5, label=f'{int(i)+1}') 
                  for i in unique_clusters]
        plt.legend(handles=legend_elements, loc='best', title='Morphotype', fontsize=8)
        ax.set_aspect('equal')
        plt.setp(ax.get_xticklabels(), size=8)
        plt.setp(ax.get_yticklabels(), size=8)
        plt.savefig(f'feature/{self.index}/xmeans_{self.index}.svg')
        plt.close()


def main():
    analysis = Analysis(index = 1
                        )
    analysis.xmeans()


if __name__ == '__main__':
    main()