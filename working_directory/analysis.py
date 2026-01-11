import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd
import os

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scipy.stats

from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import ternary

from sklearn.linear_model import LinearRegression
import itertools


class Analysis(object):

    def __init__(self,
                 index,
                ):
        self.index = index
        self.data_path = f'feature/{self.index}/{self.index}.csv'
        self.font_prop = FontProperties(fname="arial.ttf", size=8)

        self.cluster_path = f'feature/{self.index}/cluster_{self.index}.csv'

        df = pd.read_csv(self.data_path, index_col=0)
        z = scipy.stats.zscore(df.values, axis=0)
        self.data = pd.DataFrame(z, index=df.index, columns=df.columns)

        self.marker = ['^', 'o', 's', '^', 'o', 's', 'p', 'p', 'D', 'D']
        self.facecolor = [1, 0, 0, 0, 1, 1, 1, 0, 1, 0]


    def xmeans(self):
        if os.path.exists(self.cluster_path):
            self.cluster = pd.read_csv(self.cluster_path, index_col=0)
        else:
            xm_c = kmeans_plusplus_initializer(self.data, 2).initialize()
            xm_i = xmeans(data=self.data, initial_centers=xm_c, kmax=20, ccore=True)
            xm_i.process()
            print(f'Number of clusters: {len(xm_i._xmeans__clusters)}')

            z_xm = np.ones(self.data.shape[0])
            for k in range(len(xm_i._xmeans__clusters)):
                z_xm[xm_i._xmeans__clusters[k]] = k
            self.cluster = pd.DataFrame({}, index=self.data.index)
            self.cluster['cluster'] = z_xm.astype(int)
            self.cluster.to_csv(self.cluster_path)

        #plot
        tsne = TSNE(perplexity=50)
        data_2d = pd.DataFrame(tsne.fit_transform(self.data), index=self.data.index, columns=['x', 'y'])
        data_2d_factor = pd.DataFrame({}, index=self.delta_factor.index, columns=['x', 'y'])
        for i in range(self.delta_factor.shape[0]):
            delta_name = self.delta_factor.index[i]
            data_2d_factor.loc[delta_name] = data_2d.loc[delta_name].values
        fig, ax = plt.subplots(figsize=(5, 5))
        x, y = data_2d.iloc[:, 0], data_2d.iloc[:, 1]
        clusters = self.cluster['cluster']
        for i in np.unique(clusters):
            cluster_mask = clusters == i
            if self.facecolor[i] == 1:
                markerfacecolor = plt.cm.tab10(int(i))
            else:
                markerfacecolor = "white"
            plt.scatter(x[cluster_mask], y[cluster_mask],
                        marker=self.marker[int(i)],
                        edgecolors=[plt.cm.tab10(int(i))],
                        facecolors = markerfacecolor,
                        s=5, label=f"Cluster {i}")
        # Create legend for clusters
        unique_clusters = sorted(set(self.cluster['cluster']))
        legend_elements = [plt.Line2D([0], [0],
                            marker=self.marker[int(i)],
                            color=plt.cm.tab10(int(i)),
                            markerfacecolor=(plt.cm.tab10(int(i)) if self.facecolor[i] == 1 else "white"),
                            markersize=5,
                            label=f'Cluster {int(i)}')
                            for i in unique_clusters]
        plt.legend(handles=legend_elements, loc='best', title='Clusters', prop=self.font_prop)
        ax.set_aspect('equal')
        plt.setp(ax.get_xticklabels(), fontproperties=self.font_prop, size=8)
        plt.setp(ax.get_yticklabels(), fontproperties=self.font_prop, size=8)
        plt.savefig(f'feature/{self.index}/xmeans_{self.index}.svg')
        plt.close()


def main():
    analysis = Analysis(index = 12
                        )
    analysis.xmeans()


if __name__ == '__main__':
    main()