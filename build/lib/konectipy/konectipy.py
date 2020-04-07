import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import scipy.linalg as la

class plot():

    def degree_distribution(self,filename, scale='log'):
        G = nx.read_gexf(filename)
        data = [G.degree(n) for n in G.nodes()]
        data = dict(Counter(data))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.grid()
        plt.scatter(list(data.keys()), list(data.values()))
        if scale == 'log':
            ax.set_yscale('log')
            ax.set_xscale('log')
        plt.xlabel("Degree(d)")
        plt.ylabel("Frequency")
        plt.title('Degree Distribution')


    def cumulative_dd(self,filename):
        G = nx.read_gexf(filename)
        M = nx.to_scipy_sparse_matrix(G)
        degrees = M.sum(0).A[0]
        degree_distribution = np.bincount(degrees)
        s = float(degree_distribution.sum())
        cdf = degree_distribution.cumsum(0) / s
        ccdf = 1 - cdf
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.grid()
        plt.plot(range(len(ccdf)), ccdf)
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.ylabel('P(x>=d)')
        plt.xlabel('Degree(d) [vertices]')
        plt.title("Cumulative Degree Distribution")


    def assortativity(self,filename):
        G = nx.read_gexf(filename)
        temp = nx.average_neighbor_degree(G)
        avg_neigh = list(temp.values())
        degree = [G.degree(n) for n in G.nodes()]
        plt.scatter(degree, avg_neigh, s=0.75)
        plt.xlabel("Degree(d)")
        plt.ylabel("Average Neighbour Degree")
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Assortativity')
        plt.show()


    def gini(self,arr):
        sorted_arr = arr.copy()
        sorted_arr.sort()
        n = arr.size
        coef_ = 2. / n
        const_ = (n + 1.) / n
        weighted_sum = sum([(i + 1) * yi for i, yi in enumerate(sorted_arr)])
        return coef_ * weighted_sum / (sorted_arr.sum()) - const_


    def closest_node(self,node1, node2):
        node2 = np.asarray(node2)
        deltas = node2 - node1
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        return np.argmin(dist_2)
    

    def lorenz_curve(self,filename):
        G = nx.read_gexf(filename)
        temp_deg = [G.degree(n) for n in G.nodes()]
        temp_deg.sort()
        X = np.array(temp_deg)
        X_lorenz = X.cumsum() / X.sum()
        X_lorenz = np.insert(X_lorenz, 0, 0)
        X_lorenz[0], X_lorenz[-1]
        fig, ax = plt.subplots(figsize=[6, 6])
        ax.plot(np.arange(X_lorenz.size) / (X_lorenz.size - 1), X_lorenz, color='darkgreen')
        ax.plot([0, 1], [0, 1], color='k', linestyle=":")
        ax.plot([1, 0], [0, 1], color='k', linestyle=":")
        y_value = ['{:,.0f}'.format(x * 100) + '%' for x in ax.get_yticks()]
        x_value = ['{:,.0f}'.format(x * 100) + '%' for x in ax.get_xticks()]
        ax.set_yticklabels(y_value)
        ax.set_xticklabels(x_value)
        lor = []
        temp = np.arange(X_lorenz.size) / (X_lorenz.size - 1)
        lor.append(list(temp))
        temp = X_lorenz
        lor.append(list(temp))
        lor = np.array(lor)
        lor = lor.transpose()
        opp_d = []
        temp = np.arange(0, len(lor), 1)
        temp = [(i / len(lor)) for i in temp]
        opp_d.append(list(temp))
        temp.reverse()
        opp_d.append(temp)
        opp_d = np.array(opp_d)
        opp_d = opp_d.transpose()
        int_point = lor[self.closest_node(opp_d, lor)]
        ax.scatter(int_point[0], int_point[1], color='red')
        ax.set_xlabel("Share of nodes with smallest degree")
        ax.set_ylabel("Share of edges")
        ax.annotate("P = {:,.2f}%".format(int_point[1] * 100),
                    xy=(int_point[0], int_point[1]), xycoords='data',
                    xytext=(0.8, 0), textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="arc3"),
                    )
        ax.set_title("Lorenz Curve")
        ax.text(0.25, 0.2, "G = {:,.2f}%".format(self.gini(X) * 100))


    def spectral_plot(self,filename):
        G = nx.read_gexf(filename)
        A = nx.adjacency_matrix(G)
        N = nx.normalized_laplacian_matrix(G)
        L = nx.laplacian_matrix(G)
        A_eig = la.eigvals(A.toarray())
        A_eig = [round(i.real, -1) for i in A_eig]
        N_eig = la.eigvals(N.toarray())
        N_eig = [round(i.real, -1) for i in N_eig]
        L_eig = la.eigvals(L.toarray())
        L_eig = [round(i.real, -1) for i in L_eig]

        f = plt.figure(figsize=(12, 3))
        ax1 = f.add_subplot(131)
        ax2 = f.add_subplot(132)
        ax3 = f.add_subplot(133)
        ax1.hist(A_eig)
        l1 = ax1.get_xlim()
        ax1.set_xlim(-l1[1], l1[1])
        ax1.set_yscale('log')
        ax1.set_xlabel("Eigenvalue")
        ax1.set_ylabel('Frequency')
        ax1.set_title("Spec Dist of the eigenvalues of A")
        ax2.hist(N_eig)
        l2 = ax2.get_xlim()
        ax2.set_xlim(-l2[1], l2[1])
        ax2.set_yscale('log')
        ax2.set_xlabel("Eigenvalue")
        ax2.set_ylabel('Frequency')
        ax2.set_title("Spec Dist of the eigenvalues of N")
        ax3.hist(L_eig)
        l3 = ax3.get_xlim()
        ax3.set_xlim(-l3[1], l3[1])
        ax3.set_yscale('log')
        ax3.set_xlabel("Eigenvalue")
        ax3.set_ylabel('Frequency')
        ax3.set_title("Spec Dist of the eigenvalues of L")