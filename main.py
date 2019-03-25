import pandas as pd
import sys, os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from networkx import Graph, spring_layout, draw_networkx, draw_networkx_edge_labels, get_edge_attributes

"""
1: Square (A is diag)
2: Linear (A is isolated)
3: Parallel (A is 1-6, B is center): [1 for _ in range(8)] + [0.5 for _ in range(10)] + [0 for _ in range(7)]
4: Demon (A connects stars)
5: Paris (A is inner circle): [1 for _ in range(13)] + [0.5 for _ in range(12)]
6: Solway (A is smaller, B is center)
7: Inception (A is center, B is next, C is smallest)

# 2018 data
hdata = pd.read_csv('human/human_data.csv')
hdata = [34, 19, 31, 34, 10, 37,  6]
mdata = [21, 31,  5, 30, 18, 28, 14] # alpha = 2.0, no extra terms, 100-100
mdata = [19, 25, 18, 25, 14, 30, 26] # alpha = 2.0, no extra terms, 10-10
mdata = [20, 20, 22, 27,  6, 17, 17] # alpha = 2.0, extra terms, 10-10
mdata = [20,  3, 27, 30,  5, 33, 11] # alpha = 2.0, extra terms, 100-100
"""

class Data:
    """ Defines a low-level graph G = (V, E). """

    def __init__(self, fname, fully_observed=True):
        """ Creates representations for data depending on level of observation. """
        if fully_observed:
            self.load_full_graph(fname)
        else:
            self.load_partial_graph(fname)

    def load_full_graph(self, fname):
        """ Creates adjacency matrix graph representation. """
        with open(fname) as f:
            self.name = f.readline().rstrip('\n')
            self.N, self.M, self.UK = list(map(int, f.readline().split()))
            self.M += self.UK
            self.graph = [[0 for ny in range(self.N)] for nx in range(self.N)]
            self.E = []
            for m in range(self.M):
                u, v = list(map(int, f.readline().split()))
                self.graph[u-1][v-1] = 1
                self.graph[v-1][u-1] = 1
                self.E.append((u, v))
        f.close()

    def load_partial_graph(self, fname):
        """ Creates adjacency matrix graph representation and notes candidate edges to unveil. """
        with open(fname) as f:
            self.name = f.readline().rstrip('\n')
            self.N, self.M, n_unknown = list(map(int, f.readline().split()))
            self.graph = [[0 for ny in range(self.N)] for nx in range(self.N)]
            self.E = []
            self.NE = []
            self.UK = []
            for m in range(self.M):
                u, v = list(map(int, f.readline().split()))
                self.graph[u-1][v-1] = 1
                self.graph[v-1][u-1] = 1
                self.E.append((u, v))
            for k in range(n_unknown):
                u, v = list(map(int, f.readline().split()))
                self.graph[u-1][v-1] = -1
                self.graph[v-1][u-1] = -1
                self.UK.append((u, v))
            for i in range(self.N):
                for j in range(i - 1):
                    if self.graph[i][j] == 0:
                        self.NE.append((i+1, j+1))
        f.close()


class HierarchyModel:
    """ Creates a hierarchy H = (G, c, p, q, p') and performs inference. """

    def __init__(self, D, h):
        """ Creates hierarchy representation. """
        self.alpha = h['alpha']
        self.p = stats.beta.rvs(a=1, b=1)
        self.q = stats.beta.rvs(a=1, b=1)
        self.hp = stats.beta.rvs(a=1, b=1) # p' in the original generative model
        self.D = D

        # Create cluster assignment using Chinese Restaurant Process.
        self.c = np.array([1])
        self.cnt = np.array([1])
        for i in range(1, self.D.N):
            probs = np.concatenate((self.cnt, np.array([self.alpha])), axis=0) / (sum(self.cnt) + self.alpha)
            c_i = np.where(np.random.multinomial(1, probs) == 1)[0][0]
            self.c = np.append(self.c, c_i + 1)
            if c_i + 1 > len(self.cnt):
                self.cnt = np.append(self.cnt, 1)
            else:
                self.cnt[c_i] += 1
        self.nclusters = len(self.cnt)

    def log_prior(self):
        """ Computes log P(H) = log P(c) + log P(p) + log P(q) + log P(p'). """
        logp = 0
        curr = np.zeros(self.nclusters)
        curr[self.c[0]-1] = 1
        for i in range(1, self.D.N):
            c_i = self.c[i]
            if curr[c_i-1] == 0:
                logp += np.log(self.alpha) - np.log(sum(curr) + self.alpha)
            else:
                logp += np.log(curr[c_i-1]) - np.log(sum(curr) + self.alpha)
            curr[c_i-1] += 1
        assert(np.array_equal(curr, self.cnt))

        return logp + stats.beta.logpdf(self.p, a=1, b=1) + stats.beta.logpdf(self.q, a=1, b=1) + stats.beta.logpdf(self.hp, a=1, b=1)

    def log_full_edges(self):
        """ Computes log P(E|c, p, q) without unknown edges. """
        graph = np.array(self.D.graph)
        not_same_cluster = np.apply_along_axis(lambda e: self.c[e[0]] == self.c[e[1]], 0, np.indices(graph.shape))
        pqs = np.full(graph.shape, self.p) * np.maximum(not_same_cluster, self.q)
        temp1 = graph - 1
        temp2 = np.log(temp1 * 2 * pqs + pqs + temp1 * -1)
        return np.sum(np.triu(temp2, k=1))

    def log_singleton_is_edge(self, edge, c, p, q):
        """ Computes P(e|H) for a single observed edge. """
        u, v = edge
        if c[u-1] == c[v-1]:
            return np.log(p)
        else:
            return np.log(p * q)

    def log_singleton_is_not_edge(self, edge, c, p, q):
        """ Computes P(not e|H) for a single observed edge. """
        u, v = edge
        if c[u-1] == c[v-1]:
            return np.log(1 - p)
        else:
            return np.log(1 - p * q)

    def log_partial_edges(self):
        """ Computes log P(E|c, p, q) with unknown edges. """
        logp = 0
        for edge in self.D.E:
            logp += self.log_singleton_is_edge(edge, self.c, self.p, self.q)
        for edge in self.D.NE:
            logp += self.log_singleton_is_not_edge(edge, self.c, self.p, self.q)
        return logp

    def log_likelihood(self, with_unknown, extra=True):
        """ Computes P(D|H) = log P(E|c, p, q) + log (E'|p') + log (b|E', c). """
        logp = 0
        if with_unknown:
            logp += self.log_partial_edges()
        else:
            logp += self.log_full_edges()

        if extra:
            # Calculating log (E'|p').
            has_bridges = dict()
            clusters = np.nonzero(self.cnt)
            for j in range(len(clusters)):
                for k in range(j):
                    has_bridges[(clusters[k] + 1, clusters[j] + 1)] = 0
            for u, v in self.D.E:
                has_bridges[(self.c[u-1], self.c[v-1])] = 1
            for val in has_bridges.values():
                if val == 0:
                    logp += np.log(1 - self.hp)
                else:
                    logp += np.log(self.hp)

            # Calculating log (b|E', c).
            for key, val in has_bridges.items():
                if val == 1:
                    logp += np.log(1 / (self.cnt[key[0]-1] * self.cnt[key[1]-1]))
                    logp -= np.log(self.p * self.q)

        return logp

    def log_posterior(self, with_unknown=True):
        """ Computes log P(H|D) = log P(D|H) + log P(H). """
        return self.log_prior() + self.log_likelihood(with_unknown=with_unknown)

    def update_cluster(self, n, newc):
        """ Updates cluster of node n to newc. """
        if newc == self.c[n]:
            return
        self.cnt[self.c[n]-1] -= 1
        self.c[n] = newc
        if newc > self.nclusters:
            self.cnt = np.append(self.cnt, 1)
            self.nclusters += 1
        else:
            self.cnt[newc-1] += 1

    def offline_sampler(self, nsamples=5000, nburnin=1000, nsampintv=1, with_unknown=True):
        """ Offline sampling of H using Metropolis-Hastings-within-Gibbs. """

        self.accepts = 0
        self.logpost_samples = []
        self.hsamples = dict()
        for t in range(nburnin + nsamples * nsampintv):

            # Sample c using Gibbs sampling for Dirichlet processes as in Neal (1998).
            for i in range(self.D.N):
                old_i = self.c[i]
                proposal_dist = self.cnt.copy()
                proposal_dist[self.c[i]-1] -= 1
                proposal_dist = np.append(proposal_dist, self.alpha)
                proposal_dist /= (self.D.N - 1 + self.alpha)
                cand_i = np.where(np.random.multinomial(1, proposal_dist) == 1)[0][0] + 1
                old_logpost = self.log_posterior(with_unknown=with_unknown)
                self.update_cluster(i, cand_i)
                new_logpost = self.log_posterior(with_unknown=with_unknown)
                log_accept_ratio = min(0, new_logpost - old_logpost)
                if np.log(np.random.uniform()) < log_accept_ratio:
                    self.accepts += 1
                else:
                    self.update_cluster(i, old_i)

            # Sample p.
            old_p = self.p
            ll, uu = stats.norm.cdf(0, loc=old_p, scale=0.1), stats.norm.cdf(1, loc=old_p, scale=0.1)
            cand_p = stats.norm.ppf(stats.uniform.rvs(loc=ll, scale=uu-ll), loc=old_p, scale=0.1)
            assert(cand_p > 0 and cand_p < 1)
            old_logpost = self.log_posterior(with_unknown=with_unknown)
            self.p = cand_p
            new_logpost = self.log_posterior(with_unknown=with_unknown)
            cdf_corr_new = np.log(stats.norm.cdf(1, loc=cand_p, scale=0.1) - stats.norm.cdf(0, loc=cand_p, scale=0.1))
            cdf_corr_old = np.log(stats.norm.cdf(1, loc=old_p, scale=0.1) - stats.norm.cdf(0, loc=old_p, scale=0.1))
            log_accept_ratio = min(0, new_logpost - old_logpost + cdf_corr_new - cdf_corr_old)
            if np.log(np.random.uniform()) < log_accept_ratio:
                self.accepts += 1
            else:
                self.p = old_p

            # Sample q.
            old_q = self.q
            ll, uu = stats.norm.cdf(0, loc=old_q, scale=0.1), stats.norm.cdf(1, loc=old_q, scale=0.1)
            cand_q = stats.norm.ppf(stats.uniform.rvs(loc=ll, scale=uu-ll), loc=old_q, scale=0.1)
            assert(cand_q > 0 and cand_q < 1)
            old_logpost = self.log_posterior(with_unknown=with_unknown)
            self.q = cand_q
            new_logpost = self.log_posterior(with_unknown=with_unknown)
            cdf_corr_new = np.log(stats.norm.cdf(1, loc=cand_q, scale=0.1) - stats.norm.cdf(0, loc=cand_q, scale=0.1))
            cdf_corr_old = np.log(stats.norm.cdf(1, loc=old_q, scale=0.1) - stats.norm.cdf(0, loc=old_q, scale=0.1))
            log_accept_ratio = min(0, new_logpost - old_logpost + cdf_corr_new - cdf_corr_old)
            if np.log(np.random.uniform()) < log_accept_ratio:
                self.accepts += 1
            else:
                self.q = old_q

            # Sample p'.
            old_hp = self.hp
            ll, uu = stats.norm.cdf(0, loc=old_hp, scale=0.1), stats.norm.cdf(1, loc=old_hp, scale=0.1)
            cand_hp = stats.norm.ppf(stats.uniform.rvs(loc=ll, scale=uu-ll), loc=old_hp, scale=0.1)
            assert(cand_hp > 0 and cand_hp < 1)
            old_logpost = self.log_posterior(with_unknown=with_unknown)
            self.hp = cand_hp
            new_logpost = self.log_posterior(with_unknown=with_unknown)
            cdf_corr_new = np.log(stats.norm.cdf(1, loc=cand_hp, scale=0.1) - stats.norm.cdf(0, loc=cand_hp, scale=0.1))
            cdf_corr_old = np.log(stats.norm.cdf(1, loc=old_hp, scale=0.1) - stats.norm.cdf(0, loc=old_hp, scale=0.1))
            log_accept_ratio = min(0, new_logpost - old_logpost + cdf_corr_new - cdf_corr_old)
            if np.log(np.random.uniform()) < log_accept_ratio:
                self.accepts += 1
            else:
                self.hp = old_hp

            # Statistics and collection.
            self.logpost_samples.append(self.log_posterior(with_unknown=with_unknown))
            if t < nburnin:
                if (t + 1) % 1000 == 0:
                    print(f'{t+1} samples burnt-in. Acceptance rate: {100 * (self.accepts / ((t+1) * (self.D.N + 2))):.2f}%.')
            else:
                if (t + 1 - nburnin) % nsampintv == 0:
                    self.hsamples[t] = (self.c.copy(), self.cnt.copy(), self.p, self.q, self.hp)
                if (t + 1 - nburnin) % (1000 * nsampintv) == 0:
                    print(f'{(t + 1 - nburnin) / nsampintv} samples collected. Acceptance rate: {100 * (self.accepts / ((t+1) * (self.D.N + 3))):.2f}%.')

        self.bestk_samples(nburnin, nsamples, nsampintv)

    def bestk_samples(self, nburnin, nsamples, nsampintv):
        """ Sort the samples by P(H|D). """
        self.best_hsamples = []
        valids = np.argsort(np.array(self.logpost_samples)[nburnin+nsampintv-1::nsampintv])
        assert(len(valids) == nsamples)
        for k in valids:
            self.best_hsamples.append(self.hsamples[nburnin + (k+1) * nsampintv - 1])

    def init_particles(self):
        """ Initialize particles using P(H). """
        self.particles = []
        for _ in range(self.NP):
            samp_p = stats.beta.rvs(a=1, b=1)
            samp_q = stats.beta.rvs(a=1, b=1)
            samp_hp = stats.beta.rvs(a=1, b=1)
            samp_c = np.array([1])
            samp_cnt = np.array([1])
            for i in range(1, self.D.N):
                probs = np.concatenate((samp_cnt, np.array([self.alpha])), axis=0) / (sum(samp_cnt) + self.alpha)
                c_i = np.where(np.random.multinomial(1, probs) == 1)[0][0]
                samp_c = np.append(samp_c, c_i + 1)
                if c_i + 1 > len(samp_cnt):
                    samp_cnt = np.append(samp_cnt, 1)
                else:
                    samp_cnt[c_i] += 1
            self.particles.append((samp_c, samp_cnt, samp_p, samp_q, samp_hp))

        self.weights = np.array([1 / self.NP for _ in range(self.NP)])

    def update_weights(self, edge, is_in=True):
        """ Feeds a single observation and updates the importance weights accordingly. """
        for k in range(self.NP):
            c, cnt, p, q, hp = self.particles[k]
            if is_in:
                logp = self.log_singleton_is_edge(edge, c, p, q)
            else:
                logp = self.log_singleton_is_not_edge(edge, c, p, q)
            self.weights[k] *= np.exp(logp)
        self.weights /= np.sum(self.weights)

    def rejuvenate(self, niter=10):
        """ Perform iterations of MH-within-Gibbs and update particles. """

        # Perform `niter` steps of MHwG independently for each particle.
        for t in range(1, niter + 1):
            for k in range(self.NP):
                self.c, self.cnt = self.particles[k][0].copy(), self.particles[k][1].copy()
                self.p, self.q, self.hp = self.particles[k][2], self.particles[k][3], self.particles[k][4]
                self.nclusters = len(self.cnt)

                for i in range(self.D.N):
                    old_i = self.c[i]
                    proposal_dist = self.cnt.copy()
                    proposal_dist[self.c[i]-1] -= 1
                    proposal_dist = np.append(proposal_dist, self.alpha)
                    proposal_dist /= (self.D.N - 1 + self.alpha)
                    cand_i = np.where(np.random.multinomial(1, proposal_dist) == 1)[0][0] + 1
                    old_logpost = self.log_posterior()
                    self.update_cluster(i, cand_i)
                    new_logpost = self.log_posterior()
                    log_accept_ratio = min(0, new_logpost - old_logpost)
                    if np.log(np.random.uniform()) > log_accept_ratio:
                        self.update_cluster(i, old_i)

                old_p = self.p
                ll, uu = stats.norm.cdf(0, loc=old_p, scale=0.1), stats.norm.cdf(1, loc=old_p, scale=0.1)
                cand_p = stats.norm.ppf(stats.uniform.rvs(loc=ll, scale=uu-ll), loc=old_p, scale=0.1)
                old_logpost = self.log_posterior()
                self.p = cand_p
                new_logpost = self.log_posterior()
                cdf_corr_new = np.log(stats.norm.cdf(1, loc=cand_p, scale=0.1) - stats.norm.cdf(0, loc=cand_p, scale=0.1))
                cdf_corr_old = np.log(stats.norm.cdf(1, loc=old_p, scale=0.1) - stats.norm.cdf(0, loc=old_p, scale=0.1))
                log_accept_ratio = min(0, new_logpost - old_logpost + cdf_corr_new - cdf_corr_old)
                if np.log(np.random.uniform()) > log_accept_ratio:
                    self.p = old_p

                old_q = self.q
                ll, uu = stats.norm.cdf(0, loc=old_q, scale=0.1), stats.norm.cdf(1, loc=old_q, scale=0.1)
                cand_q = stats.norm.ppf(stats.uniform.rvs(loc=ll, scale=uu-ll), loc=old_q, scale=0.1)
                old_logpost = self.log_posterior()
                self.q = cand_q
                new_logpost = self.log_posterior()
                cdf_corr_new = np.log(stats.norm.cdf(1, loc=cand_q, scale=0.1) - stats.norm.cdf(0, loc=cand_q, scale=0.1))
                cdf_corr_old = np.log(stats.norm.cdf(1, loc=old_q, scale=0.1) - stats.norm.cdf(0, loc=old_q, scale=0.1))
                log_accept_ratio = min(0, new_logpost - old_logpost + cdf_corr_new - cdf_corr_old)
                if np.log(np.random.uniform()) > log_accept_ratio:
                    self.q = old_q

                old_hp = self.hp
                ll, uu = stats.norm.cdf(0, loc=old_hp, scale=0.1), stats.norm.cdf(1, loc=old_hp, scale=0.1)
                cand_hp = stats.norm.ppf(stats.uniform.rvs(loc=ll, scale=uu-ll), loc=old_hp, scale=0.1)
                assert(cand_hp > 0 and cand_hp < 1)
                old_logpost = self.log_posterior()
                self.hp = cand_hp
                new_logpost = self.log_posterior()
                cdf_corr_new = np.log(stats.norm.cdf(1, loc=cand_hp, scale=0.1) - stats.norm.cdf(0, loc=cand_hp, scale=0.1))
                cdf_corr_old = np.log(stats.norm.cdf(1, loc=old_hp, scale=0.1) - stats.norm.cdf(0, loc=old_hp, scale=0.1))
                log_accept_ratio = min(0, new_logpost - old_logpost + cdf_corr_new - cdf_corr_old)
                if np.log(np.random.uniform()) < log_accept_ratio:
                    self.accepts += 1
                else:
                    self.hp = old_hp

                self.particles[k] = (self.c.copy(), self.cnt.copy(), self.p, self.q, self.hp)

        # Re-sample new particles and reset the weights to 1/NP.
        new_indices = np.random.multinomial(self.NP, self.weights)
        new_particles = []
        for j in range(self.NP):
            for _ in range(new_indices[j]):
                new_particles.append(self.particles[j])
        self.particles = new_particles
        self.weights = np.array([1 / self.NP for _ in range(self.NP)])

    def online_sampler(self, NP=50000):
        """ Online sampling of H using importance sampling + particle filtering. """
        self.NP = NP
        self.init_particles()
        for edge in self.D.E:
            self.update_weights(edge, is_in=True)
            #self.rejuvenate()
        for edge in self.D.NE:
            self.update_weights(edge, is_in=False)
            #self.rejuvenate()

    def unveil(self, phdtype='full'):
        """ Decides which edge to unveil, from the list of unknown edges.
            For online inference, phdtype = 'full' or 'weight' to compute P(H|D).
        """
        def olp(particle):
            self.c, self.cnt, self.p, self.q, self.hp = particle
            self.nclusters = len(self.cnt)
            return self.log_posterior()

        def add_uk(edge, particle, is_in):
            c, cnt, p, q, hp = particle
            if is_in:
                return self.log_singleton_is_edge(edge, c, p, q)
            return self.log_singleton_is_not_edge(edge, c, p, q)

        self.action_entropys = []
        for edge in self.D.UK:
            # Compute H(H|D with edge) * P(edge|D).
            if phdtype == 'full':
                self.D.E.append(edge)
                logposts = np.array(list(map(olp, self.particles)))
            else:
                uklogp = np.array(list(map(lambda pp: add_uk(edge, pp, True), self.particles)))
                uklogpd = self.weights * np.exp(uklogp)
                uklogpd /= np.sum(uklogpd)
                logposts = np.log(uklogpd)
            with_entropy = -1 * np.sum(logposts * self.weights)
            logpeds = np.array(list(map(lambda pc: np.exp(self.log_singleton_is_edge(edge, pc[0], pc[2], pc[3])), self.particles)))
            with_prob = np.sum(logpeds * self.weights)
            if phdtype == 'full':
                self.D.E = self.D.E[:-1]

            # Compute H(H|D without edge) * P(no edge|D).
            if phdtype == 'full':
                self.D.NE.append(edge)
                logposts = np.array(list(map(olp, self.particles)))
            else:
                uklogp = np.array(list(map(lambda pp: add_uk(edge, pp, False), self.particles)))
                uklogpd = self.weights * np.exp(uklogp)
                uklogpd /= np.sum(uklogpd)
                logposts = np.log(uklogpd)
            without_entropy = -1 * np.sum(logposts * self.weights)
            logpeds = np.array(list(map(lambda pc: np.exp(self.log_singleton_is_not_edge(edge, pc[0], pc[2], pc[3])), self.particles)))
            without_prob = np.sum(logpeds * self.weights)
            if phdtype == 'full':
                self.D.NE = self.D.NE[:-1]

            self.action_entropys.append(with_entropy * with_prob + without_entropy * without_prob)

        # for j in range(len(self.D.UK)):
        #     print(f'For edge {self.D.UK[j]}, H(H|D, a) = {self.action_entropys[j]}.')

    def plot_partial_graph(self, action='save'):
        """ Plot graphs for edge unveiling. """
        uid = 200
        while os.path.isfile(f'{uid}_{self.D.name}.png'):
            uid += 1

        r = lambda: np.random.randint(low=0, high=256)
        ng = Graph()
        for i in range(self.D.N):
            ng.add_node(i + 1)
        for u, v in self.D.E:
            ng.add_edge(u, v, entropy='')
        for w in range(len(self.D.UK)):
            ng.add_edge(self.D.UK[w][0], self.D.UK[w][1], entropy='{:.3f}'.format(self.action_entropys[w]))
        colormap = []
        for ee in ng.edges:
            if ee in self.D.UK:
                colormap.append('red')
            else:
                colormap.append('black')
        plt.figure()
        plt.title('Edge Unveiling')
        pos = spring_layout(ng)
        draw_networkx(ng, pos, node_color='#D3D3D3', edge_color=colormap)
        edge_labels = get_edge_attributes(ng, 'entropy')
        draw_networkx_edge_labels(ng, pos, edge_labels=edge_labels)
        if action == 'save':
            plt.savefig(f'{uid}_{self.D.name}.png')
            plt.close()
        elif action == 'show':
            plt.show()

    def plot_graphs(self, action='save'):
        """ Plot various statistics. """
        uid = 100
        while os.path.isfile(f'{uid}_{self.D.name}_hgraph.png'):
            uid += 1

        # Log-posterior graph.
        # plt.figure()
        # plt.rc('font',size=14)
        # plt.rc('axes',titlesize=14)
        # plt.rc('axes',labelsize=14)
        # plt.title(f'Log-Posterior')
        # plt.xlabel('Iterations')
        # plt.ylabel('P(H|D)')
        # plt.plot(self.logpost_samples)
        # if action == 'save':
        #     plt.savefig(f'{uid}_{self.D.name}_logpost.png')
        #     plt.close()
        # elif action == 'show':
        #     plt.show()

        # Displaying H.
        for bidx in range(1, 2):
            bestc = self.best_hsamples[-bidx][0]
            print(f'No. {bidx} clustering: {bestc}.')
            colormap = dict()
            r = lambda: np.random.randint(low=0, high=256)
            for k in range(1, max(bestc) + 1):
                colormap[k] = '#{:02x}{:02x}{:02x}'.format(r(), r(), r())
            ng = Graph()
            node_colors = []
            for i in range(self.D.N):
                ng.add_node(i + 1)
                node_colors.append(colormap[bestc[i]])
            ng.add_edges_from(self.D.E)
            plt.figure()
            #plt.title('Hierarchical Clusters')
            draw_networkx(ng, node_color=node_colors)
            plt.axis('off')
            if action == 'save':
                plt.savefig(f'{uid}_{self.D.name}_hgraph.png')
                plt.close()
            elif action == 'show':
                plt.show()


def demo1(fname, nsamples=500, nburnin=0, alpha=2.0):
    """ Perform offline inference on a given (completely observed) graph and display results. """
    h = {'alpha': alpha}
    D = Data(fname, fully_observed=False)
    hm = HierarchyModel(D, h)
    for u, v in hm.D.UK:
        hm.D.E.append((u, v))
        hm.offline_sampler(nsamples=nsamples, nburnin=nburnin, with_unknown=True)
        hm.plot_graphs()
        hm.D.E = hm.D.E[:-1]

def demo2(fname, inference_type='offline', nsamples=100, nburnin=100, nlag=10, alpha=2.0):
    """ Perform online inference on a partially observed graph, and make a decision on which edge to unveil next. """
    h = {'alpha': alpha}
    D = Data(fname, fully_observed=False)
    hm = HierarchyModel(D, h)
    if inference_type == 'offline':
        hm.offline_sampler(nsamples=nsamples, nburnin=nburnin, nsampintv=nlag)
        #print('Top particles after offline sampling:')
        #for top in hm.best_hsamples[-5:]:
        #    print(top[0])
        hm.particles = hm.best_hsamples
        hm.NP = len(hm.particles)
        hm.weights = np.array([1 / hm.NP for _ in range(hm.NP)])
    else:
        hm.online_sampler(NP=nsamples)
        #print('Top particles after online sampling:')
        #for top in np.argsort(hm.weights)[-5:]:
        #    print(hm.particles[top][0])
    #print('Unveiling edges...')
    hm.unveil()

    # Statistics
    #hm.plot_partial_graph()
    namekey = fname.split('/')[1].split('.')[0]
    with open(f'2019_results/results2_{namekey}.txt', 'a+') as f:
        for j in range(len(hm.D.UK)):
            f.write(f'Edge {hm.D.UK[j]}: {hm.action_entropys[j]}\n')
        f.write('\n')
    f.close()
    return hm.action_entropys

def run_model(to_run, hp, repeats=40):
    flist = []
    names = ['square', 'linear', 'parallel', 'partial_demon', 'paris', 'partial_solway1', 'inception']
    edgenums = [0, 1, 1, 0, 1, 0, 1]
    for n in to_run:
        flist.append((f'data/{names[n-1]}.txt', edgenums[n-1]))
    #flist.append(('data/linear.txt', 0))    #A
    #flist.append(('data/linear.txt', 1)) #A
    #flist.append(('data/parallel.txt', 1)) #B
    #flist.append(('data/partial_demon.txt', 0)) #A
    #flist.append(('data/paris.txt', 1)) #A
    #flist.append(('data/partial_solway1.txt', 0)) #B
    #flist.append(('data/inception.txt', 1)) #B
    for file, num in flist:
        print(file)
        count = 0
        for i in range(repeats):
            entropys = demo2(file, nsamples=hp['nsamples'], nburnin=hp['nburnin'], nlag=hp['nlag'], alpha=hp['alpha'])
            if min(entropys) == entropys[num]:
                count += 1
            print(f'{i}')
        print('\nCount: ', count)

def plot_model_results(mdata, fname, N=40, action='save'):
    """ Plot model results. """

    # 2 edges.
    plt.figure()
    plt.rc('font',size=14)
    plt.rc('axes',titlesize=14)
    plt.rc('axes',labelsize=14)
    plt.title(f'Model')
    edge = 'A'
    perc = 0.5
    plt.ylabel(f'Fraction of Trials where Edge {edge} unveiled')
    ax = plt.gca()
    plt.ylim((0.0, 1.2))
    plt.xlim((0.0, 5))
    allresults = []
    allerrs = []
    for gnum in [1, 4, 2, 5]:
        edgenum = mdata[gnum-1]
        results = [1 for _ in range(edgenum)] + [0 for _ in range(N-edgenum)]
        assert(len(results) == N)
        allresults.append(sum(results)/N)
        allerrs.append(np.std(results, ddof=1) /(N ** 0.5))
    ax.bar([1, 2, 3, 4], allresults, yerr=allerrs, width=0.5, color='#07538F', capsize=5)
    plt.xticks([1, 2, 3, 4], ['Graph 1', 'Graph 4', 'Graph 2', 'Graph 5'])
    #low95, high95 = stats.binom.ppf(0.025,n=N, p=perc) / N, stats.binom.ppf(0.975,n=N, p=perc) / N
    #plt.axhline(y=low95, alpha=0.4, color='#7D8491')
    #plt.axhline(y=high95, alpha=0.4, color='#7D8491')
    #ax.axhspan(low95, high95, facecolor='#7D8491', alpha=0.4)
    plt.axhline(y=perc, linestyle='--', color='#7D8491')
    if action == 'show':
        plt.show()
    else:
        plt.savefig(f'2019_results/model_2e_{fname}.png')

    # 3 edges.
    plt.figure()
    plt.rc('font',size=14)
    plt.rc('axes',titlesize=14)
    plt.rc('axes',labelsize=14)
    plt.title(f'Model')
    edge = 'B'
    perc = 1/3
    plt.ylabel(f'Fraction of Trials where Edge {edge} unveiled')
    ax = plt.gca()
    plt.ylim((0.0, 1.2))
    plt.xlim((0.0, 4))
    allresults = []
    allerrs = []
    for gnum in [6, 3, 7]:
        edgenum = mdata[gnum-1]
        results = [1 for _ in range(edgenum)] + [0 for _ in range(N-edgenum)]
        assert(len(results) == N)
        allresults.append(sum(results)/N)
        allerrs.append(np.std(results, ddof=1) /(N ** 0.5))
    ax.bar([1, 2, 3], allresults, yerr=allerrs, width=0.5, color='#07538F', capsize=5)
    plt.xticks([1, 2, 3], ['Graph 6', 'Graph 3', 'Graph 7'])
    #low95, high95 = stats.binom.ppf(0.025,n=N, p=perc) / N, stats.binom.ppf(0.975,n=N, p=perc) / N
    #plt.axhline(y=low95, alpha=0.4, color='#7D8491')
    #plt.axhline(y=high95, alpha=0.4, color='#7D8491')
    #ax.axhspan(low95, high95, facecolor='#7D8491', alpha=0.4)
    plt.axhline(y=perc, linestyle='--', color='#7D8491')
    if action == 'show':
        plt.show()
    else:
        plt.savefig(f'2019_results/model_3e_{fname}.png')

def plot_human_results(N=40, action='save'):
    """ Plot model results. """

    # 2 edges.
    plt.figure()
    plt.rc('font',size=14)
    plt.rc('axes',titlesize=14)
    plt.rc('axes',labelsize=14)
    plt.title(f'Data')
    edge = 'A'
    perc = 0.5
    plt.ylabel(f'Fraction of Trials where Edge {edge} unveiled')
    ax = plt.gca()
    plt.ylim((0.0, 1.2))
    plt.xlim((0.0, 5))
    allresults = []
    allerrs = []
    for gnum in [1, 4, 2, 5]:
        edgenum = len(hdata[hdata['G' + str(gnum)] == edge])
        results = [1 for _ in range(edgenum)] + [0 for _ in range(N-edgenum)]
        assert(len(results) == N)
        allresults.append(sum(results)/N)
        allerrs.append(np.std(results, ddof=1) /(N ** 0.5))
    ax.bar([1, 2, 3, 4], allresults, yerr=allerrs, width=0.5, color='#07538F', capsize=5)
    plt.xticks([1, 2, 3, 4], ['Graph 1', 'Graph 4', 'Graph 2', 'Graph 5'])
    #low95, high95 = stats.binom.ppf(0.025,n=N, p=perc) / N, stats.binom.ppf(0.975,n=N, p=perc) / N
    #plt.axhline(y=low95, alpha=0.4, color='#7D8491')
    #plt.axhline(y=high95, alpha=0.4, color='#7D8491')
    #ax.axhspan(low95, high95, facecolor='#7D8491', alpha=0.4)
    plt.axhline(y=perc, linestyle='--', color='#7D8491')
    if action == 'show':
        plt.show()
    else:
        plt.savefig('results/human_2e.png')

    # 3 edges.
    plt.figure()
    plt.rc('font',size=14)
    plt.rc('axes',titlesize=14)
    plt.rc('axes',labelsize=14)
    plt.title(f'Data')
    edge = 'B'
    perc = 1/3
    plt.ylabel(f'Fraction of Trials where Edge {edge} unveiled')
    ax = plt.gca()
    plt.ylim((0.0, 1.2))
    plt.xlim((0.0, 4))
    allresults = []
    allerrs = []
    for gnum in [6, 3, 7]:
        edgenum = len(hdata[hdata['G' + str(gnum)] == edge])
        results = [1 for _ in range(edgenum)] + [0 for _ in range(N-edgenum)]
        assert(len(results) == N)
        allresults.append(sum(results)/N)
        allerrs.append(np.std(results, ddof=1) /(N ** 0.5))
    ax.bar([1, 2, 3], allresults, yerr=allerrs, width=0.5, color='#07538F', capsize=5)
    plt.xticks([1, 2, 3], ['Graph 6', 'Graph 3', 'Graph 7'])
    #low95, high95 = stats.binom.ppf(0.025,n=N, p=perc) / N, stats.binom.ppf(0.975,n=N, p=perc) / N
    #plt.axhline(y=low95, alpha=0.4, color='#7D8491')
    #plt.axhline(y=high95, alpha=0.4, color='#7D8491')
    #ax.axhspan(low95, high95, facecolor='#7D8491', alpha=0.4)
    plt.axhline(y=perc, linestyle='--', color='#7D8491')
    if action == 'show':
        plt.show()
    else:
        plt.savefig('results/human_3e.png')

def compute_pvalues():
    with open('pvals.txt', 'w') as f:
        hdata = [34, 19, 31, 34, 10, 37,  6]
        mdata = [20,  3, 27, 30,  5, 33, 11]
        for i in range(7):
            if i == 2 or i == 5 or i == 6:
                s = stats.binom_test(hdata[i], 40, 1.0/3.0, alternative='two-sided')
            else:
                s = stats.binom_test(hdata[i], 40, 0.5, alternative='two-sided')
            f.write(str(s) + '\n')
        f.write('\n')
        for i in range(7):
            if i == 2 or i == 5 or i == 6:
                s = stats.binom_test(mdata[i], 40, 1.0/3.0, alternative='two-sided')
            else:
                s = stats.binom_test(mdata[i], 40, 0.5, alternative='two-sided')
            f.write(str(s) + '\n')

def plot_confidence():
    ratios = []
    for i in range(1, 8):
        ai = globals()['A' + str(i)]
        bi = globals()['B' + str(i)]
        assert(len(ai) == 25)
        assert(len(bi) == 25)
        ai = np.array(ai)
        bi = np.array(bi)
        try:
            ci = globals()['C' + str(i)]
            assert(len(ci) == 25)
            ci = np.array(ci)
        except:
            ci = None
        maxi = np.maximum(ai, bi)
        mini = np.minimum(ai, bi)
        if type(ci) != type(None):
            maxi = np.maximum(maxi, ci)
            mini = np.minimum(mini, ci)
        ratio = np.mean(maxi / mini) -1
        ratio = np.log(ratio)
        ratios.append(ratio)

    plt.figure(figsize=(10, 2))
    colors = ['#07538F','#07538F','#07538F','#07538F','#07538F','#07538F','#07538F']
    labels = ['1', '2', '3', '4', '5', '6', '7']
    plt.scatter(ratios, np.zeros_like(ratios), c=colors, cmap="hot_r", vmin=-2)
    for label, x, y in zip(labels, ratios, [0,0,0,0,0,0,0]):
        plt.annotate(label, xy=(x, y), xytext=(0, 10), textcoords='offset points')
    plt.yticks([])
    plt.savefig('confidence.png')


######################
## Run model
######################

# Hyperparameters to try:
# 5000 burn-in, 50 (100) lag, 100 samples --> 10000 iters
# alpha 1.0, 1.5, 2.0, 3.0

#hyperparams = {'nsamples': 100, 'nburnin': 5000, 'nlag': 50, 'alpha': 1.0}
#run_model(to_run=[7], hp=hyperparams)
#plot_model_results()

# Results
# Human     {34, 19, 31, 34, 10, 37, 06}
# 1.0e1.0   {36, 00, 29, 37, 00, 31, 12}
# 1.0e0.8   {33, 03, 26, 34, 05, 30, 11}
# 1.0e0.6   {32, 09, 29, 33, 07, 24, 10}
# 1.0te1.0  {20, 00, 29, 26, 00, 31, 12}
# 1.0te0.8  {15, 05, 24, 28, 07, 27, 12}
# 1.0te0.6  {17, 12, 22, 28, 09, 24, 16}

names = ['square', 'linear', 'parallel', 'demon', 'paris', 'solway', 'inception']
choices = [2, 2, 3, 2, 2, 3, 3]
choose = ['A', 'A', 'B', 'A', 'A', 'B', 'B']
mdata = []
mdata.append('ssssssBsssssBsssssssssssssssssBsssssBsss')
mdata.append('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')
mdata.append('BCABBCBBABBBBCBBABCBBBBABCBBBBBBBBCBBBBC')
mdata.append('AssBsBsAAsAsssAssAssAAssAsssAsBAAssAsAAA')
mdata.append('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')
mdata.append('BBBBBBBAAABBBBBBABABBCBBBBABBBBBBBCBBCBB')
mdata.append('BBAABAAAAABAABBAABABBAAAAABBABAAAAAAAAAA')
# inception_extra = 'AABBAABABAA'


def explore(mdata, choices, choose, names, to_tiebreak=False, epsilon=0.6, N=40):
    edata = []
    for i in range(7):
        mgraph = list(mdata[i])
        assert(len(mgraph) == N)
        echoices = ''
        for k in range(N):
            if stats.uniform.rvs() < 1 - epsilon:
                echoices += chr(65 + np.random.randint(choices[i]))
            else:
                if mgraph[k] == 's':
                    if to_tiebreak:
                        echoices += chr(65 + np.random.randint(choices[i]))
                    else:
                        echoices += choose[i]
                else:
                    echoices += mgraph[k]
        assert(len(echoices) == N)
        edata.append(echoices)
    for i in range(7):
        count = len(list(filter(lambda v: v == choose[i], list(edata[i]))))
        print(f'{names[i]}: {count} ({edata[i]})')
    return edata

#explore(mdata, choices, choose, names, False, 0.6)
plot_model_results([17, 12, 22, 28, 9, 24, 16], 'tiebreak_eps60', N=40, action='save')
