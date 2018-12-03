
import sys, os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from networkx import Graph, spring_layout, draw_networkx, draw_networkx_edge_labels, get_edge_attributes


# Comments:
# 0. Theory in: https://github.com/tomov/chunking/blob/28aefc5ff5066fb0c9b626d0bbfee35b6bd8fdbf/math/chunking.pdf
# 1. Don't think re-using buckets allowed for MHwG (not that the difference is huge I think): http://www.stat.columbia.edu/npbayes/papers/neal_sampling.pdf
# 2a. Given partially observed data, do we ever observe "not an edge"? Or observation always positive? Are nodes always observed?
### Ans: We do, otherwise the trivial flat hierarchy always dominates. Not quite realistic for human behaviour though.
# 2b. Should online updating be done edge by edge?
### Ans: Shouldn't matter if no rejuvenation.
# 3. When rejuvenating, how to/should we update weights?
### Ans: After running MHwG, sample from rejuvenated particles using weights as multinomial dist, then set new weights to 1/NP.
# 4. Eq. (9) which way to calculate log P(H|D) -- update weights or recalculate whole posterior?
### Ans: For online, both should be equivalent if not rejuvenating. Full posterior safer. Use full posterior for offline sampling.


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
            self.N, self.M = list(map(int, f.readline().split()))
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
    """ Creates a hierarchy H = (G, c, p, q) and performs inference. """

    def __init__(self, D, h):
        """ Creates hierarchy representation. """
        self.alpha = h['alpha']
        self.p = stats.beta.rvs(a=1, b=1)
        self.q = stats.beta.rvs(a=1, b=1)
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
        """ Computes log P(H) = log P(c) + log P(p) + log P(q). """
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

        return logp + stats.beta.logpdf(self.p, a=1, b=1) + stats.beta.logpdf(self.q, a=1, b=1)

    def log_likelihood(self):
        """ Computes log P(D|H) = P(E|c, p, q). """
        graph = np.array(self.D.graph)
        not_same_cluster = np.apply_along_axis(lambda e: self.c[e[0]] == self.c[e[1]], 0, np.indices(graph.shape))
        pqs = np.full(graph.shape, self.p) * np.maximum(not_same_cluster, self.q)
        temp1 = graph - 1
        temp2 = np.log(temp1 * 2 * pqs + pqs + temp1 * -1)
        return np.sum(np.triu(temp2, k=1))

    def log_posterior(self, inference_type='offline'):
        """ Computes log P(H|D) = log P(D|H) + log P(H). """
        if inference_type == 'offline':
            return self.log_prior() + self.log_likelihood()
        return self.observed_log_posterior()

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

    def offline_sampler(self, nsamples=5000, nburnin=1000, nsampintv=1, inference_type='offline'):
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
                old_logpost = self.log_posterior(inference_type=inference_type)
                self.update_cluster(i, cand_i)
                new_logpost = self.log_posterior(inference_type=inference_type)
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
            old_logpost = self.log_posterior(inference_type=inference_type)
            self.p = cand_p
            new_logpost = self.log_posterior(inference_type=inference_type)
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
            old_logpost = self.log_posterior(inference_type=inference_type)
            self.q = cand_q
            new_logpost = self.log_posterior(inference_type=inference_type)
            cdf_corr_new = np.log(stats.norm.cdf(1, loc=cand_q, scale=0.1) - stats.norm.cdf(0, loc=cand_q, scale=0.1))
            cdf_corr_old = np.log(stats.norm.cdf(1, loc=old_q, scale=0.1) - stats.norm.cdf(0, loc=old_q, scale=0.1))
            log_accept_ratio = min(0, new_logpost - old_logpost + cdf_corr_new - cdf_corr_old)
            if np.log(np.random.uniform()) < log_accept_ratio:
                self.accepts += 1
            else:
                self.q = old_q

            # Statistics and collection.
            self.logpost_samples.append(self.log_posterior(inference_type=inference_type))
            if t < nburnin:
                if (t + 1) % 100 == 0:
                    print(f'{t+1} samples burnt-in. Acceptance rate: {100 * (self.accepts / ((t+1) * (self.D.N + 2))):.2f}%.')
            else:
                if (t + 1 - nburnin) % nsampintv == 0:
                    self.hsamples[t] = (self.c.copy(), self.cnt.copy(), self.p, self.q)
                if (t + 1 - nburnin) % (100 * nsampintv) == 0:
                    print(f'{(t + 1 - nburnin) / nsampintv} samples collected. Acceptance rate: {100 * (self.accepts / ((t+1) * (self.D.N + 2))):.2f}%.')

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
            self.particles.append((samp_c, samp_cnt, samp_p, samp_q))

        self.weights = np.array([1 / self.NP for _ in range(self.NP)])

    def singleton_log_likelihood(self, edge, c, p, q):
        """ Computes P(e|H) for a single observed edge. """
        u, v = edge
        if c[u-1] == c[v-1]:
            return np.log(p)
        else:
            return np.log(p * q)

    def singleton_neg_log_likelihood(self, edge, c, p, q):
        """ Computes P(not e|H) for a single observed edge. """
        u, v = edge
        if c[u-1] == c[v-1]:
            return np.log(1 - p)
        else:
            return np.log(1 - p * q)

    def update_weights(self, edge, is_in=True):
        """ Feeds a single observation and updates the importance weights accordingly. """
        for k in range(self.NP):
            c, cnt, p, q = self.particles[k]
            if is_in:
                logp = self.singleton_log_likelihood(edge, c, p, q)
            else:
                logp = self.singleton_neg_log_likelihood(edge, c, p, q)
            self.weights[k] *= np.exp(logp)
        self.weights /= np.sum(self.weights)

    def observed_log_posterior(self):
        """ Computes log P(H|D) for observed data. """
        logp = self.log_prior()
        for edge in self.D.E:
            logp += self.singleton_log_likelihood(edge, self.c, self.p, self.q)
        for edge in self.D.NE:
            logp += self.singleton_neg_log_likelihood(edge, self.c, self.p, self.q)
        return logp

    def rejuvenate(self, niter=10):
        """ Perform iterations of MH-within-Gibbs and update particles. """

        # Perform `niter` steps of MHwG independently for each particle.
        for t in range(1, niter + 1):
            for k in range(self.NP):
                self.c, self.cnt = self.particles[k][0].copy(), self.particles[k][1].copy()
                self.p, self.q = self.particles[k][2], self.particles[k][3]
                self.nclusters = len(self.cnt)

                for i in range(self.D.N):
                    old_i = self.c[i]
                    proposal_dist = self.cnt.copy()
                    proposal_dist[self.c[i]-1] -= 1
                    proposal_dist = np.append(proposal_dist, self.alpha)
                    proposal_dist /= (self.D.N - 1 + self.alpha)
                    cand_i = np.where(np.random.multinomial(1, proposal_dist) == 1)[0][0] + 1
                    old_logpost = self.observed_log_posterior()
                    self.update_cluster(i, cand_i)
                    new_logpost = self.observed_log_posterior()
                    log_accept_ratio = min(0, new_logpost - old_logpost)
                    if np.log(np.random.uniform()) > log_accept_ratio:
                        self.update_cluster(i, old_i)

                old_p = self.p
                ll, uu = stats.norm.cdf(0, loc=old_p, scale=0.1), stats.norm.cdf(1, loc=old_p, scale=0.1)
                cand_p = stats.norm.ppf(stats.uniform.rvs(loc=ll, scale=uu-ll), loc=old_p, scale=0.1)
                old_logpost = self.observed_log_posterior()
                self.p = cand_p
                new_logpost = self.observed_log_posterior()
                cdf_corr_new = np.log(stats.norm.cdf(1, loc=cand_p, scale=0.1) - stats.norm.cdf(0, loc=cand_p, scale=0.1))
                cdf_corr_old = np.log(stats.norm.cdf(1, loc=old_p, scale=0.1) - stats.norm.cdf(0, loc=old_p, scale=0.1))
                log_accept_ratio = min(0, new_logpost - old_logpost + cdf_corr_new - cdf_corr_old)
                if np.log(np.random.uniform()) > log_accept_ratio:
                    self.p = old_p

                # Sample q.
                old_q = self.q
                ll, uu = stats.norm.cdf(0, loc=old_q, scale=0.1), stats.norm.cdf(1, loc=old_q, scale=0.1)
                cand_q = stats.norm.ppf(stats.uniform.rvs(loc=ll, scale=uu-ll), loc=old_q, scale=0.1)
                old_logpost = self.observed_log_posterior()
                self.q = cand_q
                new_logpost = self.observed_log_posterior()
                cdf_corr_new = np.log(stats.norm.cdf(1, loc=cand_q, scale=0.1) - stats.norm.cdf(0, loc=cand_q, scale=0.1))
                cdf_corr_old = np.log(stats.norm.cdf(1, loc=old_q, scale=0.1) - stats.norm.cdf(0, loc=old_q, scale=0.1))
                log_accept_ratio = min(0, new_logpost - old_logpost + cdf_corr_new - cdf_corr_old)
                if np.log(np.random.uniform()) > log_accept_ratio:
                    self.q = old_q

                self.particles[k] = (self.c.copy(), self.cnt.copy(), self.p, self.q)

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
            self.c, self.cnt, self.p, self.q = particle
            self.nclusters = len(self.cnt)
            return self.observed_log_posterior()

        def add_uk(edge, particle, is_in):
            c, cnt, p, q = particle
            if is_in:
                return self.singleton_log_likelihood(edge, c, p, q)
            return self.singleton_neg_log_likelihood(edge, c, p, q)

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
            logpeds = np.array(list(map(lambda pc: np.exp(self.singleton_log_likelihood(edge, pc[0], pc[2], pc[3])), self.particles)))
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
            logpeds = np.array(list(map(lambda pc: np.exp(self.singleton_neg_log_likelihood(edge, pc[0], pc[2], pc[3])), self.particles)))
            without_prob = np.sum(logpeds * self.weights)
            if phdtype == 'full':
                self.D.NE = self.D.NE[:-1]

            self.action_entropys.append(with_entropy * with_prob + without_entropy * without_prob)

        for j in range(len(self.D.UK)):
            print(f'For edge {self.D.UK[j]}, H(H|D, a) = {self.action_entropys[j]}.')

    def plot_partial_graph(self, action='save'):
        """ Plot graphs for edge unveiling. """
        uid = 200
        while os.path.isfile(f'{uid}_{self.D.name}_ugraph.png'):
            uid += 1

        r = lambda: np.random.randint(low=0, high=256)
        ng = Graph()
        for i in range(self.D.N):
            ng.add_node(i + 1)
        for u, v in self.D.E:
            ng.add_edge(u, v, entropy='')
        for w in range(len(self.D.UK)):
            ng.add_edge(self.D.UK[w][0], self.D.UK[w][1], entropy='{:.2f}'.format(self.action_entropys[w]))
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
            plt.savefig(f'{uid}_{self.D.name}_ugraph.png')
        elif action == 'show':
            plt.show()

    def plot_graphs(self, action='save'):
        """ Plot various statistics. """
        uid = 100
        while os.path.isfile(f'{uid}_{self.D.name}_logpost.png'):
            uid += 1

        # Log-posterior graph.
        plt.figure()
        plt.rc('font',size=14)
        plt.rc('axes',titlesize=14)
        plt.rc('axes',labelsize=14)
        plt.title(f'Log-Posterior')
        plt.xlabel('Iterations')
        plt.ylabel('P(H|D)')
        plt.plot(self.logpost_samples)
        if action == 'save':
            plt.savefig(f'{uid}_{self.D.name}_logpost.png')
        elif action == 'show':
            plt.show()

        # Displaying H.
        for bidx in range(1, 6):
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
            plt.title('Hierarchical Clusters')
            draw_networkx(ng, node_color=node_colors)
            if action == 'save':
                plt.savefig(f'{uid}_{self.D.name}_{bidx}hgraph.png')
            elif action == 'show':
                plt.show()

    def plot_christmas_graph(self, action='save'):
        """ Plot graph for human experiments. """
        ng = Graph()
        for i in range(self.D.N):
            ng.add_node(i + 1)
        for u, v in self.D.E:
            ng.add_edge(u, v, llabel='')
        chrval = 65
        for w in range(len(self.D.UK)):
            ng.add_edge(self.D.UK[w][0], self.D.UK[w][1], llabel=chr(chrval))
            chrval += 1
        colormap = []
        for ee in ng.edges:
            if ee in self.D.UK:
                colormap.append('red')
            else:
                colormap.append('black')
        plt.figure()
        plt.axis('off')
        pos = spring_layout(ng)
        draw_networkx(ng, pos, with_labels=False, node_color='black', edge_color=colormap)
        edge_labels = get_edge_attributes(ng, 'llabel')
        draw_networkx_edge_labels(ng, pos, edge_labels=edge_labels, font_color='red', font_weight='bold', font_size=14)
        if action == 'save':
            plt.savefig(f'christmas_{self.D.name}.png')
        elif action == 'show':
            plt.show()

def demo1(fname):
    """ Perform offline inference on a given (completely observed) graph and display results. """
    h = {'alpha': 1.5}
    D = Data(fname)
    hm = HierarchyModel(D, h)
    hm.offline_sampler(nsamples=1000, nburnin=0)
    hm.plot_graphs()

def demo2(fname, inference_type='offline'):
    """ Perform online inference on a partially observed graph, and make a decision on which edge to unveil next. """
    h = {'alpha': 1.5}
    D = Data(fname, fully_observed=False)
    hm = HierarchyModel(D, h)
    if inference_type == 'offline':
        hm.offline_sampler(inference_type='online')
        # print('Top particles after offline sampling:')
        # for top in hm.best_hsamples[-5:]:
        #     print(top[0])

        # Testing with MAP.
        hm.particles = []
        for top in hm.best_hsamples[round(0.9 * len(hm.best_hsamples)):]:
            hm.particles.append(top)
            print(top[0])
        hm.NP = len(hm.particles)
        hm.weights = np.array([1 / hm.NP for _ in range(hm.NP)])
        print('Unveiling edges...')
        hm.unveil()
        hm.plot_partial_graph()

        # Testing with full offline posterior.
        hm.particles = hm.best_hsamples
        hm.NP = len(hm.particles)
        hm.weights = np.array([1 / hm.NP for _ in range(hm.NP)])
        print('Unveiling edges...')
        hm.unveil()
        hm.plot_partial_graph()
    else:
        hm.online_sampler()
        print('Top particles after online sampling:')
        for top in np.argsort(hm.weights)[-5:]:
            print(hm.particles[top][0])
        print('Unveiling edges...')
        hm.unveil()
        hm.plot_partial_graph()


#demo1('data/demon.txt')
demo2('data/partial_demon.txt', 'online')
#demo2('data/partial_hourglass.txt')
#demo2('data/partial_solway1.txt')
#demo2('data/partial_solway2.txt')

def plot_model_results(action='save'):
    """ Plot model results. """
    plt.figure()
    plt.rc('font',size=14)
    plt.rc('axes',titlesize=14)
    plt.rc('axes',labelsize=14)
    plt.title(f'Model Results')
    plt.xlabel('Graph')
    plt.ylabel('Fraction of Trials where Edge A unveiled')
    ax = plt.gca()
    #plt.bar()
    #plt.xticks()
    low95, high95 = stats.binom.ppf(0.025,n=50, p=0.5) / 50, stats.binom.ppf(0.975,n=50, p=0.5) / 50
    plt.axhline(y=low95, alpha=0.4, color='red')
    plt.axhline(y=high95, alpha=0.4, color='red')
    ax.axhspan(low95, high95, facecolor='red', alpha=0.4)
    plt.axhline(y=0.5, linestyle='--', color='black')
    if action == 'save':
        plt.savefig('model_results.png')
    elif action == 'show':
        plt.show()

#plot_model_results('show')

