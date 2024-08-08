import torch.nn.modules.loss
from tqdm import tqdm
from torch.backends import cudnn

from .Module import *
from .Utils import *

class G3net:
    def __init__(self, adata, graph_dict, num_cluster,  device, config, roundseed=0):
        seed = config['seed'] + roundseed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True)

        self.device = device
        self.adata = adata
        self.graph_dict = graph_dict
        self.mode = config['mode']
        self.train_config = config['train']
        self.model_config = config['model']
        self.num_cluster = num_cluster

    def _start_(self):
        if self.mode == 'clustering':
            self.X = torch.FloatTensor(self.adata.obsm['X_pca'].copy()).to(self.device)
        elif self.mode == 'imputation':
            self.X = torch.FloatTensor(self.adata.X.copy()).to(self.device)
        else:
            raise Exception
        self.section_ids = np.array(self.adata.obs['batch_name'].unique())
        label = self.adata.obs['slice_id']
        label = F.one_hot(torch.tensor(label)).float()
        self.label = torch.FloatTensor(label).to(self.device)
        self.adj_norm = self.graph_dict["adj_norm"].to(self.device)
        self.adj_label = self.graph_dict["adj_label"].to(self.device)
        self.norm_value = self.graph_dict["norm_value"]

        self.input_dim = self.X.shape[-1]
        self.class_num = self.label.shape[-1]
        self.autoencoder = Autoencoder(self.input_dim, self.model_config).to(self.device)
        self.discriminator = Discriminator(self.class_num, self.label, self.model_config).to(self.device)
        self.g_optimizer = torch.optim.Adam(
            params=list(self.autoencoder.parameters()),
            lr=self.train_config['lr'],
            weight_decay=self.train_config['decay'],
        )
        self.d_optimizer = torch.optim.Adam(
            params=list(self.discriminator.parameters()),
            lr=self.train_config['lr'],
            weight_decay=self.train_config['decay'],
        )

    def _train_d_model_(self):
        self.autoencoder.eval()
        self.discriminator.train()
        self.d_optimizer.zero_grad()
        latent_emb = self.autoencoder.embeding(self.X, self.adj_norm)
        disc_loss = self.discriminator(latent_emb)
        disc_loss.backward()
        if self.train_config['gradient_clipping'] > 1:
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.train_config['gradient_clipping'])
        self.d_optimizer.step()

    def _train_g_model_(self):
        self.autoencoder.train()
        self.discriminator.eval()
        self.g_optimizer.zero_grad()
        tri_loss, rec_loss, latent_emb = self.autoencoder(self.X, self.adj_norm)
        disc_loss = - self.discriminator.evaluate(latent_emb)
        loss = self.train_config['w_recon'] * rec_loss + self.train_config['w_tri'] * tri_loss + self.train_config['w_disc'] * disc_loss
        loss.backward()
        if self.train_config['gradient_clipping'] > 1:
            torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), self.train_config['gradient_clipping'])
        self.g_optimizer.step()

    @torch.no_grad()
    def _test_model_(self):
        self.autoencoder.eval()
        self.discriminator.eval()
        tri_loss, rec_loss, latent_emb = self.autoencoder(self.X, self.adj_norm)
        disc_loss = self.discriminator(latent_emb)
        loss = self.train_config['w_recon'] * rec_loss + self.train_config['w_tri'] * tri_loss + self.train_config['w_disc'] * disc_loss
        return rec_loss, tri_loss, disc_loss, loss

    def _fit_(self, early_stop_epochs=100, convergence=0.001, verbose=0, method='mclust', random_seed=2024):
        g_step = self.train_config['g_step']
        d_step = self.train_config['d_step']
        t_step = self.train_config['t_step']
        start = self.train_config['start_step']
        plot_step = self.train_config['plot_step']

        best_loss = np.inf
        early_stop_count = 0
        pbar = tqdm(range(self.train_config['epochs']))
        for epoch in pbar:
            for _ in range(d_step):
                self._train_d_model_()
            for _ in range(g_step):
                self._train_g_model_()

            if epoch % t_step == 0 and epoch >= start:
                self.autoencoder.set_anchor_pair(self._get_anchor_(verbose=verbose, method=method, random_seed=random_seed))

            if epoch % plot_step == 0:
                rec_loss, tri_loss, disc_loss, total_loss = self._test_model_()
                current_loss = total_loss.cpu().detach().numpy()
                pbar.set_description(
                    "Epoch {0} total loss={1:.3f} recon loss={2:.3f} tri loss={3:.3f} disc loss={4:.3f}".format(
                        epoch, current_loss, rec_loss, tri_loss, disc_loss),
                    refresh=True)
                if best_loss - current_loss > convergence:
                    if best_loss > current_loss:
                        best_loss = current_loss
                    early_stop_count = 0
                else:
                    early_stop_count += 1
                if early_stop_count > early_stop_epochs:
                    print('Stop trainning because of loss convergence')
                    break

    def train(self, early_stop_epochs=100, convergence=0.001, verbose=0, method='mclust', random_seed=2024):
        self._start_()
        self._fit_(early_stop_epochs, convergence, verbose, method=method, random_seed=random_seed)

    def process(self):
        self.autoencoder.eval()
        enc_rep, recon = self.autoencoder.evaluate(self.X, self.adj_norm)
        return enc_rep, recon

    def clustering(self, adata, num_cluster, used_obsm, key_added_pred, random_seed=2024, method='mclust'):
        assert method in ['mclust', 'kmeans', 'louvain', 'louvain2']
        if method == 'mclust':
            adata = mclust_R(adata, num_cluster=num_cluster, used_obsm=used_obsm, key_added_pred=key_added_pred, random_seed=random_seed)
        elif method == 'kmeans':
            adata = Kmeans_cluster(adata, num_cluster=num_cluster, used_obsm=used_obsm, key_added_pred=key_added_pred, random_seed=random_seed)
        elif method == 'louvain':
            adata = louvain_py(adata, num_cluster=num_cluster, used_obsm=used_obsm, key_added_pred=key_added_pred, random_seed=random_seed)
        elif method == 'louvain2':
            sc.pp.neighbors(adata, use_rep=used_obsm, random_state=random_seed)
            sc.tl.louvain(adata, random_state=random_seed, key_added=method, resolution=num_cluster)
        else:
            raise Exception
        return adata

    def _get_anchor_(self, verbose=0, method='mclust', random_seed=2024):
        self.autoencoder.eval()
        latent_emb = self.autoencoder.embeding(self.X, self.adj_norm)

        self.adata.obsm['latent'] = latent_emb.data.cpu().numpy()
        key_pred = 'Tmp_domain'
        self.adata = self.clustering(self.adata, num_cluster=self.num_cluster, used_obsm='latent', key_added_pred=key_pred, method=method, random_seed=random_seed)

        gnn_dict = create_dictionary_gnn(self.adata, use_rep='latent', use_label=key_pred, batch_name='batch_name', k=self.train_config['knn_neigh'], verbose=verbose)
        anchor_ind = []
        positive_ind = []
        negative_ind = []
        for batch_pair in gnn_dict.keys():
            batchname_list = self.adata.obs['batch_name'][gnn_dict[batch_pair].keys()]

            cellname_by_batch_dict = dict()
            for batch_id in range(len(self.section_ids)):
                cellname_by_batch_dict[self.section_ids[batch_id]] = self.adata.obs_names[
                    self.adata.obs['batch_name'] == self.section_ids[batch_id]].values

            anchor_list = []
            positive_list = []
            negative_list = []
            for anchor in gnn_dict[batch_pair].keys():
                anchor_list.append(anchor)
                positive_spot = gnn_dict[batch_pair][anchor][0]
                positive_list.append(positive_spot)
                section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                negative_list.append(
                    cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])

            batch_as_dict = dict(zip(list(self.adata.obs_names), range(0, self.adata.shape[0])))
            anchor_ind = np.append(anchor_ind, list(map(lambda _: batch_as_dict[_], anchor_list)))
            positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))
            negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))
        anchor_pair = (anchor_ind, positive_ind, negative_ind)
        return anchor_pair
