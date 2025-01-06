from metaflow import FlowSpec, step, Parameter, resources, conda_base, profile
from sklearn.mixture import GaussianMixture
from analyze_kmeans import top_words
@conda_base(python="3.10.16", libraries={"scikit-learn": "1.5.2"})
class GaussianFlow(FlowSpec):
    num_docs = Parameter('num-docs',help='Number of documents')
    @resources(memory=200)
    @step
    def start(self):
        import scale_data
        
        docs = scale_data.load_chat(self.num_docs)
        self.mtx, self.cols = scale_data.make_matrix(docs)
        self.gmm_params = [3,4,5]
        self.next(self.train_gmm, foreach='gmm_params')
    
    @resources(cpu=1, memory=200)
    @step
    def train_gmm(self):
        self.k = self.input
        with profile('gmm'):
            gmm = GaussianMixture(n_components=self.k, random_state=42,n_init=10)
            mtx_dense = self.mtx.toarray()
            gmm.fit(mtx_dense)
        self.clusters = gmm.predict(mtx_dense)
        self.next(self.analyze)
    
    @step
    def analyze(self):
        self.top = top_words(self.k, self.clusters, self.mtx, self.cols)
        self.next(self.join)
    
    @step
    def join(self,inputs):
        self.top = {inp.k:inp.top for inp in inputs}
        self.next(self.end)
    
    @step
    def end(self):
        pass
if __name__ == '__main__':
    GaussianFlow()
