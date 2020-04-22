class DAG(object):
    
    def __init__(self, dag_list):
        self.dag_list = dag_list
        self.n_nodes = len(dag_list)
        self.root_nodes = []
        for k in range(self.n_nodes):
            if len(dag_list[k]) == 0:
                self.root_nodes.append(k)
    
    def get_n_nodes(self):
        return self.n_nodes
    
    def get_parent_nodes(self, k):
        return self.dag_list[k]
    
    def get_root_nodes(self):
        return self.root_nodes
    