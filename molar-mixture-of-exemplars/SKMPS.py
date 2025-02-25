
from cuml import KMeans
import faiss
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

from pdb import set_trace as pb

seed = 0
budget = 40
input_features = '../output/local/save-representation_0/tensorboard/version_0/predictions/epoch=0-step=0-_predict_backbone_0.npy'
input_labels = '../output/local/save-representation_0/tensorboard/version_0/predictions/data_train_labels.npy'
output_name = 'labelled_prototypes/CIFAR10_X.csv'

use_features = np.load(input_features, allow_pickle=True)
labels = np.load(input_labels, allow_pickle=True)

kmeans = KMeans(random_state=seed, n_clusters=budget, n_init= 10)
kmeans_cuml = kmeans.fit(use_features)

cluster_centers = kmeans_cuml.cluster_centers_

cluster_centers = normalize(cluster_centers, axis=1)
use_features = normalize(use_features, axis=1)
index = faiss.IndexFlatIP(cluster_centers.shape[1])

# ngus=faiss.get_num_gpus()
# index = faiss.index_cpu_to_all_gpus(index,ngpu=ngus)    
index.add(np.ascontiguousarray(use_features))
D, I = index.search(cluster_centers, 1)
I = I.squeeze()

result_dict = {}
result_dict['indices'] = I
result_dict['classes'] = labels[I]

total_classes = np.unique(result_dict['classes']).shape[0]
max_classes = np.unique(labels).shape[0]

result_dict['total_classes'] = total_classes
result_dict['max_classes'] = max_classes
result_dict['seed'] = seed

df = pd.DataFrame({'indices': result_dict['indices'], 'label': result_dict['classes']})
df.to_csv(output_name)
