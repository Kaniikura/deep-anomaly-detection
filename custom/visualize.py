import umap
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# silence NumbaPerformanceWarning
import warnings
from numba.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

def show_umap(embs, targets):
    targets = [float(targets[i]) for i in range(len(targets))]
    # UMAP
    embedding = umap.UMAP().fit_transform(embs)
    plt.scatter(embedding[:,0],embedding[:,1],c=targets,cmap=cm.tab10)
    plt.colorbar()
    plt.show()