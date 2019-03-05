# Graph Kernel & Manifold Learning
Assignment for the course of Artificial Intelligence: Knowledge Representation and Planning, taught by Professor Andrea Torsello of the Ca' Foscari University of Venice.


## Further information
Check the [report](Report.pdf) for a complete analysis of the task and much more information on the actual implementation and the results.

## Description (by the Professor)
Read [this article](http://www.dsi.unive.it/~atorsell/AI/graph/Unfolding.pdf) presenting a way to improve the disciminative power of graph kernels.

Choose one [graph kernel](http://www.dsi.unive.it/~atorsell/AI/graph/kernels.pdf) among
* Shortest-path Kernel
* Graphlet Kernel
* Random Walk Kernel
* Weisfeiler-Lehman Kernel

Choose one manifold learning technique among
* Isomap
* Diffusion Maps
* Laplacian Eigenmaps
* Local Linear Embedding

Compare the performance of an SVM trained on the given kernel, with or without the manifold learning step, on the following datasets:
* [PPI](http://www.dsi.unive.it/~atorsell/AI/graph/PPI.mat)
* [Shock](http://www.dsi.unive.it/~atorsell/AI/graph/SHOCK.mat)

**Note**: the datasets are contained in Matlab files. The variable G contains a vector of cells, one per graph. The entry am of each cell is the adjacency matrix of the graph. The variable labels, contains the class-labels of each graph. 

