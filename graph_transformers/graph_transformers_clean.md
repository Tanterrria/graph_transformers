# Graph Transformers: A Survey

## Abstract
Graph Transformers (GTs) have emerged as a powerful alternative to Message Passing Neural Networks (MPNNs) in graph learning, addressing their limitations in long-range dependencies and over-squashing. This survey provides a comprehensive overview of GTs, covering their foundations, architectures, and applications. We first introduce the background and motivation behind GTs, followed by a detailed exploration of their key components: graph learning, self-attention mechanisms, and positional encodings. We then present a systematic taxonomy of GTs, categorizing them into shallow, deep, scalable, and pre-trained variants. The survey also examines various applications of GTs across different domains, including node-level, edge-level, and graph-level tasks. Finally, we discuss open issues and future directions in GT research, providing insights for both practitioners and researchers.

## 1 Introduction
Graphs are ubiquitous data structures that represent relationships between entities in various domains, such as social networks, molecular structures, and knowledge graphs. Traditional graph learning methods, particularly Message Passing Neural Networks (MPNNs), have shown remarkable success in capturing local structural information. However, MPNNs face inherent limitations in modeling long-range dependencies and handling graph bottlenecks, leading to the phenomenon known as over-squashing.

Graph Transformers (GTs) have emerged as a promising alternative to address these limitations. By leveraging the self-attention mechanism, GTs can directly model interactions between any pair of nodes, regardless of their distance in the graph. This capability enables GTs to capture both local and global structural information effectively.

The development of GTs has been influenced by several key factors:
1. The success of Transformers in natural language processing and computer vision
2. The need to overcome the limitations of MPNNs
3. The increasing complexity of graph-structured data in real-world applications

This survey aims to provide a comprehensive overview of GTs, covering their theoretical foundations, architectural designs, and practical applications. We organize the survey as follows:

- Section 2 introduces the notations and preliminaries
- Section 3 discusses the key components of GTs
- Section 4 presents a taxonomy of GT architectures
- Section 5 explores various applications of GTs
- Section 6 discusses open issues and future directions

## Table of Contents

1. [Abstract](#abstract)
2. [Introduction](#1-introduction)
3. [Notations and Preliminaries](#2-notations-and-preliminaries)
   - [Graph Notations](#21-graph-notations)
   - [Graph Neural Networks](#22-graph-neural-networks)
   - [Transformer Architecture](#23-transformer-architecture)
4. [Design Perspectives](#design-perspectives)
   - [Graph Inductive Bias](#graph-inductive-bias)
     - [Node Positional Bias](#node-positional-bias)
     - [Edge Structural Bias](#edge-structural-bias)
     - [Message-passing Bias](#message-passing-bias)
     - [Attention Bias](#4-attention-bias)
   - [Graph Attention Mechanisms](#b-graph-attention-mechanisms)
     - [Global Attention Mechanisms](#1-global-attention-mechanisms)
     - [Local Attention Mechanisms](#2-local-attention-mechanisms)
5. [Taxonomy of Graph Transformers](#iv-taxonomy-of-graph-transformers)
   - [Shallow Graph Transformers](#a-shallow-graph-transformers)
   - [Deep Graph Transformers](#b-deep-graph-transformers)
   - [Scalable Graph Transformers](#c-scalable-graph-transformers)
   - [Pre-trained Graph Transformers](#d-pre-trained-graph-transformers)
   - [Design Guide for Effective Graph Transformers](#e-design-guide-for-effective-graph-transformers)
6. [Application Perspectives](#v-application-perspectives-of-graph-transformers)
   - [Node-level Tasks](#a-node-level-tasks)
     - [Protein Structure Prediction](#1-protein-structure-prediction)
     - [Entity Resolution](#2-entity-resolution)
     - [Anomaly Detection](#3-anomaly-detection)
   - [Edge-level Tasks](#b-edge-level-tasks)
     - [Drug-Drug Interaction Prediction](#1-drug-drug-interaction-prediction)
     - [Knowledge Graph Completion](#2-knowledge-graph-completion)
     - [Recommender Systems](#3-recommender-systems)
   - [Graph-level Tasks](#c-graph-level-tasks)
     - [Molecular Property Prediction](#1-molecular-property-prediction)
     - [Graph Clustering](#2-graph-clustering)
     - [Graph Synthesis](#3-graph-synthesis)
   - [Other Application Scenarios](#d-other-application-scenarios)
     - [Text Summarization](#1-text-summarization)
     - [Image Captioning](#2-image-captioning)
     - [Image Generation](#3-image-generation)
     - [Video Generation](#4-video-generation)
7. [Open Issues and Future Directions](#vi-open-issues-and-future-directions)
   - [Scalability and Efficiency](#a-scalability-and-efficiency)
   - [Generalization and Robustness](#b-generalization-and-robustness)
   - [Interpretability and Explainability](#c-interpretability-and-explainability)
   - [Learning on Dynamic Graphs](#d-learning-on-dynamic-graphs)
   - [Data Quality and Sparsity](#e-data-quality-and-sparsity)
8. [Conclusion](#vii-conclusion)

## Introduction

Graphs, as data structures with high expressiveness, are widely used to present complex data in diverse domains, such as social media, knowledge graphs, biology, chemistry, and transportation networks. They capture both structural and semantic information from data, facilitating various tasks, such as recommendation, question answering, anomaly detection, sentiment analysis, text generation, and information retrieval.

To effectively deal with graph-structured data, researchers have developed various graph learning models, such as graph neural networks (GNNs), learning meaningful representations of nodes, edges and graphs. Particularly, GNNs following the message-passing framework iteratively aggregate neighboring information and update node representations, leading to impressive performance on various graph-based tasks.

More recently, the graph transformer, as a newly arisen and potent graph learning method, has attracted great attention in both academic and industrial communities. Graph transformer research is inspired by the success of transformers in natural language processing (NLP) and computer vision (CV), coupled with the demonstrated value of GNNs.

### Main Contributions

1. Comprehensive review of the design perspectives of graph transformers, including graph inductive bias and graph attention mechanisms
2. Novel taxonomy of graph transformers based on depth, scalability, and pre-training strategy
3. Review of application perspectives in various graph learning tasks and other domains
4. Identification of crucial open issues and future directions

## Notations and Preliminaries

### Graph Basics

A graph is a data structure consisting of a set of nodes (or vertices) \( V \) and a set of edges (or links) \( E \) that connect pairs of nodes. Formally, a graph can be defined as \( G = (V,E) \), where:
- \( V = \{v_1,v_2,\ldots,v_N\} \) is node set with \( N \) nodes
- \( E = \{e_1,e_2,\ldots,e_M\} \) is edge set with \( M \) edges
- Edge \( e_k = (v_i , v_j) \) indicates the connection between node \( v_i \) and node \( v_j \)

#### Common Representations:
- Adjacency matrix \( \mathbf{A} \in \mathbb{R}^{N \times N} \)
- Edge list \( E \in \mathbb{R}^{M \times 2} \)
- Node features matrix \( \mathbf{X} \in \mathbb{R}^{N \times d_n} \)
- Edge features tensor \( \mathbf{F} \in \mathbb{R}^{M \times d_e} \)

### Graph Learning

Graph learning refers to the task of acquiring low-dimensional vector representations (embeddings) for nodes, edges, or the entire graph. These embeddings capture both structural and semantic information of the graph.

#### Two Main Approaches:

1. **Spectral Methods**
   - Based on graph signal processing and graph Fourier transform
   - Implement convolution operations in the spectral domain
   - Can capture global information but suffer from high computational complexity

2. **Spatial Methods**
   - Based on message-passing and neighborhood aggregation
   - Implement convolution operations in the spatial domain
   - Can capture local information but have limitations in modeling long-range dependencies

#### Message-Passing Framework

The core of spatial methods is the message-passing framework:

\[
\mathbf{h}_v^{(l+1)} = \phi \left( \mathbf{h}_v^{(l)}, \bigoplus_{u \in \mathcal{N}(v)} f(\mathbf{h}_u^{(l)}, \mathbf{h}_v^{(l)}, e_{uv}) \right)
\]

Where:
- \( \mathbf{h}_v^{(l)} \): Hidden state of node \( v \) at layer \( l \)
- \( \mathcal{N}(v) \): Neighborhood of node \( v \)
- \( \phi \): Update function
- \( \bigoplus \): Aggregation function
- \( f \): Message function
- \( e_{uv} \): Edge feature between nodes \( u \) and \( v \)

### Self-attention and Transformers

Self-attention is a mechanism that enables a model to learn to focus on pertinent sections of input or output sequences. It calculates a weighted sum of all elements in a sequence with weights determined by the similarity between each element and a query vector.

The self-attention mechanism is defined as:

\[
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{QK}^T}{\sqrt{d_k}}\right) \mathbf{V}
\]

Where:
- \( \mathbf{Q}, \mathbf{K}, \mathbf{V} \): Query, key, and value matrices
- \( d_k \): Dimension of query and key matrices

Transformers are neural network models that use self-attention as their main building block, consisting of:
- Encoder: Processes input sequence
- Decoder: Generates output sequence
- Attention mechanisms: Enable focus on relevant parts of sequences

## Design Perspectives

In this section, we discuss the primary architectures of graph transformers, aiming to explore their design perspectives in depth. Particularly, we will focus on two key components: graph inductive biases and graph attention mechanisms, to understand how these elements shape graph transformer models' capabilities.

### Graph Inductive Bias

Unlike Euclidean data, such as texts and images, graph data is non-Euclidean data, which has intricate structures and lacks a fixed order and dimensionality, posing difficulties in directly applying standard transformers on graph data. To address this issue, graph transformers incorporate graph inductive bias to encode the structural information of graphs and achieve effective generalization of transformers across new tasks and domains.

We classify graph inductive bias into four categories:
1. Node positional bias
2. Edge structural bias
3. Message-passing bias
4. Attention bias

#### Node Positional Bias

Node positional bias is a crucial inductive bias for graph transformers because it provides information about the relative or absolute positions of nodes in a graph. Formally, given a graph \( G = (V, E) \) with \( N \) nodes and \( M \) edges, each node \( v_i \in V \) has a feature vector \( x_i \in \mathbb{R}^{d_n} \). A graph transformer aims to learn a new feature vector \( h_i \in \mathbb{R}^{d_k} \) for each node by applying a series of self-attention layers.

##### Local Node Positional Encodings

Building on the success of relative positional encodings in NLP, graph transformers leverage a similar concept for local node positional encodings. This encoding technique aims to preserve the local connectivity and neighborhood information of nodes, which is critical for tasks like node classification, link prediction, and graph generation.

A proficient approach for integrating local node positional information involves the utilization of one-hot vectors representing the hop distance between nodes:

\[
\mathbf{p}_i = [I(d(i,j)=1), I(d(i,j)=2), \ldots, I(d(i,j)=\text{max})]
\]

where \( d(i,j) \) represents the shortest path distance between nodes \( v_i \) and \( v_j \), and \( I \) is an indicator function.

Another approach uses learnable embeddings that capture the relationship between nodes based on edge features:

\[
\mathbf{p}_i = [f(e_{ij1}), f(e_{ij2}), \ldots, f(e_{ijl})]
\]

where \( e_{ij} \) is the edge feature between nodes \( v_i \) and \( v_j \), and \( f \) is a learnable function.

##### Global Node Positional Encodings

Global node positional encodings are inspired by absolute positional encodings in NLP. These encodings aim to encapsulate the overall geometry and spectrum of graphs, revealing their intrinsic properties.

One method uses eigenvectors of the graph Laplacian matrix:

\[
\mathbf{p}_i = [u_{i1}, u_{i2}, \ldots, u_{ik}]
\]

where \( u_{ij} \) is the j-th component of the i-th eigenvector.

Another approach uses graph embedding techniques to map nodes to a lower-dimensional space:

\[
\mathbf{p}_i = [y_{i1}, y_{i2}, \ldots, y_{ik}]
\]

where \( y_{ij} \) is the j-th component of the i-th node embedding, obtained by minimizing:

\[
\min_Y \sum_{i,j=1}^N w_{ij}\|y_i - y_j\|^2
\]

#### Edge Structural Bias

Edge structural bias is crucial for extracting and understanding complex information within graph structure. It can represent various aspects including node distances, edge types, edge directions, and local sub-structures.

##### Local Edge Structural Encodings

Local edge structural encodings capture the local structure of a graph by encoding relative position or distance between nodes. GraphiT introduces these encodings using positive definite kernels:

\[
k(u, v) = \exp(-\alpha d(u, v))
\]

where \( d(u,v) \) is the shortest path distance between nodes \( u \) and \( v \).

The Edge-augmented Graph Transformer (EGT) introduces residual edge channels:

\[
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{R}_e) = \text{softmax}\left(\frac{\mathbf{QK}^T}{\sqrt{d_k}} + \mathbf{R}_e\right) \mathbf{V}
\]

where \( \mathbf{R}_e \) is the residual edge channel matrix.

##### Global Edge Structural Encodings

Global edge structural encodings aim to capture the overall structure of a graph. Unlike NLP and CV domains, the exact position of a node in graphs is not well-defined because there is no natural order or coordinate system. Several approaches have been suggested to tackle this issue.

GPT-GNN is an early work that utilizes graph pooling and unpooling operations to encode the hierarchical structure of a graph. It reduces graph size by grouping similar nodes and then restoring the original size by assigning cluster features to individual nodes. This approach results in a multiscale representation of graphs and has demonstrated enhanced performance on diverse tasks.

Graphormer uses spectral graph theory to encode global structure. It uses eigenvectors of normalized Laplacian matrix as global positional encodings for nodes. This method can capture global spectral features (e.g., connectivity, centrality and community structure).

Park et al. extended Graphormer by using singular value decomposition (SVD) to encode global structure. They utilized the left singular matrix of the adjacency matrix as global positional encodings for nodes. This approach can handle both symmetric and asymmetric matrices.

Global edge structural encodings excel at capturing coarse-grained structural information at the graph level, benefiting tasks that require global understanding. However, they may struggle with capturing fine-grained node-level information and can lose data or introduce noise during encoding. In addition, their effectiveness may depend on the choice of encoding technique and matrix representations.

#### Message-passing Bias

Message-passing bias is a crucial inductive bias for graph transformers to facilitate learning from the local structure of graphs. This bias enables graph transformers to exchange information between nodes and edges, thereby capturing dependencies and interactions among graph elements. Moreover, message-passing bias helps graph transformers in overcoming certain limitations of standard transformer architecture, such as the quadratic complexity of self-attention, the absence of positional information and challenges associated with handling sparse graphs.

Formally, message-passing bias can be expressed as follows:

\[
h_v^{(t+1)} = f(h_v^{(t)}, \{h_u^{(t)} : u \in N(v)\}, \{e_{uv} : u \in N(v)\})
\]

where:
- \( h_v^{(t)} \) is the hidden state of node v at time step t
- \( N(v) \) is the neighborhood of node v
- \( e_{uv} \) is the edge feature between nodes u and v
- \( f \) is a message-passing function

Here, \( h^{(0)}_v = x_v \), where \( x_v \) is the input feature of node \( v \), and SelfAttention is a function that performs self-attention over all nodes. The limitation of preprocessing is that it applies message-passing only once before self-attention layers. This approach may not fully capture intricate interactions between nodes and edges on various scales. Additionally, the preprocessing step may introduce redundancy and inconsistency between the message-passing module and the self-attention layer as they both serve similar functions of aggregating information from neighboring elements.

**Interleaving.** Interleaving refers to the technique employed in graph transformer architecture that involves alternating message-passing operations and self-attention layers. The objective of this technique is to achieve a balance between local and global information processing, thereby enabling multi-hop reasoning over graphs. By integrating message-passing modules into core components of graph transformers, interleaving enhances their expressive power and flexibility.

The interleaving approach can be mathematically defined as:

\[
h_v^{(t+1)} = \theta + \text{SelfAttention}(h_v^{(t)}, \{h_u^{(t)} : u \in V\}),
\]

where:

\[
\theta = f(h_v^{(t)}, \{h_u^{(t)} : u \in \mathcal{N}(v)\}, \{e_{uv} : u \in \mathcal{N}(v)\}).
\]

One drawback of the interleaving technique is its impact on the complexity and computational requirements of graph transformers. This is due to the need for additional parameters and operations compared to pre-processing or post-processing. Furthermore, interleaving can potentially lead to interference and conflict between the message-passing module and the self-attention layer as they each update node representations in distinct manners.

**Post-processing.** Post-processing refers to the technique of applying message-passing operations to node representations obtained from self-attention layers. The purpose of this approach is to refine and adjust node representations based on underlying graph structure, thereby enhancing their interpretability and robustness. By doing so, this method aims to improve the quality and utility of node representations for downstream tasks and applications. It can be mathematically defined as follows:

\[
h_v^{(t+1)} = \text{SelfAttention}(h_v^{(t)}, \{h_u^{(t)} : u \in V\}),
\]

\[
h_v^{(T+1)} = f(h_v^{(T)}, \{h_u^{(T)} : u \in \mathcal{N}(v)\}, \{e_{uv} : u \in \mathcal{N}(v)\}).
\]

Here, \( T \) is the final layer of the graph transformer.

The drawback of post-processing is its limited application of message-passing after self-attention layers, potentially failing to capture intricate semantics and dynamics of graph data. Additionally, post-processing runs the risk of introducing noise and distortion to node representations as it has the potential to overwrite or override information acquired by self-attention layers.

### 4) Attention Bias

Attention bias enables graph transformers to effectively incorporate graph structure information into the attention mechanism without message-passing or positional encodings. Attention bias modifies attention scores between nodes based on their relative positions or distances in the graph. It can be categorized as either local or global depending on whether it focuses on the local neighborhood or global topology of the graph.

**Local Attention Bias.** Local attention bias limits the attention to a local neighborhood surrounding each node. This concept is analogous to the message-passing mechanism observed in GNNs. It can be mathematically defined as follows:

\[
\alpha_{ij} = \frac{\exp(g(x_i, x_j) \cdot b_{ij})}{\sum_{k \in \mathcal{N}(v_i)} \exp(g(x_i, x_k) \cdot b_{ik})},
\]

where \( \alpha_{ij} \) is the attention score between node \( v_i \) and node \( v_j \), \( x_i \) and \( x_j \) are their node features, \( g \) is a function that computes the similarity between two nodes (such as dot-product and linear transformation), and \( b_{ij} \) is a local attention bias term that modifies attention score based on the distance between node \( v_i \) and node \( v_j \). The local attention bias term can be either a binary mask that only allows attention within a certain hop distance or a decay function that decreases attention score with increasing distance.

**Global Attention Bias.** Global attention bias integrates global topology information into the attention mechanism independent of message-passing and positional encodings. It can be mathematically defined as follows:

\[
\alpha_{ij} = \frac{\exp(g(x_i, x_j) + c(A, D, L, P)_{ij})}{\sum_{k=1}^N \exp(g(x_i, x_k) + c(A, D, L, P)_{ik})},
\]

where \( c \) is the function that computes the global attention bias term, modifying attention score based on some graph-specific matrices or vectors, such as adjacency matrix \( A \), degree matrix \( D \), Laplacian matrix \( L \), and PageRank vector \( P \). The global attention bias term can be additive or multiplicative to the similarity function. Generally, global attention bias can enhance the global structure awareness and expressive power of graph transformers.

### B. Graph Attention Mechanisms

Graph attention mechanisms play an important role in the construction of graph transformers. By dynamically assigning varying weights to nodes and edges in the graph, these mechanisms enable transformers to prioritize and emphasize the most relevant and important elements for a given task. Specifically, a graph attention mechanism is a function that maps each node \( v_i \in V \) to a vector \( h_i \in \mathbb{R}^{d_k} \):

\[
h_i = f_n(x_i, \{x_j\}_{v_j \in \mathcal{N}(v_i)}, \{A_{ij}\}_{v_j \in \mathcal{N}(v_i)}),
\]

where \( f_n \) is a nonlinear transformation, \( x_i \) is the input feature vector of node \( v_i \), and \( x_j \) is the input feature vector of node \( v_j \). The function \( f_n \) can be decomposed into two parts: an attention function and an aggregation function.

The attention function computes a scalar weight for each neighbor of node \( v_i \), denoted by \( \alpha_{ij} \), which reflects the importance or relevance of node \( v_j \) for node \( v_i \):

\[
\alpha_{ij} = \text{softmax}_i(\text{LeakyReLU}(W_a[x_i \| x_j])),
\]

where \( W_a \in \mathbb{R}^{1 \times 2d_n} \) is a learnable weight matrix, \( \| \) is the concatenation operation, and \( \text{softmax}_i \) normalizes weights over all neighbors of node \( v_i \). The aggregation function combines the weighted features of the neighbors to obtain the output representation of node \( v_i \):

\[
h_i = W_h\left(x_i + \sum_{v_j \in \mathcal{N}(v_i)} \alpha_{ij}x_j\right),
\]

where \( W_h \) is another learnable weight matrix. Graph attention mechanisms have the potential to extend their application beyond nodes to edges. This can be achieved by substituting nodes with edges and utilizing edge features instead of node features. Furthermore, stacking of graph attention mechanisms in multiple layers allows for the utilization of output representations from one layer as input features for the subsequent layer.

#### 1) Global Attention Mechanisms

Global attention mechanisms can determine how each node calculates its attention weights across all other nodes in the graph. Global attention mechanisms can be broadly categorized into two types: quadratic attention mechanisms and linear attention mechanisms.

**Quadratic Attention Mechanisms.** Quadratic attention mechanisms are derived from the conventional self-attention formula. This formula calculates attention weights by applying a softmax function to scale the dot product between the query and key vectors of each node:

\[
\alpha_{ij} = \frac{\exp\left(\frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_k}}\right)}{\sum_{n=1}^{N} \exp\left(\frac{\mathbf{q}_i^\top \mathbf{k}_n}{\sqrt{d_k}}\right)}.
\]

The computational complexity of this method is \( O(N^2) \).

One of the pioneering studies that introduced quadratic attention mechanisms for graph transformers is the work by Velickovic et al. The authors proposed the use of multiple attention heads to capture different types of relations between nodes. Choromanski et al. introduced the Graph Kernel Attention Transformer (GKAT), an approach that integrates graph kernels, structural priors, and efficient transformer architectures. GKAT leverages graph kernels as positional encodings to capture the structural characteristics of the graph while employing low-rank decomposition techniques to minimize the memory requirements of quadratic attention:

\[
\alpha_{ij} = \frac{\exp\left(\frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_k}} + \mathbf{p}_i^\top \mathbf{p}_j\right)}{\sum_{n=1}^{N} \exp\left(\frac{\mathbf{q}_i^\top \mathbf{k}_n}{\sqrt{d_k}} + \mathbf{p}_i^\top \mathbf{p}_n\right)},
\]

\[
\mathbf{z}_i = \sum_{j=1}^{N} \alpha_{ij}(\mathbf{UV}^\top)_{ij}
\]

where \(\mathbf{z}_i\) is the output vector of node \(v_i\), \(\mathbf{p}_i\) is the positional encoding vector of node \(v_i\), which is computed by applying a graph kernel function to node features. \(\mathbf{U}\) and \(\mathbf{V}\) are low-rank matrices that approximate the value matrix in the GKAT model. GKAT also introduced a novel kernel-based masking scheme to control the sparsity of the attention matrix. 

Yun et al. introduced Graph Transformer Networks (GTN), an approach that uses multiple layers of quadratic attention to acquire hierarchical representations of graphs. GTN further enhances attention computation by incorporating edge features through the use of bilinear transformations instead of concatenation. This innovative technique demonstrates the potential for improved graph representation learning in the field of graph analysis.

Quadratic attention mechanisms possess a significant advantage in capturing extensive dependencies and global information within graphs, thereby offering potential benefits for various graph learning tasks. Nonetheless, these mechanisms have some limitations, including their computationally expensive nature, high memory consumption, and susceptibility to noise and outliers. Additionally, quadratic attention mechanisms may not effectively preserve and leverage local information within graphs which holds significance in certain domains and tasks.

**Linear Attention Mechanisms.** Linear attention mechanisms employ different approximations and hashing techniques to decrease the computational complexity of self-attention from \(O(N^2)\) to \(O(N)\). These mechanisms can be categorized into two subtypes: kernel-based linear attention mechanisms and locality-sensitive linear attention mechanisms.

Kernel-based linear attention mechanisms leverage the concept of employing kernel functions to map query and key vectors into a feature space that allows efficient computation of their inner products. In the work by Katharopoulos et al., Linear Transformer Networks (LTN) were introduced. LTN utilizes random Fourier features as kernel functions to approximate self-attention:

\[
\alpha_{ij} = \frac{\exp\left(\frac{\phi(\mathbf{q}_i)^\top \phi(\mathbf{k}_j)}{\sqrt{d_k}}\right)}{\sum_{n=1}^{N} \exp\left(\frac{\phi(\mathbf{q}_i)^\top \phi(\mathbf{k}_n)}{\sqrt{d_k}}\right)},
\]

\[
\phi(\mathbf{x}) = \sqrt{\frac{2}{m}}[\cos(\omega_1^\top \mathbf{x} + b_1), ..., \cos(\omega_m^\top \mathbf{x} + b_m)]^\top.
\]

Here, \(\phi\) is a random Fourier feature function that maps input vector into a \(m\)-dimensional feature space. \(\omega_i\) and \(b_i\) are randomly sampled from a Gaussian distribution and a uniform distribution, respectively.

Locality-sensitive linear attention mechanisms are based on the concept of employing hashing functions to partition the query and key vectors into discrete buckets that facilitate local computation of their inner products. Kitaev et al. introduced the Reformer model which uses locality-sensitive hashing (LSH) as a hashing function to approximate self-attention:

\[
\alpha_{ij} = \frac{\exp\left(\frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_k}}\right)}{\sum_{n \in B_i} \exp\left(\frac{\mathbf{q}_i^\top \mathbf{k}_n}{\sqrt{d_k}}\right)},
\]

\[
B_i = \{j | h(\mathbf{q}_i) = h(\mathbf{k}_j)\},
\]

where \( h(\cdot) \) is a hashing function that maps the input vector into a bucket index and \( B_i \) is a set of nodes that share the same bucket index with node \( v_i \). RGN also incorporated edge features into the attention computation using a shared query-key projection matrix.

Linear attention mechanisms offer a significant advantage in terms of reducing computational cost and memory usage in self-attention, thereby enabling graph transformers to efficiently process large-scale graphs. However, they do come with certain limitations, including approximation error, hashing collision and loss of global information. Additionally, linear attention mechanisms may not be capable of capturing intricate or nonlinear relationships between nodes.

#### 2) Local Attention Mechanisms

Local attention mechanisms determine how each node computes its attention weights over a subset of nodes in the graph. Local attention mechanisms can be broadly categorized into two types: message-passing attention mechanisms and spectral attention mechanisms.

**Message-passing Attention Mechanisms.** Message-passing attention mechanisms are built on the foundational framework of message-passing neural networks (MPNNs), which iteratively aggregate messages from neighboring nodes to compute node representations. To enhance MPNNs, message-passing attention mechanisms employ self-attention to compute messages or aggregation weights. The computational complexity of this method is \( O(E) \), where \( E \) represents the number of edges in the graph.

One of the pioneering works in introducing message-passing attention mechanisms for graph transformers is GraphSAGE. GraphSAGE proposes the use of mean pooling as an aggregation function and self-attention as a combination function. Additionally, GraphSAGE incorporates node sampling techniques to efficiently handle large-scale graphs. The function can be mathematically defined as:

\[
\mathbf{h}_i^{(l+1)} = \text{ReLU}(\mathbf{W}^{(l)}[\mathbf{h}_i^{(l)} \| \text{MEAN}(\{\mathbf{h}_j^{(l)}, \forall v_j \in \mathcal{N}(v_i)\})])
\]

\[
\mathbf{z}_i = \sum_{l=0}^{L} \alpha_l \mathbf{h}_i^{(l)}.
\]

Here, \(\mathbf{h}_i^{(l)}\) is hidden vector of node \(v_i\) at layer \(l\), \(\mathbf{W}^{(l)}\) is a linear transformation matrix, \(\|\) is the concatenation operator, MEAN indicates a mean pooling function. \(\alpha_l\) means an attention weight for the hidden vector at layer \(l\). More recently, Javaloy et al. proposed L-CAT which uses self-attention as both an aggregation function and a combination function. L-CAT also incorporates edge features into the attention computation by using bilinear transformations. Message-passing attention mechanisms, while adept at preserving and leveraging local graph information, are constrained by scalability issues, such as limited expressiveness and graph connectivity dependence. Their capacity to capture long-range dependencies and global graph information is also questionable.

**Spectral Attention Mechanisms.** Spectral attention mechanisms are founded on the concept of transforming node features into a spectral domain, where the graph structure is encoded by eigenvalues and eigenvectors of the graph Laplacian matrix. Spectral attention mechanisms employ self-attention to calculate spectral coefficients and spectral filters. The computational complexity of this approach is \(O(N)\).

Wang et al. proposed graph isomorphism networks (GIN) as a spectral attention mechanism for graph transformers, employing an expressive aggregation function of sum pooling and self-attention. GIN also integrates a unique graph readout scheme via set functions to encapsulate global graph characteristics. The equations of GIN are as follows:

\[
\mathbf{h}_i^{(l+1)} = \text{MLP}^{(l)}((1 + \epsilon^{(l)})\mathbf{h}_i^{(l)} + \sum_{j \in \mathcal{N}(v_i)} \mathbf{h}_j^{(l)}),
\]

\[
\mathbf{z}_i = \sum_{l=0}^{L} \alpha_l \mathbf{h}_i^{(l)},
\]

\[
\mathbf{z}_G = \text{READOUT}(\{\mathbf{z}_i, \forall v_i \in V\}).
\]

Here, \(\text{MLP}^{(l)}\) is a multi-layer perceptron, \(\epsilon^{(l)}\) is a learnable parameter, READOUT indicates a set function that aggregates node output vectors into a graph output vector. \(\mathbf{z}_G\) is the output vector of graph \(G\). In addition, Nguyen et al. introduced UGformer, a self-attention-based method for the spectral coefficient computation of each node using eigenvalues and eigenvectors of the Laplacian graph matrix. UGformer further integrates edge features into spectral computation via bilinear transformations.

Spectral attention mechanisms possess the ability to incorporate the structural information of a graph into the spectral domain, thereby offering potential benefits for certain tasks or domains. However, they are also accompanied by certain limitations, including high computational cost, memory consumption and sensitivity to graph size and density.

## IV. TAXONOMY OF GRAPH TRANSFORMERS

The past few years have witnessed a surge of interest in graph transformers. This section dives into four key categories dominating the current literature: shallow, deep, scalable, and pre-trained graph transformers. By analyzing representative models within each category, we aim to establish valuable guidelines for designing effective graph transformers.

### A. Shallow Graph Transformers

Shallow graph transformers represent a class of GNNs that leverage the power of self-attention to acquire node representations from data structured in graphs. Inspired by transformer models, which effectively capture long-range dependencies in sequential data through self-attention, shallow graph transformers extend this concept to graph data by computing self-attention weights based on both node features and graph topology. The primary objective of shallow graph transformers is to achieve exceptional performance while minimizing computational complexity and memory usage.

Shallow graph transformers can be seen as a generalization of graph attention networks (GAT). GAT use a multi-head attention mechanism to calculate node embeddings. However, GAT has some limitations, such as the inability to model edge features and lack of diversity among attention heads. Several GAT extensions have been proposed in the literature to address these issues. For example, GTN by Yun et al. introduces edge-wise self-attention to incorporate edge information into node embeddings. Ahmad et al. proposed the graph attention transformer encoder (GATE), which applies a masked self-attention mechanism to learn different attention patterns for different nodes. GATE also uses a position-wise feed-forward network and dropout to enhance model capacity and generalization. The summary of shallow graph transformer methods is given in Table II.

Shallow graph transformers are efficient and adaptable capable of handling various graph learning tasks and different types of graphs, but their lack of depth and recurrence may limit their ability to capture complex dependencies. Their performance can also be influenced by the choice of mask matrix and the number of attention heads, indicating a need for further research on their optimal design and regularization.

### B. Deep Graph Transformers

Deep graph transformers consist of multiple self-attention layers stacked on top of each other, with optional skip connections, residual connections or dense connections between layers. They are designed to achieve higher performance with increased model depth and complexity. Deep graph transformers extend shallow graph transformers by applying self-attention layers to node features and graph topology hierarchically.

However, deep graph transformers also face several challenges that need to be addressed. One challenge is the difficulty of training deeper models, which can be mitigated by employing techniques, such as PairNorm introduced in DeeperGCN. Another challenge is the over-smoothing problem, which can be addressed by using a gated residual connection and a generalized convolution operator as proposed in DeeperGCN. Additionally, the disappearance of global attention capacity and the lack of diversity among attention heads are challenges that can be tackled by approaches like DeepGraph. DeepGraph incorporates substructure tokens and local attention to improve the focus and diversity of global attention.

Deep graph transformers, while complex, can achieve top-tier results on various graph learning tasks and adapt to different types of graphs and domains. However, their high computational cost, difficulty in optimization, and sensitivity to hyperparameters pose challenges that need to be addressed for practical applications. Table III summarizes the methods of deep graph transformers.

### C. Scalable Graph Transformers

Scalable graph transformers are a category of graph transformers that tackle the challenges of scalability and efficiency when applying self-attention to large-scale graphs. These transformers are specifically designed to reduce computational cost and memory usage while maintaining or improving performance. To achieve this, various techniques are employed to reduce the complexity of self- attention, such as sparse attention, local attention, and low- rank approximation. Scalable graph transformers can be regarded as an enhancement of deep graph transformers addressing challenges, such as over-smoothing and limited capacity of global attention. 

Several scalable graph transformer models have been pro- posed to enhance the scalability and efficiency of graph transformers.Forinstance,Rampa ́sˇeketal.[39]introduced GPS, use low-rank matrix approximations to reduce com- putational complexity, and achieve state-of-the-art results on diverse benchmarks. GPS decouples local real-edge aggre- gation from a fully-connected transformer and incorporates different positional and structural encodings to capture graph topology. It also offers a modular framework that supports multiple encoding types and mechanisms for local and global attention. Cong et al. [116] developed DyFormer, a dynamic graph transformer that utilizes substructure tokens and local attention to enhance the focus and diversity of global attention. DyFormer employs a temporal union graph structure and a subgraph-based node sampling strategy for efficient and scalable training.

Scalable graph transformers are an innovative and efficient category of graph transformers that excel in handling large-scale graphs while minimizing computational cost and memory usage. However, scalable graph transformers face certain limitations, including the trade-off between scalability and expressiveness, the challenge of selecting optimal hyperparameters and encodings, and the absence of theoretical analysis regarding their convergence and stability. Consequently, further investigation is required to explore optimal designs and evaluations of scalable graph transformers for various applications. For a comprehensive overview of scalable graph transformer methods, please refer to Table IV.

### D. Pre-trained Graph Transformers

Pre-trained graph transformers utilize large-scale unlabeled graphs to acquire transferable node embeddings. These embeddings can be fine-tuned for downstream tasks with scarce labeled data that address the challenges of data scarcity and domain adaptation in graph learning tasks. These transformers are similar to pre-trained large language models (LLMs) and are trained on graph datasets using self-supervised learning objectives, such as masked node prediction, edge reconstruction, and graph contrastive learning. These objectives aim to encapsulate the inherent properties of graph data independently of external labels or supervision. The pre-trained model can be fine-tuned on a specific downstream task with a smaller or domain-specific graph dataset by incorporating a task-specific layer or loss function and optimizing it on labeled data. This allows the pre-trained model to transfer the knowledge acquired from the large-scale graph dataset to the subsequent task, giving better performance compared to the training from scratch.

Pre-trained graph transformers face some challenges, such as the selection of appropriate pre-training tasks, domain knowledge incorporation, heterogeneous information integration, and pre-training quality evaluation. To address these issues, KPGT and KGTransformer have been proposed. KPGT leverages additional domain knowledge for pre-training, while KGTransformer serves as a uniform Knowledge Representation and Fusion (KRF) module in diverse tasks. Despite their power and flexibility, pre-trained graph transformers encounter issues related to graph data heterogeneity and sparsity, domain adaptation, model generalization and performance interpretation. A summary of pre-trained graph transformer methods is provided in Table V.

### E. Design Guide for Effective Graph Transformers

Developing effective graph transformers requires meticulous attention to detail and careful consideration. This guide provides general principles and tips for designing graph transformers for various scenarios and tasks.

* **Choose the appropriate type of graph transformers based on the nature and complexity of your graph data and tasks.** For simple and small graph data, a shallow graph transformer with a few layers may suffice. For complex and large graph data, a deep graph transformer with many layers can learn more expressive representations. For dynamic or streaming graph data, a scalable graph transformer is more efficient. Pre-trained graph transformers are more suitable for sparse or noisy graph data.

* **Design suitable structural and positional encodings for your graph data.** These encodings capture the structure of graphs and are added to input node or edge features before feeding them to transformer layers. The choice of encodings depends on the characteristics of the graph data, such as directionality, weight, and homogeneity. The careful design of these encodings ensures their informativeness.

* **Optimize the self-attention mechanism for your graph data.** Self-attention mechanisms compute attention scores among all pairs of nodes or edges in the graph, capturing long-range dependencies and interactions. However, it introduces challenges like computational complexity, memory consumption, overfitting, over-smoothing, and over-squashing. Techniques like sampling, sparsification, partitioning, hashing, masking, regularization, and normalization can be employed to address these challenges and improve the quality and efficiency of the self-attention mechanism.

* **Utilize pre-training techniques to enhance the performance of graph transformers.** Pre-training techniques leverage pre-trained models or data from other domains or tasks transferring knowledge or parameters to a specific graph learning task. Methods like fine-tuning, distillation, and adaptation can be used to adapt pre-trained models or data. Utilizing pre-training techniques is particularly beneficial when a large amount of pre-training data or resources are available.

## V. APPLICATION PERSPECTIVES OF GRAPH TRANSFORMERS

Graph transformers are finding applications in various domains that involve interconnected data. This section delves into their applications for graph-related tasks, categorized by the level of analysis: node-level, edge-level and graph-level. Beyond these core tasks, graph transformers are also making strides in applications that handle text, images, and videos, where data can be effectively represented as graphs for analysis.

### A. Node-level Tasks

Node-level tasks involve the acquisition of node representations or the prediction of node attributes using the graph structure and node features.

#### 1) Protein Structure Prediction

In the field of bioinformatics, graph transformers have demonstrated substantial potential in Protein Structure Prediction (PSP). Gu et al. introduced HEAL, which employs hierarchical graph transformers on super-nodes that imitate functional motifs to interact with nodes in the protein graph, effectively capturing structural semantics. Pepe et al. used Geometric Algebra (GA) modelling to introduce a new metric based on the relative orientations of amino acid residues, serving as an additional input feature to a graph transformer, assisting in the prediction of the 3D coordinates of a protein. Chen et al. proposed gated-graph transformers integrating node and edge gates within a graph transformer framework to regulate information flow during graph message passing, proving beneficial in predicting the quality of 3D protein complex structures. Despite the encouraging outcomes, various challenges persist, such as the complexity of protein structures, scarcity of high-quality training data, and substantial computational resource requirements. Further investigation is necessary to address these challenges and enhance the precision of these models.

#### 2) Entity Resolution

Entity Resolution (ER) is a crucial task in data management that aims to identify and link disparate representations of real-world entities from diverse sources. Recent research has highlighted the efficacy of graph transformers in ER. For example, Yao et al. proposed Hierarchical Graph Attention Networks (HierGAT) integrating the self-attention mechanism and graph attention network mechanism to capture and leverage the relationships between different ER decisions, leading to substantial enhancements over conventional approaches. Ying et al. proposed TranER, using a transformer-based architecture for entity resolution via self-supervised pre-training. TranER learns contextual entity representations and relation-aware matching functions through contrastive learning, demonstrating superior transfer capabilities across domains. However, challenges like the handling of incomplete or noisy data and the need for interpretable ER decisions still exist, requiring future research.

extended the standard transformer architecture and introduced several straightforward yet powerful structural encoding techniques to enhance the modeling of graph-structured data. Despite facing challenges related to data complexity and structural information encoding, these techniques have exhibited promising outcomes in terms of enhanced performance, scalability, and accuracy. Furthermore, Dou et al. proposed the Hybrid Matching Knowledge for Entity Matching (GTA) method improves the transformer for representing relational data by integrating additional hybrid matching knowledge acquired through graph contrastive learning on a specially designed hybrid matching graph. This approach has also demonstrated promising results by effectively boosting the transformer for representing relational data and surpassing existing entity matching frameworks.

#### 3) Anomaly Detection

Graph transformers are valuable tools for anomaly detection, especially in dynamic graphs and time series data. They tackle key challenges like encoding information for unattributed nodes and extracting discriminative knowledge from spatial-temporal dynamic graphs. Liu et al proposed TADDY, a transformer-based Anomaly Detection framework, enhancing node encoding to represent each node's structural and temporal roles in evolving graph streams. Similarly, Xu et al. proposed the Anomaly Transformer which uses an Anomaly-Attention mechanism to measure association discrepancy and employs a minimax strategy to enhance normal-abnormal differentiation. Chen et al. proposed the GTA framework for multivariate time series anomaly detection incorporates graph structure learning, graph convolution, and temporal dependency modeling with a transformer-based architecture. Tuli et al. developed TranAD, a deep transformer network for anomaly detection in multivariate time series data, showing efficient anomaly detection and diagnosis in modern industrial applications. Despite their effectiveness, further research is needed to enhance their performance and applicability across different domains.

### B. Edge-level Tasks

Edge-level tasks aim to learn edge representations or predict edge attributes based on graph structure and node features.

#### 1) Drug-Drug Interaction Prediction

Graph transformers have been increasingly employed in the prediction of Drug-Drug Interactions (DDIs) owing to their capability to adeptly model the intricate relationships between drugs and targets. Wang et al. proposed a method which uses a line graph with drug-protein pairs as vertices and a graph transformer network (DTI-GTN) for the purpose of forecasting drug-target interactions. Disidi et al. proposed a novel approach named DTIOG for the prediction of DTIs, leveraging a Knowledge Graph Embedding (KGE) strategy and integrating contextual information derived from protein sequences. Despite the encouraging outcomes, these approaches encounter challenges such as overlooking certain facets of the intermolecular information and identifying potential interactions for newly discovered drugs. Nevertheless, findings from multiple studies indicate that graph transformers can proficiently anticipate DDIs and surpass the performance of existing algorithms.

#### 2) Knowledge Graph Completion

In the domain of Knowledge Graph (KG) completion, the utilization of graph transformers has been extensively investigated. Chen et al. proposed a novel inductive KG representation model, known as iHT, for KG completion through large-scale pre-training. This model comprises an entity encoder and a neighbor-aware relational scoring function, both parameterized by transformers. The application of this approach has led to remarkable advancements in performance, with a relative enhancement of more than 25% in mean reciprocal rank compared to previous models. Liu et al. introduced a generative transformer with knowledge-guided decoding for academic KG completion, which incorporates pertinent knowledge from the training corpus to provide guidance. Chen et al. developed a hybrid transformer with multi-level fusion to tackle challenges in multimodal KG completion tasks. This model integrates visual and textual representations through coarse-grained prefix-guided interaction and fine-grained correlation-aware fusion modules.

#### 3) Recommender Systems

Graph transformers have been effectively utilized in recommender systems by combining generative self-supervised learning with a graph transformer architecture. Xia et al. used the generative self-supervised learning method to extract representations from the data in an unsupervised manner and utilized graph transformer architecture to capture intricate relationships between users and items in the recommendation system. Li et al. introduced a new method for recommender systems that leverages graph transformers (GFormer). Their approach automates the self-supervision augmentation process through a technique called rationale-aware generative self-supervised learning. This technique identifies informative patterns in user-item interactions. The proposed recommender system utilizes a special type of collaborative rationale discovery to selectively augment the self-supervision while preserving the overall relationships between users and items. The rationale-aware self-supervised learning in the graph transformer enables graph collaborative filtering. While challenges remain in areas like graph construction, network design, model optimization, computation efficiency, and handling diverse user behaviors, experiments show that the approach consistently outperforms baseline models on various datasets.

### C. Graph-level Tasks

Graph-level tasks aim to learn graph representations or predict graph attributes based on graph structure and node features.

#### 1) Molecular Property Prediction

Graph transformers are powerful tools for molecular property prediction, utilizing the graph structure of molecules to capture essential structural and semantic information. Chen et al. proposed Algebraic Graph-Assisted Bidirectional Transformer (AGBT) framework which integrates complementary 3D molecular information into graph invariants, rectifying the oversight of three-dimensional stereochemical information in certain machine learning models. Li et al utilized Knowledge-Guided Pre-training of Graph Transformer (KPGT) in a self-supervised learning framework which emphasizes the importance of chemical bonds and models the structural information of molecular graphs. Buterez et al. proposed transfer learning with graph transformer to enhance molecular property prediction on sparse and costly high-fidelity data.

#### 2) Graph Clustering

Graph transformers have been increasingly utilized in the field of Graph Clustering, offering innovative methodologies and overcoming significant challenges. Yun et al. proposed a graph transformer network to generate new graph structures for identifying useful connections between unconnected nodes on the original graph while learning effective node representation on the new graphs in an end-to-end fashion. Gao et al. proposed a patch graph transformer (PatchGT) that segments a graph into patches based on spectral clustering without any trainable parameters and allows the model to first use GNN layers to learn patch-level representations and then use transformer to obtain graph-level representations. These methodologies have addressed issues such as the limitations of the local attention mechanism and difficulties in learning high-level information, leading to enhanced graph representation, improved model performance, and effective node representation.

#### 3) Graph Synthesis

Graph transformers have been applied in the field of graph synthesis to improve graph data mining and representation learning. Existing graph transformers with Positional Encodings have limitations in node classification tasks on complex graphs, as they do not fully capture the local node properties. To address this, Ma et al. introduced the Adaptive Graph Transformer (AGT). This model tackles the challenge of extracting structural patterns from graphs in a way that is both effective and efficient. AGT achieves this by learning from two different graph perspectives: centrality and subgraph views. This approach has been shown to achieve state-of-the-art performance on real-world web graphs and synthetic graphs characterized by heterophily and noise. Jiang et al. proposed an Anchor Graph Transformer (AGformer) that leverages an anchor graph model to perform more efficient and robust node-to-node message passing for overcoming the computational cost and sensitivity to graph noises of regular graph transformers. Zhu et al. developed a Hierarchical Scalable Graph Transformer (HSGT) which scales the transformer architecture to node representation learning tasks on large-scale graphs by utilizing graph hierarchies and sampling-based training methods.

### D. Other Application Scenarios

Graph transformers have a wide range of applications beyond graph-structured data. They can also be utilized in scenarios involving text, images, or videos.

#### 1) Text Summarization

Text summarization is a crucial aspect of NLP which has been significantly advanced with the introduction of Graph transformers. These models utilize extractive, abstractive, and hybrid methods for summarization. Extractive summarization selects and extracts key sentences or phrases from the original text to create a summary. In contrast, abstractive summarization interprets the core concepts in the text and produces a concise summary. Hybrid summarization combines the advantages of both approaches. Despite the progress, challenges remain in text comprehension, main idea identification, and coherent summary generation. Nevertheless, the application of graph transformers in text summarization has demonstrated promising outcomes in terms of summary quality and efficiency.

#### 2) Image Captioning

Graph transformers have emerged as a potent tool within the domain of image captioning, offering a structured representation of images and efficiently processing them to produce descriptive captions. Techniques such as Transforming Scene Graphs (TSG) leverage multi-head attention to architect graph neural networks for embedding scene graphs, which encapsulate a myriad of specific knowledge to facilitate the generation of words across various parts of speech. Despite encountering challenges, such as training complexity, absence of contextual information, and lack of fine-grained details in the extracted features, graph transformers have exhibited promising outcomes. They have enhanced the quality of generated sentences and attained state-of-the-art performance in image captioning endeavors.

#### 3) Image Generation

Graph transformers have been effectively utilized in image generation, as demonstrated by various research studies. Sortino et al. proposed a transformer-based method conditioned by scene graphs for image generation, employing a decoder to sequentially compose images. Zhang et al. proposed StyleSwin which uses transformers in constructing a generative adversarial network for high-resolution image creation. Despite challenges like redundant interactions and the requirement for intricate architectures, these studies have exhibited promising outcomes in terms of image quality and variety.

#### 4) Video Generation

Graph transformers have been extensively applied in the field of Video Generation. Xiao et al. proposed a Video Graph Transformer (VGT) model, which utilizes a dynamic graph transformer module to encode videos, capturing visual objects, their relationships, and dynamics. It incorporates disentangled video and text transformers for comparing relevance between the video and text. Wu et al. proposed The Object-Centric Video Transformer (OCVT) which adopts an object-centric strategy to break down scenes into tokens suitable for a generative video transformer and understanding the intricate spatiotemporal dynamics of multiple interacting objects within a scene. Yan et al. developed VideoGPT, which learns downsampled discrete latent representations of a raw video through 3D convolutions and axial self-attention. A GPT-like architecture is then used to model the discrete latent in a spatiotemporal manner using position encodings. Tulyakov et al. proposed the MoCoGAN model which creates a video by mapping a series of random vectors to a sequence of video frames. Despite the challenges in capturing complex spatio-temporal dynamics in videos, these methodologies have exhibited promising outcomes across various facets of video generation, ranging from question answering to video summarization and beyond.

## VI. OPEN ISSUES AND FUTURE DIRECTIONS

Despite their immense potential for learning from graph-structured data, graph transformers still face open issues and challenges that require further exploration. Here we highlight some of these open challenges.

### A. Scalability and Efficiency

The scalability and efficiency of graph transformers pose considerable challenges due to their substantial memory and computational requirements especially when employing global attention mechanisms to deal with large-scale graphs. These challenges are further amplified in deep architectures which are susceptible to overfitting and over-smoothing. To address these issues, several potential strategies can be proposed:
1) Developing efficient attention mechanisms, such as linear, sparse and low-rank attention, to reduce the complexity and memory usage of graph transformers.
2) Applying graph sparsification or coarsening techniques to decrease the size and density of graphs while maintaining their crucial structural and semantic information.
3) Using graph partitioning or sampling methods to divide large graphs into smaller subgraphs or batches for parallel or sequential processing.
4) Exploring graph distillation or compression methods to create compact and effective graph transformer models, which are suitable for deployment on resource-limited devices.
5) Investigating regularization or normalization techniques, such as dropout, graph diffusion, convolution, and graph spectral normalization, to prevent overfitting and over-smoothing in graph transformer models.

### B. Generalization and Robustness

Graph transformers often face challenges when it comes to generalizing to graphs that they have not encountered before or that fall outside of their usual distribution. This is especially true for graphs that have different sizes, structures, features, and domains. Additionally, graph transformers can be vulnerable to adversarial attacks and noisy inputs, which can result in a decline in performance and the production of misleading results. In order to improve the generalization and robustness of graph transformers, the following strategies could be taken into account:
1) Developing adaptive and flexible attention mechanisms, such as dynamic attention, span-adaptive attention, and multi-head attention with different scales, to accommodate varying graphs and tasks.
2) Applying domain adaptation or transfer learning techniques to facilitate learning from multiple source domains and transfer the knowledge from source domains to target domains.
3) Exploring meta-learning or few-shot learning techniques to enable learning from limited data and rapid adaptation to new tasks.
4) Designing robust and secure attention mechanisms, such as adversarial attention regularization, attention masking, and attention perturbation, to resist adversarial attacks and noisy inputs.
5) Evaluating the uncertainty and reliability of graph transformer models using probabilistic or Bayesian methods, such as variational inference, Monte Carlo dropout, and deep ensembles.

### C. Interpretability and Explainability

Graph transformers commonly regarded as black box models present significant challenges in terms of interpretability and explainability. The lack of sufficient justification and evidence for their decisions can undermine their credibility and transparency. To address this issue, several approaches can be considered:
1) Developing transparent and interpretable attention mechanisms, such as attention visualization, attention attribution, and attention pruning, to highlight the importance and relevance of different nodes and edges in graphs.
2) Applying explainable artificial intelligence (XAI) techniques, such as saliency maps, influence functions, and counterfactual explanations, to analyze and understand the behaviour and logic of graph transformer models.
3) Exploring natural language generation techniques, such as template-based generation, neural text generation, and question-answering generation, to produce natural language explanations for outputs or actions of graph transformer models.
4) Investigating human-in-the-loop methods, such as active learning, interactive learning, and user studies, to incorporate human guidance in the learning or evaluation process of graph transformer models.

### D. Learning on Dynamic Graphs

Graphs, which are frequently characterized by their dynamic and intricate nature, possess the ability to transform over time as a result of the addition or removal of nodes and edges, as well as the modification of node and edge attributes. Moreover, these graphs may exhibit diverse types and modalities of nodes and edges. In order to empower graph transformers to effectively manage such dynamic graphs, it is advisable to explore the following strategies:
1) Developing temporal and causal attention mechanisms, such as recurrent, temporal, and causal attention, to capture the temporal and causal evolution of graphs.
2) Applying continual learning techniques on dynamic graphs to void forgetting previous knowledge and retraining.
3) Exploring multimodal attention mechanisms, such as image-text, audio-visual, and heterogeneous attention, to integrate multimodal nodes and edges.
4) Leveraging multi-level and multi-layer attention mechanisms, such as node-edge, graph-graph, and hypergraph attention, to aggregate information from different levels and layers of graphs.

### E. Data Quality and Sparsity

The quality and sparsity of graph data can significantly impact the performance and effectiveness of graph transformers. Graph data often suffer from issues such as missing values, noisy labels, imbalanced classes, and sparse connections. Additionally, the collection and labeling of graph data can be expensive and time-consuming, limiting the availability of high-quality datasets for training and evaluation. To address these challenges, several strategies can be explored:

1) Developing data augmentation techniques for graphs, such as node and edge perturbation, subgraph sampling, and graph manipulation, to increase the diversity and quantity of training data.
2) Applying semi-supervised or unsupervised learning methods, such as self-training, co-training, and contrastive learning, to leverage unlabeled data and reduce the reliance on labeled data.
3) Exploring graph imputation or completion techniques, such as matrix factorization, graph neural networks, and diffusion models, to handle missing values and sparse connections in graphs.
4) Utilizing active learning or curriculum learning strategies to identify and prioritize the most informative or representative graph samples for labeling and training.
5) Investigating data quality assessment and improvement methods, such as outlier detection, label cleaning, and data validation, to enhance the reliability and consistency of graph datasets.

## VII. CONCLUSION

This survey provides a comprehensive overview of recent advancements and challenges in graph transformer research. We have discussed the primary architectures of graph transformers, their design perspectives, and applications. We have also highlighted open issues and future directions for research on graph transformers. Graph transformers are a powerful tool for learning from graph-structured data, but they still face several challenges that need to be addressed for practical applications. We hope that this survey will serve as a valuable resource for researchers and practitioners in the field of graph transformers.