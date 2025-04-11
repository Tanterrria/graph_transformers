# Graph Transformers: A Survey

## Abstract

Graph transformers are a recent advancement in machine learning, offering a new class of neural network models for graph-structured data. The synergy between transformers and graph learning demonstrates strong performance and ver- satility across various graph-related tasks. This survey provides an in-depth review of recent progress and challenges in graph transformer research. We begin with foundational concepts of graphs and transformers. We then explore design perspec- tives of graph transformers, focusing on how they integrate graph inductive biases and graph attention mechanisms into the transformer architecture. Furthermore, we propose a taxonomy classifying graph transformers based on depth, scalability, and pre-training strategies, summarizing key principles for effective development of graph transformer models. Beyond technical analysis, we discuss the applications of graph transformer models for node-level, edge-level, and graph-level tasks, exploring their potential in other application scenarios as well. Finally, we identify remaining challenges in the field, such as scalability and efficiency, generalization and robustness, interpretability and explainability, dynamic and complex graphs, as well as data quality and diversity, charting future directions for graph transformer research.

**Short description:** Overview of graph transformers, their architecture, applications, and future challenges in machine learning.

**Russian translation:**
Графовые трансформеры представляют собой недавнее достижение в области машинного обучения, предлагая новый класс нейронных сетевых моделей для данных с графовой структурой. Синергия между трансформерами и обучением на графах демонстрирует высокую производительность и универсальность в различных задачах, связанных с графами. В данном обзоре представлен детальный анализ последних достижений и проблем в исследованиях графовых трансформеров. Мы начинаем с основополагающих концепций графов и трансформеров. Затем рассматриваем подходы к проектированию графовых трансформеров, уделяя особое внимание тому, как они интегрируют графовые индуктивные смещения и механизмы графового внимания в архитектуру трансформера. Кроме того, мы предлагаем таксономию, классифицирующую графовые трансформеры на основе глубины, масштабируемости и стратегий предварительного обучения, обобщая ключевые принципы эффективной разработки моделей графовых трансформеров. Помимо технического анализа, мы обсуждаем применение моделей графовых трансформеров для задач на уровне узлов, ребер и графов, а также исследуем их потенциал в других сценариях применения. Наконец, мы определяем оставшиеся проблемы в этой области, такие как масштабируемость и эффективность, обобщение и устойчивость, интерпретируемость и объяснимость, динамические и сложные графы, а также качество и разнообразие данных, намечая будущие направления исследований графовых трансформеров.

____

## I. INTRODUCTION

GRAPHS, as data structures with high expressiveness, are widely used to present complex data in diverse domains, such as social media, knowledge graphs, biology, chemistry, and transportation networks. They capture both structural and semantic information from data, facilitating various tasks, such as recommendation, question answering, anomaly detection, sentiment analysis, text generation, and information retrieval. To effectively deal with graph- structured data, researchers have developed various graph learning models, such as graph neural networks (GNNs), learning meaningful representations of nodes, edges and graphs. Particularly, GNNs following the message-passing framework iteratively aggregate neighboring information and update node representations, leading to impressive performance on various graph-based tasks. Applications ranging from information extraction to recommender systems have benefited from GNN modelling of knowledge graphs.

More recently, the graph transformer, as a newly arisen and potent graph learning method, has attracted great attention in both academic and industrial communities. Graph transformer research is inspired by the success of transformers in natural language processing (NLP) and computer vision (CV), coupled with the demonstrated value of GNNs. Graph transformers incorporate graph inductive bias (e.g., prior knowledge or assumptions about graph properties) to effectively process graph data. Furthermore, they can adapt to dynamic and heterogeneous graphs, leveraging both node and edge features and attributes. Various adaptations and expansions of graph transformers have shown their superiority in tackling diverse challenges of graph learning, such as large-scale graph processing. Furthermore, graph transformers have been successfully employed in various domains and applications, demonstrating their effectiveness and versatility.

Existing surveys do not adequately cover the latest advancements and comprehensive applications of graph transformers. In addition, most do not provide a systematic taxonomy of graph transformer models. For instance, Chen et al. focused primarily on the utilization of GNNs and graph transformers in CV, but they failed to summarize the taxonomy of graph transformer models and ignored other domains, such as NLP. Similarly, Müller et al. offered an overview of graph transformers and their theoretical properties, but they did not provide a comprehensive review of existing methods or evaluate their performance on various tasks. Lastly, Min et al. concentrated on the architectural design aspect of graph transformers, offering a systematic evaluation of different components on different graph benchmarks, but they did not include significant applications of graph transformers or discuss open issues in this field.

**Short description:** Introduction to graphs, their applications, and the emergence of graph transformers as a powerful learning method.

**Russian translation:**
ГРАФЫ, как структуры данных с высокой выразительностью, широко используются для представления сложных данных в различных областях, таких как социальные медиа, базы знаний, биология, химия и транспортные сети. Они захватывают как структурную, так и семантическую информацию из данных, облегчая выполнение различных задач, таких как рекомендации, ответы на вопросы, обнаружение аномалий, анализ настроений, генерация текста и поиск информации. Для эффективной работы с графовыми данными исследователи разработали различные модели обучения на графах, такие как графовые нейронные сети (GNN), которые учатся осмысленным представлениям узлов, ребер и графов. В частности, GNN, следующие за фреймворком передачи сообщений, итеративно агрегируют информацию о соседях и обновляют представления узлов, что приводит к впечатляющей производительности в различных задачах на графах. Приложения, начиная от извлечения информации до рекомендательных систем, выиграли от моделирования баз знаний с помощью GNN.

Совсем недавно графовый трансформер, как новый и мощный метод обучения на графах, привлек большое внимание как в академических, так и в промышленных кругах. Исследования графовых трансформеров вдохновлены успехом трансформеров в обработке естественного языка (NLP) и компьютерном зрении (CV), в сочетании с доказанной ценностью GNN. Графовые трансформеры включают графовую индуктивную предвзятость (например, предварительные знания или предположения о свойствах графа) для эффективной обработки графовых данных. Более того, они могут адаптироваться к динамическим и гетерогенным графам, используя как признаки узлов, так и ребер. Различные адаптации и расширения графовых трансформеров показали свое превосходство в решении разнообразных задач обучения на графах, таких как обработка крупномасштабных графов. Кроме того, графовые трансформеры успешно применяются в различных областях и приложениях, демонстрируя свою эффективность и универсальность.

Существующие обзоры неадекватно охватывают последние достижения и комплексные применения графовых трансформеров. Кроме того, большинство из них не предоставляют систематической таксономии моделей графовых трансформеров. Например, Чен и др. сосредоточились в основном на использовании GNN и графовых трансформеров в CV, но они не смогли обобщить таксономию моделей графовых трансформеров и проигнорировали другие области, такие как NLP. Аналогично, Мюллер и др. предложили обзор графовых трансформеров и их теоретических свойств, но они не предоставили всестороннего обзора существующих методов или оценки их производительности на различных задачах. Наконец, Мин и др. сосредоточились на аспекте архитектурного проектирования графовых трансформеров, предлагая систематическую оценку различных компонентов на различных графовых бенчмарках, но они не включили значительные применения графовых трансформеров или обсуждение открытых проблем в этой области.

_______

To fill these gaps, this survey aims to present a comprehensive and systematic review of recent advancements and challenges in graph transformer research from both design and application perspectives. In comparison to existing surveys, our main contributions are as follows:

1) We provide a comprehensive review of the design perspectives of graph transformers, including graph inductive bias and graph attention mechanisms. We classify these techniques into different types and discuss their advantages and limitations.

2) We present a novel taxonomy of graph transformers based on their depth, scalability, and pre-training strategy. We also provide a guide to choosing effective graph transformer architectures for different tasks and scenarios.

3) We review the application perspectives of graph transformers in various graph learning tasks, as well as the application scenarios in other domains, such as NLP and CV tasks.

4) We identify the crucial open issues and future directions of graph transformer research, such as the scalability, generalization, interpretability, and explainability of models, efficient temporal graph learning, and data-related issues.

**Short description:** Overview of the survey's main contributions in reviewing graph transformer research and applications.

**Russian translation:**
Для заполнения этих пробелов, данный обзор направлен на представление всестороннего и систематического анализа последних достижений и проблем в исследованиях графовых трансформеров с точки зрения как проектирования, так и применения. По сравнению с существующими обзорами, наши основные вклады заключаются в следующем:

1) Мы предоставляем всесторонний обзор подходов к проектированию графовых трансформеров, включая графовую индуктивную предвзятость и механизмы графового внимания. Мы классифицируем эти техники на различные типы и обсуждаем их преимущества и ограничения.

2) Мы представляем новую таксономию графовых трансформеров, основанную на их глубине, масштабируемости и стратегии предварительного обучения. Мы также предоставляем руководство по выбору эффективных архитектур графовых трансформеров для различных задач и сценариев.

3) Мы рассматриваем аспекты применения графовых трансформеров в различных задачах обучения на графах, а также сценарии применения в других областях, таких как задачи NLP и CV.

4) Мы определяем ключевые открытые проблемы и будущие направления исследований графовых трансформеров, такие как масштабируемость, обобщение, интерпретируемость и объяснимость моделей, эффективное обучение на временных графах и проблемы, связанные с данными.

\[\mathbf{h}_v^{(l+1)} = \phi \left( \mathbf{h}_v^{(l)}, \bigoplus_{u \in \mathcal{N}(v)} f(\mathbf{h}_u^{(l)}, \mathbf{h}_v^{(l)}, e_{uv}) \right)\]

___

An overview of this paper is depicted in Figure 1. The subsequent survey is structured as follows: Section II introduces notations and preliminaries pertaining to graphs and transformers. Section III delves into the design perspectives of graph transformers that encompass graph inductive bias and graph attention mechanisms. Section IV presents a taxonomy of graph transformers categorizing them based on their depth, scalability and pre-training strategy. Additionally, a guide is provided for selecting appropriate graph transformer models for diverse tasks and domains. Section V explores the application perspectives of graph transformers on various node-level, edge-level, and graph-level tasks, along with other application scenarios. Section VI identifies open issues and future directions for research on graph transformers. Lastly, Section VII concludes the paper and highlights its main contributions.

**Short description:** Overview of the paper's structure and key sections.

**Russian translation:**
Обзор данной статьи представлен на Рисунке 1. Последующий обзор структурирован следующим образом: Раздел II вводит обозначения и предварительные сведения, касающиеся графов и трансформеров. Раздел III углубляется в проектные аспекты графовых трансформеров, охватывающие графовую индуктивную предвзятость и механизмы графового внимания. Раздел IV представляет таксономию графовых трансформеров, классифицируя их на основе глубины, масштабируемости и стратегии предварительного обучения. Кроме того, предоставляется руководство по выбору подходящих моделей графовых трансформеров для различных задач и областей. Раздел V исследует аспекты применения графовых трансформеров на различных задачах уровня узлов, ребер и графов, а также в других сценариях применения. Раздел VI определяет открытые вопросы и будущие направления исследований графовых трансформеров. Наконец, Раздел VII завершает статью и подчеркивает ее основные вклады.

____

## II. NOTATIONS AND PRELIMINARIES

In this section, we present fundamental notations and concepts utilized throughout this survey paper. Additionally, we provide a concise summary of current methods for graph learning and self-attention mechanisms which serve as the basis for graph transformers. Table I includes mathematical notations used in this paper.

**Short description:** Introduction to notations and concepts used in the paper, including graph learning and self-attention mechanisms.

**Russian translation:**
В этом разделе представлены основные обозначения и концепции, используемые на протяжении всей статьи. Кроме того, мы предоставляем краткое резюме текущих методов обучения на графах и механизмов самовнимания, которые служат основой для графовых трансформеров. Таблица I включает математические обозначения, используемые в этой статье.

| Notation | Definition |
|----------|------------|
| \( G = (V, E) \) | A graph with node set \( V \) and edge set \( E \) |
| \( N \) | Number of nodes |
| \( M \) | Number of edges |
| \( \mathbf{A} \in \mathbb{R}^{N \times N} \) | Adjacency matrix of graph \( G \) |
| \( \mathbf{X} \in \mathbb{R}^{N \times d_n} \) | Node feature matrix, \( \mathbf{x}_i \in \mathbf{X} \) |
| \( \mathbf{F} \in \mathbb{R}^{M \times d_e} \) | Edge feature matrix |
| \( \mathbf{h}_v^{(l)} \) | Hidden state of node \( v \) at layer \( l \) |
| \( \phi \) | An update function for node states |
| \( \bigoplus \) | An aggregation function for neighbor states |
| \( \mathcal{N}(v) \) | Neighbor set of node \( v \) |
| \( f \) | A message function for the node and edge states |
| \( \mathbf{Q}, \mathbf{K}, \mathbf{V} \) | Query, key and value matrices for self-attention |
| \( d_k \) | Dimension of query and key matrices |
| \( p_i \) | Positional encoding of node \( v_i \) |
| \( d(i, j) \) | The shortest path distance between node \( v_i \) and node \( v_j \) |
| \( e_{ij} \) | Edge feature between node \( v_i \) and node \( v_j \) |
| \( a_{ij} \) | Attention score between node \( v_i \) and node \( v_j \) |
| \( W, b \) | Learnable parameters for the self-attention layer |
| \( \mathbf{L} = \mathbf{I}_N - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2} \) | Normalized graph Laplacian matrix |
| \( \mathbf{U} \) | Eigenvectors matrix of \( \mathbf{L} \) |



________


A graph is a data structure consisting of a set of nodes (or vertices) \( V \) and a set of edges (or links) \( E \) that connect pairs of nodes. Formally, a graph can be defined as \( G = (V,E) \), where \( V = \{v_1,v_2,\ldots,v_N\} \) is node set with \( N \) nodes and \( E = \{e_1,e_2,\ldots,e_M\} \) is edge set with \( M \) edges. Edge \( e_k = (v_i , v_j) \) indicates the connection between node \( v_i \) and node \( v_j \), where \( i,j \in \{1,2,\ldots,N\} \) and \( k \in \{1,2,\ldots,M\} \). A graph can be represented by an adjacency matrix \( \mathbf{A} \in \mathbb{R}^{N \times N} \), where \( A_{ij} \) indicates the presence or absence of an edge between node \( v_i \) and node \( v_j \). Alternatively, a graph can be represented by the edge list \( E \in \mathbb{R}^{M \times 2} \), where each row of \( E \) contains the indices of two nodes connected by an edge. A graph can also have node features and edge features that describe the attributes or properties of nodes and edges, respectively. The features of the nodes can be represented by a feature matrix \( \mathbf{X} \in \mathbb{R}^{N \times d_n} \), where \( d_n \) is the dimension of the node features. The edge features can be represented by a feature tensor \( \mathbf{F} \in \mathbb{R}^{M \times d_e} \), where \( d_e \) is the dimension of the edge features.

**Short description:** Explanation of graph data structures, including nodes, edges, adjacency matrix, and feature representations.

**Russian translation:**
Граф — это структура данных, состоящая из множества узлов (или вершин) \( V \) и множества рёбер (или связей) \( E \), которые соединяют пары узлов. Формально граф можно определить как \( G = (V, E) \), где \( V = \{v_1, v_2, \ldots, v_N\} \) — множество узлов с \( N \) узлами, а \( E = \{e_1, e_2, \ldots, e_M\} \) — множество рёбер с \( M \) рёбрами. Ребро \( e_k = (v_i, v_j) \) указывает на связь между узлами \( v_i \) и \( v_j \), где \( i, j \in \{1, 2, \ldots, N\} \) и \( k \in \{1, 2, \ldots, M\} \). Граф может быть представлен матрицей смежности \( \mathbf{A} \in \mathbb{R}^{N \times N} \), где \( A_{ij} \) указывает на наличие или отсутствие ребра между узлами \( v_i \) и \( v_j \). Альтернативно, граф может быть представлен списком рёбер \( E \in \mathbb{R}^{M \times 2} \), где каждая строка \( E \) содержит индексы двух узлов, соединённых ребром. Граф также может иметь признаки узлов и рёбер, которые описывают атрибуты или свойства узлов и рёбер соответственно. Признаки узлов могут быть представлены матрицей признаков \( \mathbf{X} \in \mathbb{R}^{N \times d_n} \), где \( d_n \) — размерность признаков узлов. Признаки рёбер могут быть представлены тензором признаков \( \mathbf{F} \in \mathbb{R}^{M \times d_e} \), где \( d_e \) — размерность признаков рёбер.

_____

Graph learning refers to the task of acquiring low-dimensional vector representations, also known as embeddings for nodes, edges, or the entire graph. These embeddings are designed to capture both structural and semantic information of the graph. GNNs are a type of neural network model that excels at learning from graph-structured data. They achieve this by propagating information along edges and aggregating information from neighboring nodes. GNNs can be categorized into two main groups: spectral methods and spatial methods.

Spectral methods are based on graph signal processing and graph Fourier transform, implementing convolution operations on graphs in the spectral domain. The Fourier graph transform is defined as \( \hat{X} = \mathbf{U}^T \mathbf{X} \mathbf{U} \), where \( \hat{X} \) is the spectral representation of node feature matrix \( \mathbf{X} \) and \( \mathbf{U} \) is the eigenvector matrix of the normalized graph Laplacian matrix \( \mathbf{L} = \mathbf{I}_N - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2} \), where \( \mathbf{I}_N \) is the identity matrix and \( \mathbf{D} \) is diagonal degree matrix with \( D_{ii} = \sum_{j=1}^N A_{ij} \). Spectral methods can capture global information about the graph, but they suffer from high computational complexity, poor scalability, and the lack of generalization to unseen graphs.

Spatial methods are based on message-passing and neighborhood aggregation, implementing convolution operations on graphs in the spatial domain. The message-passing framework is defined as:

\[
\mathbf{h}_v^{(l+1)} = \phi \left( \mathbf{h}_v^{(l)}, \bigoplus_{u \in \mathcal{N}(v)} f(\mathbf{h}_u^{(l)}, \mathbf{h}_v^{(l)}, e_{uv}) \right)
\]

where \( \mathbf{h}_v^{(l)} \) is the hidden state of node \( v \) at layer \( l \), \( \mathcal{N}(v) \) is the neighborhood of node \( v \), \( \phi \) is an update function, \( \bigoplus \) is an aggregation function, and \( f \) is a message function that computes messages between nodes. \( e_{uv} \) is the feature vector of the edge between nodes \( u \) and \( v \). Spatial methods can capture local information of the graph, but they have limitations in modeling long-range dependencies, complex interactions and heterogeneous structures.

**Short description:** Overview of spatial methods in GNNs, focusing on message-passing and neighborhood aggregation, and their limitations.

**Russian translation:**

Обучение на графах относится к задаче получения векторных представлений низкой размерности, также известных как встраивания, для узлов, рёбер или всего графа. Эти встраивания предназначены для захвата как структурной, так и семантической информации графа. Графовые нейронные сети (GNN) — это тип модели нейронной сети, которая превосходно обучается на данных с графовой структурой. Они достигают этого, распространяя информацию по рёбрам и агрегируя информацию от соседних узлов. GNN можно разделить на две основные группы: спектральные методы и пространственные методы.

Спектральные методы основаны на обработке сигналов на графах и графовом преобразовании Фурье, реализуя операции свёртки на графах в спектральной области. Графовое преобразование Фурье определяется как \( \hat{X} = \mathbf{U}^T \mathbf{X} \mathbf{U} \), где \( \hat{X} \) — спектральное представление матрицы признаков узлов \( \mathbf{X} \), а \( \mathbf{U} \) — матрица собственных векторов нормализованной графовой лапласиановой матрицы \( \mathbf{L} = \mathbf{I}_N - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2} \), где \( \mathbf{I}_N \) — единичная матрица, а \( \mathbf{D} \) — диагональная матрица степеней с \( D_{ii} = \sum_{j=1}^N A_{ij} \). Спектральные методы могут захватывать глобальную информацию о графе, но они страдают от высокой вычислительной сложности, плохой масштабируемости и отсутствия обобщения на невидимые графы.

Пространственные методы основаны на передаче сообщений и агрегации соседних узлов, реализуя операции свёртки на графах в пространственной области. Фреймворк передачи сообщений определяется как:

\[
\mathbf{h}_v^{(l+1)} = \phi \left( \mathbf{h}_v^{(l)}, \bigoplus_{u \in \mathcal{N}(v)} f(\mathbf{h}_u^{(l)}, \mathbf{h}_v^{(l)}, e_{uv}) \right)
\]

где \( \mathbf{h}_v^{(l)} \) — скрытое состояние узла \( v \) на слое \( l \), \( \mathcal{N}(v) \) — окрестность узла \( v \), \( \phi \) — функция обновления, \( \bigoplus \) — функция агрегации, а \( f \) — функция сообщений, которая вычисляет сообщения между узлами. \( e_{uv} \) — вектор признаков ребра между узлами \( u \) и \( v \). Пространственные методы могут захватывать локальную информацию графа, но они имеют ограничения в моделировании дальних зависимостей, сложных взаимодействий и гетерогенных структур.


___ 

### Understanding the Message-Passing Framework

The message-passing framework can be broken down into several key components:

1. **Hidden States**:
   - \( \mathbf{h}_v^{(l)} \) — current state of node \( v \) at layer \( l \)
   - \( \mathbf{h}_v^{(l+1)} \) — new state of node \( v \) at the next layer \( l+1 \)

2. **Neighbors**:
   - \( \mathcal{N}(v) \) — set of neighbors of node \( v \)
   - \( u \in \mathcal{N}(v) \) — each neighbor \( u \) of node \( v \)

3. **Message Function**:
   - \( f(\mathbf{h}_u^{(l)}, \mathbf{h}_v^{(l)}, e_{uv}) \) — function that creates a message from neighbor \( u \) to node \( v \)
   - It considers:
     - State of the neighbor \( \mathbf{h}_u^{(l)} \)
     - State of the current node \( \mathbf{h}_v^{(l)} \)
     - Features of the edge between them \( e_{uv} \)

4. **Aggregation**:
   - \( \bigoplus \) — aggregation operation (e.g., sum, mean, or max)
   - \( \bigoplus_{u \in \mathcal{N}(v)} \) — aggregates messages from all neighbors

5. **Update Function**:
   - \( \phi \) — function that updates the node's state
   - It takes:
     - Current state of the node \( \mathbf{h}_v^{(l)} \)
     - Aggregated messages from neighbors

### Simple Example

Imagine a group of friends (nodes) exchanging information (messages):

1. Each friend (node) has their current opinion (state)
2. They send messages to their friends (neighbors)
3. Each collects all messages from friends
4. Based on received messages and their current opinion, each updates their opinion

Thus, the formula describes how information propagates through the graph and how each node updates its state based on information from its neighbors.

### Понимание фреймворка передачи сообщений

Фреймворк передачи сообщений можно разбить на несколько ключевых компонентов:

1. **Скрытые состояния**:
   - \( \mathbf{h}_v^{(l)} \) — текущее состояние узла \( v \) на слое \( l \)
   - \( \mathbf{h}_v^{(l+1)} \) — новое состояние узла \( v \) на следующем слое \( l+1 \)

2. **Соседи**:
   - \( \mathcal{N}(v) \) — множество соседей узла \( v \)
   - \( u \in \mathcal{N}(v) \) — каждый сосед \( u \) узла \( v \)

3. **Функция сообщений**:
   - \( f(\mathbf{h}_u^{(l)}, \mathbf{h}_v^{(l)}, e_{uv}) \) — функция, которая создаёт сообщение от соседа \( u \) к узлу \( v \)
   - Она учитывает:
     - Состояние соседа \( \mathbf{h}_u^{(l)} \)
     - Состояние текущего узла \( \mathbf{h}_v^{(l)} \)
     - Признаки ребра между ними \( e_{uv} \)

4. **Агрегация**:
   - \( \bigoplus \) — операция агрегации (например, сумма, среднее или максимум)
   - \( \bigoplus_{u \in \mathcal{N}(v)} \) — агрегирует сообщения от всех соседей

5. **Функция обновления**:
   - \( \phi \) — функция, которая обновляет состояние узла
   - Она принимает:
     - Текущее состояние узла \( \mathbf{h}_v^{(l)} \)
     - Агрегированные сообщения от соседей

### Простой пример

Представьте, что у вас есть группа друзей (узлы), и они обмениваются информацией (сообщениями):

1. Каждый друг (узел) имеет своё текущее мнение (состояние)
2. Они отправляют сообщения своим друзьям (соседям)
3. Каждый собирает все сообщения от друзей
4. На основе полученных сообщений и своего текущего мнения, каждый обновляет своё мнение

Таким образом, формула описывает процесс, как информация распространяется по графу, и как каждый узел обновляет своё состояние на основе информации от соседей.

## B. Self-attention and Transformers

Self-attention is a mechanism that enables a model to learn to focus on pertinent sections of input or output sequences. It calculates a weighted sum of all elements in a sequence with weights determined by the similarity between each element and a query vector. Formally, self-attention is defined as:

\[
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{QK}^T}{\sqrt{d_k}}\right) \mathbf{V}
\]

where \( \mathbf{Q}, \mathbf{K}, \mathbf{V} \) are query, key, and value matrices, respectively. \( d_k \) is the dimension of the query and key matrices. Self-attention can capture long-range dependencies, global context, and variable-length sequences without using recurrence or convolution.

**Short description:** Introduction to self-attention and its role in transformers, highlighting its ability to capture dependencies and context.

**Russian translation:**

## B. Самовнимание и трансформеры

Самовнимание — это механизм, который позволяет модели учиться фокусироваться на значимых частях входных или выходных последовательностей. Оно вычисляет взвешенную сумму всех элементов в последовательности с весами, определяемыми сходством между каждым элементом и вектором запроса. Формально самовнимание определяется как:

\[
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{QK}^T}{\sqrt{d_k}}\right) \mathbf{V}
\]

где \( \mathbf{Q}, \mathbf{K}, \mathbf{V} \) — это матрицы запросов, ключей и значений соответственно. \( d_k \) — это размерность матриц запросов и ключей. Самовнимание может захватывать дальние зависимости, глобальный контекст и последовательности переменной длины без использования рекуррентности или свёртки.

Transformers are neural network models that use self-attention as the main building block. Transformers consist of two main components: an encoder and a decoder. The encoder takes an input sequence \( X = \{x_1,x_2,...,x_N\} \) and generates a sequence of hidden states \( Z = \{z_1, z_2, . . . , z_N \} \). The decoder takes an output sequence \( Y = \{y_1, y_2, . . . , y_N \} \) and generates a sequence of hidden states \( S = \{s_1, s_2, . . . , s_N \} \). The decoder also uses an attention mechanism to attend to the encoder's hidden states. Formally, the encoder and decoder are defined as:

\[
\begin{align*}
    z_i &= \text{EncoderLayer}(x_i, Z_{<i}), \\
    s_j &= \text{DecoderLayer}(y_j, S_{<j}, Z).
\end{align*}
\]

Here, EncoderLayer and DecoderLayer are composed of multiple self-attention and feed-forward sublayers. Transformers can achieve state-of-the-art results on various tasks, such as machine translation, text mining, document comprehending, image retrieval, visual question answering, image generation, and recommendation systems. An overview of vanilla transformer is shown in Figure 2.

Graph transformers integrate graph inductive bias into transformer to acquire knowledge from graph-structured data. By employing self-attention mechanisms on nodes and edges, graph transformers can effectively capture both local and global information of the graph. In particular, graph transformers exhibit the ability to handle heterogeneous graphs containing diverse types of nodes and edges, as well as complex graphs featuring higher-order structures.

**Short description:** Overview of transformers and graph transformers, highlighting their components and applications.

**Russian translation:**

Трансформеры — это модели нейронных сетей, которые используют самовнимание в качестве основного строительного блока. Трансформеры состоят из двух основных компонентов: кодировщика и декодировщика. Кодировщик принимает входную последовательность \( X = \{x_1,x_2,...,x_N\} \) и генерирует последовательность скрытых состояний \( Z = \{z_1, z_2, . . . , z_N \} \). Декодировщик принимает выходную последовательность \( Y = \{y_1, y_2, . . . , y_N \} \) и генерирует последовательность скрытых состояний \( S = \{s_1, s_2, . . . , s_N \} \). Декодировщик также использует механизм внимания для обращения к скрытым состояниям кодировщика. Формально, кодировщик и декодировщик определяются как:

\[
\begin{align*}
    z_i &= \text{EncoderLayer}(x_i, Z_{<i}), \\
    s_j &= \text{DecoderLayer}(y_j, S_{<j}, Z).
\end{align*}
\]

Здесь EncoderLayer и DecoderLayer состоят из нескольких подслоёв самовнимания и прямого распространения. Трансформеры могут достигать передовых результатов в различных задачах, таких как машинный перевод, текстовый майнинг, понимание документов, поиск изображений, визуальные вопросы и ответы, генерация изображений и рекомендательные системы. Обзор стандартного трансформера показан на Рисунке 2.

Графовые трансформеры интегрируют графовый индуктивный уклон в трансформер для получения знаний из данных с графовой структурой. Используя механизмы самовнимания на узлах и рёбрах, графовые трансформеры могут эффективно захватывать как локальную, так и глобальную информацию графа. В частности, графовые трансформеры демонстрируют способность обрабатывать гетерогенные графы, содержащие различные типы узлов и рёбер, а также сложные графы с высокопорядковыми структурами.

## III. DESIGN PERSPECTIVES OF GRAPH TRANSFORMERS

In this section, we discuss the primary architectures of graph transformers, aiming to explore their design perspectives in depth. Particularly, we will focus on two key components: graph inductive biases and graph attention mechanisms, to understand how these elements shape graph transformer models' capabilities.

**Short description:** Exploration of graph transformer architectures, focusing on graph inductive biases and attention mechanisms.

**Russian translation:**

## III. ПЕРСПЕКТИВЫ ДИЗАЙНА ГРАФОВЫХ ТРАНСФОРМЕРОВ

В этом разделе мы обсуждаем основные архитектуры графовых трансформеров, стремясь глубже изучить их дизайнерские перспективы. В частности, мы сосредоточимся на двух ключевых компонентах: графовых индуктивных уклонах и механизмах внимания на графах, чтобы понять, как эти элементы формируют возможности моделей графовых трансформеров.

### A. Graph Inductive Bias

Unlike Euclidean data, such as texts and images, graph data is non-Euclidean data, which has intricate structures and lacks a fixed order and dimensionality, posing difficulties in directly applying standard transformers on graph data. To address this issue, graph transformers incorporate graph inductive bias to encode the structural information of graphs and achieve effective generalization of transformers across new tasks and domains. In this section, we explore the design perspectives of graph transformers through the lens of graph inductive bias. We classify graph inductive bias into four categories: node positional bias, edge structural bias, message-passing bias, and attention bias.

#### 1) Node Positional Bias

Node positional bias is a crucial inductive bias for graph transformers because it provides information about the relative or absolute positions of nodes in a graph. Formally, given a graph \( G = (V, E) \) with \( N \) nodes and \( M \) edges, each node \( v_i \in V \) has a feature vector \( x_i \in \mathbb{R}^{d_n} \). A graph transformer aims to learn a new feature vector \( h_i \in \mathbb{R}^{d_k} \) for each node by applying a series of self-attention layers. A self-attention layer can be defined as:

\[
h_i = \sum_{j=1}^{N} a_{ij} W x_j + b,
\]

where \( a_{ij} \) is the attention score between nodes \( v_i \) and \( v_j \), measuring the relevance or similarity of their features. \( W \) and \( b \) are learnable parameters. However, this self-attention mechanism does not consider the structural and positional information of nodes, which is crucial for capturing graph semantics and inductive biases. Node positional encoding is a way to address this challenge by providing additional positional features to nodes, reflecting their relative or absolute positions in the graph.

**Short description:** Explanation of node positional bias in graph transformers and its role in encoding positional information.

**Russian translation:**

#### 1) Позиционный bias узлов

Позиционный bias узлов является важным индуктивным bias для графовых трансформеров, так как он предоставляет информацию о относительных или абсолютных позициях узлов в графе. Формально, для графа \( G = (V, E) \) с \( N \) узлами и \( M \) рёбрами, каждый узел \( v_i \in V \) имеет вектор признаков \( x_i \in \mathbb{R}^{d_n} \). Графовый трансформер стремится обучить новый вектор признаков \( h_i \in \mathbb{R}^{d_k} \) для каждого узла, применяя серию слоёв самовнимания. Слой самовнимания может быть определён как:

\[
h_i = \sum_{j=1}^{N} a_{ij} W x_j + b,
\]

где \( a_{ij} \) — это оценка внимания между узлами \( v_i \) и \( v_j \), измеряющая релевантность или сходство их признаков. \( W \) и \( b \) — обучаемые параметры. Однако этот механизм самовнимания не учитывает структурную и позиционную информацию узлов, что важно для захвата семантики графа и индуктивных bias. Позиционное кодирование узлов — это способ решения этой задачи, предоставляя дополнительные позиционные признаки узлам, отражающие их относительные или абсолютные позиции в графе.

##### Local Node Positional Encodings

Building on the success of relative positional encodings in NLP, graph transformers leverage a similar concept for local node positional encodings. In NLP, each token receives a feature vector that captures its relative position and relationship to other words in the sequence. Likewise, graph transformers assign feature vectors to nodes based on their distance and relationships with other nodes in the graph. This encoding technique aims to preserve the local connectivity and neighborhood information of nodes, which is critical for tasks like node classification, link prediction, and graph generation.

A proficient approach for integrating local node positional information involves the utilization of one-hot vectors. These vectors represent the hop distance between a node and its neighboring nodes:

\[
\mathbf{p}_i = [I(d(i,j)=1), I(d(i,j)=2), \ldots, I(d(i,j)=\text{max})]
\]

In this equation, \( d(i,j) \) represents the shortest path distance between node \( v_i \) and node \( v_j \) and \( I \) is an indicator function that returns 1 if its argument is true and 0 otherwise. The maximum hop distance is denoted by max. This encoding technique was utilized to enhance Graph Attention Networks (GATs) with relative position-aware self-attention. Another way to incorporate local node positional encodings in a graph is by using learnable embeddings that capture the relationship between two nodes. This approach is particularly useful when a node has multiple neighbors with different edge types or labels. In this case, the local node positional encoding can be learned based on these edge features:

\[
\mathbf{p}_i = [f(e_{ij1}), f(e_{ij2}), \ldots, f(e_{ijl})]
\]

where \( e_{ij} \) is edge feature between node \( v_i \) and node \( v_j \), \( f \) is a learnable function that maps edge features to embeddings and \( l \) is the number of neighbors considered.

**Short description:** Explanation of local node positional encodings in graph transformers, including one-hot vector and learnable embedding approaches.

**Russian translation:**
Опираясь на успех относительных позиционных кодировок в NLP, графовые трансформеры используют аналогичную концепцию для локальных позиционных кодировок узлов. В NLP каждый токен получает вектор признаков, который отражает его относительную позицию и связь с другими словами в последовательности. Аналогично, графовые трансформеры присваивают векторы признаков узлам на основе их расстояния и связей с другими узлами в графе. Эта техника кодирования направлена на сохранение локальной связности и информации о соседях узлов, что критически важно для таких задач, как классификация узлов, предсказание связей и генерация графов.

Эффективный подход для интеграции локальной позиционной информации узлов включает использование one-hot векторов. Эти векторы представляют расстояние в переходах (hop distance) между узлом и его соседними узлами:

\[
\mathbf{p}_i = [I(d(i,j)=1), I(d(i,j)=2), \ldots, I(d(i,j)=\text{max})]
\]

В этом уравнении \( d(i,j) \) представляет кратчайшее расстояние пути между узлом \( v_i \) и узлом \( v_j \), а \( I \) - это индикаторная функция, которая возвращает 1, если её аргумент истинен, и 0 в противном случае. Максимальное расстояние в переходах обозначено как max. Эта техника кодирования была использована для улучшения Graph Attention Networks (GATs) с помощью self-attention, учитывающей относительные позиции. Другой способ включения локальных позиционных кодировок узлов в граф заключается в использовании обучаемых эмбеддингов, которые захватывают связь между двумя узлами. Этот подход особенно полезен, когда узел имеет несколько соседей с разными типами или метками рёбер. В этом случае локальная позиционная кодировка узла может быть обучена на основе этих признаков рёбер:

\[
\mathbf{p}_i = [f(e_{ij1}), f(e_{ij2}), \ldots, f(e_{ijl})]
\]

где \( e_{ij} \) - это признак ребра между узлом \( v_i \) и узлом \( v_j \), \( f \) - это обучаемая функция, которая отображает признаки рёбер в эмбеддинги, а \( l \) - количество рассматриваемых соседей.

To enhance the implementation of local node positional encodings in a broader context, a viable strategy is leveraging graph kernels or similarity functions to evaluate the structural similarity between two nodes within the graph. For instance, when a node exhibits three neighbors with unique subgraph patterns or motifs in their vicinity, its local node positional encoding can be computed as a vector of kernel values between the node and its neighboring nodes:

\[
\mathbf{p}_i = [K(G_i, G_{j1}), K(G_i, G_{j2}), \ldots, K(G_i, G_{jl})]
\]

In this equation, \( G_i \) refers to the subgraph formed by node \( v_i \) and its neighbors. The function \( K \) is a graph kernel function that measures the similarity between two subgraphs. This approach was utilized in the GraphiT model, which incorporates positive definite kernels on graphs as relative positional encodings for graph transformers.

Local node positional encodings enhance the self-attention mechanism of graph transformers by integrating structural and topological information of nodes. This encoding approach offers the advantage of preserving the sparsity and locality of graph structure, resulting in enhanced efficiency and interpretability. However, a limitation of this method is its restricted ability to capture long-range dependencies or global properties of graphs, which are essential for tasks such as graph matching or alignment.

**Short description:** Explanation of graph kernels and their application in local node positional encodings, including advantages and limitations.

**Russian translation:**
Для улучшения реализации локальных позиционных кодировок узлов в более широком контексте, жизнеспособной стратегией является использование графовых ядер или функций сходства для оценки структурного сходства между двумя узлами в графе. Например, когда узел имеет трех соседей с уникальными паттернами или мотивами подграфов в их окрестности, его локальная позиционная кодировка может быть вычислена как вектор значений ядра между узлом и его соседними узлами:

\[
\mathbf{p}_i = [K(G_i, G_{j1}), K(G_i, G_{j2}), \ldots, K(G_i, G_{jl})]
\]

В этом уравнении \( G_i \) относится к подграфу, образованному узлом \( v_i \) и его соседями. Функция \( K \) - это графовая ядерная функция, которая измеряет сходство между двумя подграфами. Этот подход был использован в модели GraphiT, которая включает положительно определенные ядра на графах в качестве относительных позиционных кодировок для графовых трансформеров.

Локальные позиционные кодировки узлов улучшают механизм самовнимания графовых трансформеров, интегрируя структурную и топологическую информацию узлов. Этот подход кодирования предлагает преимущество сохранения разреженности и локальности структуры графа, что приводит к повышению эффективности и интерпретируемости. Однако ограничением этого метода является его ограниченная способность захватывать дальние зависимости или глобальные свойства графа, которые важны для таких задач, как сопоставление или выравнивание графов.

##### Global Node Positional Encodings

The concept of global node positional encodings is inspired by the use of absolute positional encodings in NLP. In NLP, as mentioned previously, every token receives a feature vector indicating its position within a sequence. Extending this idea to graph transformers, each node can be assigned a feature vector representing its position within the embedding space of the graph. The objective of this encoding technique is to encapsulate the overall geometry and spectrum of graphs, thereby unveiling its intrinsic properties and characteristics.

One method for obtaining global node positional encodings is to leverage eigenvectors or eigenvalues of a matrix representation, such as adjacency matrix or Laplacian matrix. For instance, if a node's coordinates lie within the first k eigenvectors of graph Laplacian, its global node positional encoding can be represented by the coordinate vector:

\[
\mathbf{p}_i = [u_{i1}, u_{i2}, \ldots, u_{ik}]
\]

where \( u_{ij} \) is j-th component of i-th eigenvector of the graph Laplacian matrix. One alternative approach to incorporating global node positional encodings is by utilizing diffusion or random walk techniques, such as personalized PageRank or heat kernel. For example, if a node possesses a probability distribution over all other nodes in the graph, following a random walk, its global node positional encoding can be represented by this probability vector:

\[
\mathbf{p}_i = [\pi_{i1}, \pi_{i2}, \ldots, \pi_{iN}]
\]

where \( \pi_{ij} \) is the probability of reaching node \( v_j \) from node \( v_i \) after performing a random walk on the graph.

**Short description:** Explanation of global node positional encodings in graph transformers, including eigenvector-based and random walk approaches.

**Russian translation:**
##### Глобальные позиционные кодировки узлов

Концепция глобальных позиционных кодировок узлов вдохновлена использованием абсолютных позиционных кодировок в NLP. В NLP, как упоминалось ранее, каждый токен получает вектор признаков, указывающий его позицию в последовательности. Расширяя эту идею на графовые трансформеры, каждому узлу может быть присвоен вектор признаков, представляющий его позицию в пространстве встраивания графа. Цель этой техники кодирования заключается в инкапсуляции общей геометрии и спектра графов, тем самым раскрывая их внутренние свойства и характеристики.

Один из методов получения глобальных позиционных кодировок узлов заключается в использовании собственных векторов или собственных значений матричного представления, такого как матрица смежности или лапласиан. Например, если координаты узла лежат в пределах первых k собственных векторов графового лапласиана, его глобальная позиционная кодировка может быть представлена вектором координат:

\[
\mathbf{p}_i = [u_{i1}, u_{i2}, \ldots, u_{ik}]
\]

где \( u_{ij} \) - это j-я компонента i-го собственного вектора матрицы лапласиана графа. Альтернативный подход к включению глобальных позиционных кодировок узлов заключается в использовании методов диффузии или случайных блужданий, таких как персонализированный PageRank или тепловое ядро. Например, если узел имеет распределение вероятностей по всем другим узлам в графе после случайного блуждания, его глобальная позиционная кодировка может быть представлена этим вектором вероятностей:

\[
\mathbf{p}_i = [\pi_{i1}, \pi_{i2}, \ldots, \pi_{iN}]
\]

где \( \pi_{ij} \) - это вероятность достижения узла \( v_j \) из узла \( v_i \) после выполнения случайного блуждания по графу.


A more prevalent approach to implementing global node positional encodings is to utilize graph embedding or dimensionality reduction techniques that map nodes to a lower-dimensional space while maintaining a sense of similarity or distance. For instance, if a node possesses coordinates in a space derived from applying multi-dimensional scaling or graph neural networks to the graph, its global node positional encoding can be represented by that coordinate vector:

\[
p_i = [y_{i1}, y_{i2}, ..., y_{ik}]
\]

where \( y_{ij} \) is j-th component of i-th node embedding in a k-dimensional space, which can be obtained by minimizing the objective function that preserves graph structure:

\[
\min_Y \sum_{i,j=1}^N w_{ij}\|y_i - y_j\|^2
\]

Here, \( w_{ij} \) is a weight matrix that reflects the similarity or distance between node \( v_i \) and node \( v_j \), \( Y \) is the matrix of node embeddings.

The primary aim of global node positional encodings is to improve the representation of node attributes in graph transformers by incorporating geometric and spectral information from graphs. This encoding method offers the advantage of capturing long-range dependencies and overall graph characteristics, benefiting tasks like graph matching and alignment. However, a drawback of this encoding approach is that it could undermine the sparsity and locality of graph structures, potentially impacting efficiency and interpretability.

Короткое описание: В этом разделе описывается подход к глобальным позиционным кодировкам узлов с использованием графовых эмбеддингов, который позволяет сохранять сходство и расстояния между узлами в пространстве меньшей размерности.

Более распространенный подход к реализации глобальных позиционных кодировок узлов заключается в использовании методов графовых эмбеддингов или уменьшения размерности, которые отображают узлы в пространство меньшей размерности, сохраняя при этом представление о сходстве или расстоянии. Например, если узел имеет координаты в пространстве, полученном путем применения многомерного масштабирования или графовых нейронных сетей к графу, его глобальная позиционная кодировка может быть представлена этим вектором координат:

\[
p_i = [y_{i1}, y_{i2}, ..., y_{ik}]
\]

где \( y_{ij} \) - j-я компонента эмбеддинга i-го узла в k-мерном пространстве, которая может быть получена путем минимизации целевой функции, сохраняющей структуру графа:

\[
\min_Y \sum_{i,j=1}^N w_{ij}\|y_i - y_j\|^2
\]

Здесь \( w_{ij} \) - матрица весов, отражающая сходство или расстояние между узлами \( v_i \) и \( v_j \), \( Y \) - матрица эмбеддингов узлов.

Основная цель глобальных позиционных кодировок узлов - улучшить представление атрибутов узлов в графовых трансформерах путем включения геометрической и спектральной информации из графов. Этот метод кодирования имеет преимущество в виде улавливания долгосрочных зависимостей и общих характеристик графа, что полезно для таких задач, как сопоставление и выравнивание графов. Однако недостатком этого подхода к кодированию является то, что он может подорвать разреженность и локальность структур графа, потенциально влияя на эффективность и интерпретируемость.

#### 2) Edge Structural Bias

In the realm of graph transformers, edge structural bias is crucial for extracting and understanding complex information within graph structure. Edge structural bias is versatile and can represent various aspects of graph structure, including node distances, edge types, edge directions and local sub-structures. Empirical evidence has shown that edge structural encodings can improve the effectiveness of graph transformers.

##### Local Edge Structural Encodings

Local edge structural encodings capture the local structure of a graph by encoding relative position or distance between two nodes. These encodings borrow ideas from relative positional encodings used in NLP and CV, where they are used for modeling sequential or spatial order of tokens or pixels. However, in the context of graphs, the concept of relative position or distance between nodes becomes ambiguous due to the presence of multiple connecting paths with varying lengths or weights. Consequently, the scientific community has proposed various methods to define and encode this information specifically for graph structures.

GraphiT introduces local edge structural encodings to graph transformers. It promotes the use of positive definite kernels on graphs to measure node similarity considering the shortest path distance between them. The kernel function is defined as:

\[
k(u, v) = \exp(-\alpha d(u, v))
\]

where \( u \) and \( v \) are two nodes in a graph, \( d(u,v) \) is their shortest path distance and \( \alpha \) is a hyperparameter that controls decay rate. The kernel function is then used to modify the self-attention score between two nodes as:

\[
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{QK}^T}{\sqrt{d_k}} + k(\mathbf{Q},\mathbf{K})\right) \mathbf{V}
\]

where \( k(\mathbf{Q},\mathbf{K}) \) is the matrix of kernel values computed for each pair of nodes. EdgeBERT proposes using edge features as additional input tokens for graph transformers. These edge features are obtained by applying a learnable function to the source and target node features of each edge. The resulting edge features are then concatenated with node features and fed into a standard transformer encoder.

More recently, the Edge-augmented Graph Transformer (EGT) introduces residual edge channels as a mechanism to directly process and output both structural and node information. The residual edge channels are matrices that store edge information for each pair of nodes. They are initialized with either an adjacency matrix or the shortest path matrix and updated at each transformer layer by applying residual connections. These channels are then used to adjust the self-attention score between two nodes:

\[
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{R}_e) = \text{softmax}\left(\frac{\mathbf{QK}^T}{\sqrt{d_k}} + \mathbf{R}_e\right) \mathbf{V}
\]

where \( \mathbf{R}_e \) is the residual edge channel matrix.

**Short description:** Explanation of edge structural bias and local edge structural encodings in graph transformers, including various approaches like GraphiT, EdgeBERT, and EGT.

**Russian translation:**

#### 2) Структурный bias рёбер

В области графовых трансформеров структурный bias рёбер имеет решающее значение для извлечения и понимания сложной информации в структуре графа. Структурный bias рёбер универсален и может представлять различные аспекты структуры графа, включая расстояния между узлами, типы рёбер, направления рёбер и локальные подструктуры. Эмпирические данные показали, что структурные кодировки рёбер могут повысить эффективность графовых трансформеров.

##### Локальные структурные кодировки рёбер

Локальные структурные кодировки рёбер захватывают локальную структуру графа, кодируя относительную позицию или расстояние между двумя узлами. Эти кодировки заимствуют идеи из относительных позиционных кодировок, используемых в NLP и CV, где они применяются для моделирования последовательного или пространственного порядка токенов или пикселей. Однако в контексте графов концепция относительной позиции или расстояния между узлами становится неоднозначной из-за наличия множества соединяющих путей с различными длинами или весами. Следовательно, научное сообщество предложило различные методы для определения и кодирования этой информации специально для графовых структур.

GraphiT вводит локальные структурные кодировки рёбер в графовые трансформеры. Он продвигает использование положительно определенных ядер на графах для измерения сходства узлов с учетом кратчайшего расстояния пути между ними. Ядерная функция определяется как:

\[
k(u, v) = \exp(-\alpha d(u, v))
\]

где \( u \) и \( v \) - два узла в графе, \( d(u,v) \) - их кратчайшее расстояние пути, а \( \alpha \) - гиперпараметр, контролирующий скорость затухания. Ядерная функция затем используется для модификации оценки самовнимания между двумя узлами:

\[
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{QK}^T}{\sqrt{d_k}} + k(\mathbf{Q},\mathbf{K})\right) \mathbf{V}
\]

где \( k(\mathbf{Q},\mathbf{K}) \) - матрица значений ядра, вычисленных для каждой пары узлов. EdgeBERT предлагает использовать признаки рёбер в качестве дополнительных входных токенов для графовых трансформеров. Эти признаки рёбер получаются путем применения обучаемой функции к признакам исходного и целевого узлов каждого ребра. Полученные признаки рёбер затем объединяются с признаками узлов и подаются в стандартный кодировщик трансформера.

Совсем недавно Edge-augmented Graph Transformer (EGT) ввел остаточные каналы рёбер как механизм для прямой обработки и вывода как структурной, так и узловой информации. Остаточные каналы рёбер - это матрицы, которые хранят информацию о рёбрах для каждой пары узлов. Они инициализируются либо матрицей смежности, либо матрицей кратчайших путей и обновляются на каждом слое трансформера путем применения остаточных связей. Эти каналы затем используются для корректировки оценки самовнимания между двумя узлами:

\[
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{R}_e) = \text{softmax}\left(\frac{\mathbf{QK}^T}{\sqrt{d_k}} + \mathbf{R}_e\right) \mathbf{V}
\]

где \( \mathbf{R}_e \) - матрица остаточных каналов рёбер.

Although local edge structural encodings can capture detailed structural information, they may have limitations in capturing overall structural information. This can lead to increased computational complexity or memory usage as pairwise information needs to be computed and stored. Additionally, the effectiveness of these encodings may vary depending on the selection and optimization of encoding or kernel function for different graphs and tasks.

##### Global Edge Structural Encodings

Global edge structural encodings aim to capture the overall structure of a graph. Unlike NLP and CV domains, the exact position of a node in graphs is not well-defined because there is no natural order or coordinate system. Several approaches have been suggested to tackle this issue.

GPT-GNN is an early work that utilizes graph pooling and unpooling operations to encode the hierarchical structure of a graph. It reduces graph size by grouping similar nodes and then restoring the original size by assigning cluster features to individual nodes. This approach results in a multiscale representation of graphs and has demonstrated enhanced performance on diverse tasks. Graphormer uses spectral graph theory to encode global structure. It uses eigenvectors of normalized Laplacian matrix as global positional encodings for nodes. This method can capture global spectral features (e.g., connectivity, centrality and community structure). Park et al. extended Graphormer by using singular value decomposition (SVD) to encode global structure. They utilized the left singular matrix of the adjacency matrix as global positional encodings for nodes. This approach can handle both symmetric and asymmetric matrices.

Global edge structural encodings excel at capturing coarse-grained structural information at the graph level, benefiting tasks that require global understanding. However, they may struggle with capturing fine-grained node-level information and can lose data or introduce noise during encoding. In addition, their effectiveness may depend on the choice of encoding technique and matrix representations.

#### 3) Message-passing Bias

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

**Short description:** Explanation of global edge structural encodings and message-passing bias in graph transformers, including various approaches and their mathematical formulations.

**Russian translation:**

Хотя локальные структурные кодировки рёбер могут захватывать детальную структурную информацию, они могут иметь ограничения в захвате общей структурной информации. Это может привести к увеличению вычислительной сложности или использования памяти, так как попарная информация должна быть вычислена и сохранена. Кроме того, эффективность этих кодировок может варьироваться в зависимости от выбора и оптимизации функции кодирования или ядра для различных графов и задач.

#### Глобальные структурные кодировки рёбер

Глобальные структурные кодировки рёбер направлены на захват общей структуры графа. В отличие от областей NLP и CV, точная позиция узла в графах не имеет четкого определения, поскольку нет естественного порядка или системы координат. Было предложено несколько подходов для решения этой проблемы.

GPT-GNN - это ранняя работа, которая использует операции пулинга и анпулинга графа для кодирования иерархической структуры графа. Она уменьшает размер графа, группируя похожие узлы, а затем восстанавливает исходный размер, назначая признаки кластера отдельным узлам. Этот подход приводит к многомасштабному представлению графов и продемонстрировал улучшенную производительность на различных задачах. Graphormer использует спектральную теорию графов для кодирования глобальной структуры. Он использует собственные векторы нормализованной матрицы Лапласа в качестве глобальных позиционных кодировок для узлов. Этот метод может захватывать глобальные спектральные признаки (например, связность, центральность и структуру сообществ). Park et al. расширили Graphormer, используя сингулярное разложение (SVD) для кодирования глобальной структуры. Они использовали левую сингулярную матрицу матрицы смежности в качестве глобальных позиционных кодировок для узлов. Этот подход может обрабатывать как симметричные, так и асимметричные матрицы.

Глобальные структурные кодировки рёбер преуспевают в захвате грубозернистой структурной информации на уровне графа, что полезно для задач, требующих глобального понимания. Однако они могут бороться с захватом мелкозернистой информации на уровне узлов и могут терять данные или вносить шум во время кодирования. Кроме того, их эффективность может зависеть от выбора техники кодирования и матричных представлений.

### 3) Message-passing Bias

Message-passing bias (смещение передачи сообщений) является важным индуктивным смещением для графовых трансформеров, облегчающим обучение на основе локальной структуры графов. Это смещение позволяет графовым трансформерам обмениваться информацией между узлами и рёбрами, тем самым захватывая зависимости и взаимодействия между элементами графа. Более того, смещение передачи сообщений помогает графовым трансформерам преодолевать определенные ограничения стандартной архитектуры трансформера, такие как квадратичная сложность самовнимания, отсутствие позиционной информации и проблемы, связанные с обработкой разреженных графов.

Формально смещение передачи сообщений может быть выражено следующим образом:

\[
h_v^{(t+1)} = f(h_v^{(t)}, \{h_u^{(t)} : u \in N(v)\}, \{e_{uv} : u \in N(v)\})
\]

где:
- \( h_v^{(t)} \) - скрытое состояние узла v на временном шаге t
- \( N(v) \) - окрестность узла v
- \( e_{uv} \) - признак ребра между узлами u и v
- \( f \) - функция передачи сообщений


Here, \( h^{(0)}_v = x_v \), where \( x_v \) is the input feature of node \( v \), and SelfAttention is a function that performs self-attention over all nodes. The limitation of preprocessing is that it applies message-passing only once before self-attention layers. This approach may not fully capture intricate interactions between nodes and edges on various scales. Additionally, the preprocessing step may introduce redundancy and inconsistency between the message-passing module and the self-attention layer as they both serve similar functions of aggregating information from neighboring elements.

**Interleaving.** Interleaving refers to the technique employed in graph transformer architecture that involves alternating message-passing operations and self-attention layers. The objective of this technique is to achieve a balance between local and global information processing, thereby enabling multi-hop reasoning over graphs. By integrating message-passing modules into core components of graph transformers, interleaving enhances their expressive power and flexibility.

**Short description:** Explanation of message-passing bias in graph transformers, including interleaving approach.

**Russian translation:**

Здесь \( h^{(0)}_v = x_v \), где \( x_v \) - это входной признак узла \( v \), а SelfAttention - это функция, которая выполняет самовнимание по всем узлам. Ограничение предварительной обработки заключается в том, что она применяет передачу сообщений только один раз перед слоями самовнимания. Этот подход может не полностью захватывать сложные взаимодействия между узлами и рёбрами на различных масштабах. Кроме того, этап предварительной обработки может вводить избыточность и несоответствие между модулем передачи сообщений и слоем самовнимания, так как они оба выполняют аналогичные функции агрегации информации от соседних элементов.

**Чередование.** Чередование относится к технике, используемой в архитектуре графовых трансформеров, которая включает чередование операций передачи сообщений и слоёв самовнимания. Цель этой техники - достичь баланса между обработкой локальной и глобальной информации, что позволяет выполнять многопереходное рассуждение по графам. Интегрируя модули передачи сообщений в основные компоненты графовых трансформеров, чередование увеличивает их выразительную мощность и гибкость.

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







