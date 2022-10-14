Awesome-RSPapers
================

Included Conferences: CIKM 2021, CIKM 2022, WSDM 2022



-   [Task](#task)
    -   [Collaborative Filtering](#collaborative-filtering)
    -   [Sequential/Session-based
        Recommendations](#sequentialsession-based-recommendations)
    -   [Knowledge-aware
        Recommendations](#knowledge-aware-recommendations)
    -   [Feature Interactions](#feature-interactions)
    -   [Conversational Recommender
        System](#conversational-recommender-system)
    -   [Social Recommendations](#social-recommendations)
    -   [News Recommendation](#news-recommendation)
    -   [Text-aware Recommendations](#text-aware-recommendations)
    -   [POI](#poi)
    -   [Online Recommendations](#online-recommendations)
    -   [Group Recommendation](#group-recommendation)
    -   [Multi-task/Multi-behavior/Cross-domain
        Recommendations](#multi-taskmulti-behaviorcross-domain-recommendations)
    -   [Other Task](#other-task)
-   [Topic](#topic)
    -   [Debias in Recommender
        System](#debias-in-recommender-system)
    -   [Fairness in Recommender
        System](#fairness-in-recommender-system)
    -   [Attack in Recommender
        System](#attack-in-recommender-system)
    -   [Explanation in Recommender
        System](#explanation-in-recommender-system)
    -   [Long-tail/Cold-start in
        Recommendations](#long-tailcold-start-in-recommendations)
    -   [Diversity in Recommendations](#diversity-in-recommendations)
    -   [Evaluation](#evaluation)
-   [Technique](#technique)
    -   [Pre-training in Recommender
        System](#pre-training-in-recommender-system)
    -   [Reinforcement Learning in
        Recommendations](#reinforcement-learning-in-recommendations)
    -   [Knowledge Distillation in
        Recommendations](#knowledge-distillation-in-recommendations)
    -   [NAS in Recommendations](#nas-in-recommendations)
    -   [Federated Learning in
        Recommendations](#federated-learning-in-recommendations)
    -   [GNN in Recommendations](#gnn-in-Recommendations)
    -   [Transformer in Recommendations](#transformer-in-Recommendations)
    -   [Contrastive Learning in Recommendations](#contrastive-learning-in-Recommendations)
    -   [Multi-Modality in Recommendations](#multi-modality-in-Recommendations)
    -   [Data Augmentation in Recommendations](#data-augmentation-in-Recommendations)
    -   [Meta Learning in Recommendations](#meta-learning-in-Recommendations)
    -   [Few-Shot Learning in Recommendations](#few-shot-learning-in-Recommendations)
-   [Analysis](#analysis)
-   [Other](#other)

Task
----

### Collaborative Filtering

-   SimpleX: A Simple and Strong Baseline for Collaborative Filtering. CIKM 2021【将Cosine Contrastive Loss引入协同过滤】
-   Incremental Graph Convolutional Network for Collaborative Filtering. CIKM 2021【增量图卷积神经网络用于协同过滤】
-   LT-OCF: Learnable-Time ODE-based Collaborative Filtering. CIKM 2021【Learnable-Time CF】
-   CausCF: Causal Collaborative Filtering for Recommendation Effect Estimation. CIKM 2021【applied paper，因果关系协同过滤用于推荐效果评估】
-   Vector-Quantized Autoencoder With Copula for Collaborative Filtering. CIKM 2021【short paper，用于协同过滤的矢量量化自动编码器】
-   Anchor-based Collaborative Filtering for Recommender Systems. CIKM 2021【short paper，Anchor-based推荐系统协同过滤】
-   VAE++: Variational AutoEncoder for Heterogeneous One-Class Collaborative Filtering. WSDM 2022【异构单类协同过滤的变分自动编码器】
-   Geometric Inductive Matrix Completion:
    A Hyperbolic Approach with Unified Message Passing. WSDM 2022 【具有统一消息传递的双曲线方法】
-   Asymmetrical Context-aware Modulation for Collaborative Filtering Recommendation. CIKM 2022 【用于协同过滤推荐的非对称上下文感知调制】
-   Dynamic Causal Collaborative Filtering. CIKM 2022 【动态因果协同过滤】


### Sequential/Session-based Recommendations

-   Seq2Bubbles: Region-Based Embedding Learning for User Behaviors in Sequential Recommenders. CIKM 2021【序列推荐中基于区域的用户行为Embedding学习】
-   Enhancing User Interest Modeling with Knowledge-Enriched Itemsets for Sequential Recommendation. CIKM 2021【序列推荐中使用物品集增强用户兴趣建模】
-   Continuous-Time Sequential Recommendation with Temporal Graph Collaborative Transformer. CIKM 2021【将时序图协同Transformer用于连续时间序列推荐】
-   Extracting Attentive Social Temporal Excitation for Sequential Recommendation. CIKM 2021【提取时序激励用于序列推荐】
-   Hyperbolic Hypergraphs for Sequential Recommendation. CIKM 2021【使用双曲超图进行序列推荐】
-   Learning Dual Dynamic Representations on Time-Sliced User-Item Interaction Graphs for Sequential Recommendation. CIKM 2021【用于序列推荐的在时间片用户物品交互图上的对偶动态表示】
-   Lightweight Self-Attentive Sequential Recommendation. CIKM 2021【使用CNN捕获局部特征，使用Self-Attention捕获全局特征】
-   What is Next when Sequential Prediction Meets Implicitly Hard Interaction? CIKM 2021【序列预测与交互】
-   Modeling Sequences as Distributions with Uncertainty for Sequential Recommendation. CIKM 2021【short paper，序列建模】
-   Locker: Locally Constrained Self-Attentive Sequential Recommendation. CIKM 2021【short paper，局部约束的自注意力序列推荐】
-   CBML: A Cluster-based Meta-learning Model for Session-based Recommendation. CIKM 2021【用于会话推荐的基于聚类的元学习】
-   Self-Supervised Graph Co-Training for Session-based Recommendation. CIKM 2021【用于会话推荐的自监督图协同训练】
-   S-Walk: Accurate and Scalable Session-based Recommendation with Random Walks. WSDM 2022【具有随机游走的准确且可扩展的基于会话的推荐】
-   Learning Multi-granularity Consecutive User Intent Unit for Session-based Recommendation. WSDM 2022【基于会话的推荐学习多粒度连续用户意图单元】
-   Beyond Learning from Next Item: Sequential Recommendation via Personalized Interest Sustainability. CIKM 2022 【基于个性化兴趣可持续性的序列推荐】
-   FwSeqBlock: A Field-wise Approach for Modeling Behavior Representation in Sequential Recommendation. CIKM 2022 【建模行为表示】
-   Dually Enhanced Propensity Score Estimation in Sequential Recommendation. CIKM 2022 【双重增强倾向得分估计】
-   Hierarchical Item Inconsistency Signal learning for Sequence Denoising in Sequential Recommendation. CIKM 2022 【序列推荐中序列去噪的分层项目不一致信号学习】
-   ContrastVAE: Contrastive Variational AutoEncoder for Sequential Recommendation. CIKM 2022 【用于序列推荐的对比变分自动编码器】
-   Disentangling Past-Future Modeling in Sequential Recommendation via Dual Networks. CIKM 2022 【通过双网络解耦序列推荐中的过去未来】
-   Evolutionary Preference Learning via Graph Nested GRU ODE for Session-based Recommendation. CIKM 2022 【通过图嵌套 GRU ODE 进行进化偏好学习】
-   Spatiotemporal-aware Session-based Recommendation with Graph Neural Networks. CIKM 2022 【使用图神经网络的时空感知基于会话的推荐】
-   Time Lag Aware Sequential Recommendation. CIKM 2022 【延时感知序列推荐】


### Knowledge-aware Recommendations

-   A Knowledge-Aware Recommender with Attention-Enhanced Dynamic Convolutional Network. CIKM 2021【动态卷积用于知识感知的推荐】
-   Entity-aware Collaborative Relation Network with Knowledge Graph for Recommendation. CIKM 2021【short paper，KG+RS】
-   Conditional Graph Attention Networks for Distilling and Refining Knowledge Graphs in Recommendation. CIKM 2021【GNN+KG+RS】
-   Tracking Semantic Evolutionary Changes in Large-Scale Ontological Knowledge Bases. CIKM 2021【大规模本体知识库中语义演化的跟踪】
-   Cycle or Minkowski: Which is More Appropriate for Knowledge Gragh Embedding? CIKM 2021【KG Embedding】
-   HopfE: Knowledge Graph Representation Learning using Inverse Hopf Fibrations. CIKM 2021【知识图谱表示学习】
-   Automated Query Graph Generation for Querying Knowledge Graphs. CIKM 2021【用于查询知识图谱的自动查询图生成】
-   A Lightweight Knowledge Graph Embedding Framework for Efficient Inference and Storage. CIKM 2021【轻量化KG Embedding】
-   Predicting Instance Type Assertions in Knowledge Graphs Using Stochastic Neural Networks. CIKM 2021【知识图谱中的实例类型断言预测】
-   When Hardness Makes a Difference: Multi-Hop Knowledge Graph Reasoning over Few-Shot Relations. CIKM 2021【小样本关系上的知识图谱多跳推理】
-   Query Reformulation for Descriptive Queries of Jargon Words Using a Knowledge Graph based on a Dictionary. CIKM 2021【使用基于字典的知识图谱进行查询重构】
-   Computing and Maintaining Provenance of Query Result Probabilities in Uncertain Knowledge Graphs. CIKM 2021【不确定知识图谱中计算和维护查询结果概率】
-   REFORM: Error-Aware Few-Shot Knowledge Graph Completion. CIKM 2021【错误感知的小样本知识图谱补全】
-   DisenKGAT: Knowledge Graph Embedding with Disentangled Graph Attention Network. CIKM 2021【KG Embedding+GNN】
-   Complex Temporal Question Answering on Knowledge Graphs. CIKM 2021【QA+KG】
-   Mixed Attention Transformer for Leveraging Word-Level Knowledge to Neural Cross-Lingual Information Retrieval. CIKM 2021【Transformer+IR】
-   Knowledge Graph Representation Learning as Groupoid: Unifying TransE, RotatE, QuatE, ComplEx. CIKM 2021【知识图谱表示学习】
-   DataType-Aware Knowledge Graph Representation Learning in Hyperbolic Space. CIKM 2021【双曲空间中基于数据类型的知识图谱表示学习】
-   Evidential Relational-Graph Convolutional Networks for Entity Classification in Knowledge Graphs. CIKM 2021【short paper，GNN+KG】
-   Modeling Scale-free Graphs with Hyperbolic Geometry for Knowledge-aware Recommendation. WSDM 2022【使用双曲几何建模无标度图以进行知识感知推荐】
-   Causal Relationship over Knowledge Graphs. CIKM 2022 【知识图谱上的因果关系】

### Feature Interactions

-   Multi-task Learning for Bias-Free Joint CTR Prediction and Market Price Modeling in Online Advertising. CIKM 2021【在线广告无偏差联合CTR预估和市场价格建模的多任务学习】
-   Enhancing Explicit and Implicit Feature Interactions via Information Sharing for Parallel Deep CTR Models. CIKM 2021【applied paper，用于并行 CTR 的显式和隐式特征交互增强】
-   TSI: An Ad Text Strength Indicator using Text-to-CTR and Semantic-Ad-Similarity. CIKM 2021【applied paper，使用 Text-to-CTR 和 Semantic-Ad-Similarity 的广告文本强度指标】
-   One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction. CIKM 2021【applied paper，用于多领域CTR预估的自适应推荐】
-   Efficient Learning to Learn a Robust CTR Model for Web-scale Online Sponsored Search Advertising. CIKM 2021【applied paper，用于在线搜索广告的CTR模型】
-   AutoIAS: Automatic Integrated Architecture Searcher for Click-Trough Rate Prediction. CIKM 2021【CTR预估的自动集成搜索架构】
-   Click-Through Rate Prediction with Multi-Modal Hypergraphs. CIKM 2021【使用多模态超图的点击率预测】
-   Open Benchmarking for Click-Through Rate Prediction. CIKM 2021【开源CTR预估Benchmark】
-   Disentangled Self-Attentive Neural Networks for Click-Through Rate Prediction. CIKM 2021【short paper，用于CTR预估的自注意力网络】
-   AutoHERI: Automated Hierarchical Representation Integration for Post-Click Conversion Rate Estimation. CIKM 2021【short paper，用于点击后转换率估计的分层表示学习】
-   Sequential Modeling with Multiple Attributes for Watchlist Recommendation in E-Commerce. WSDM 2022【电子商务中观察列表推荐的多属性序列建模】
-   Modeling Users’ Contextualized Page-wise Feedback for Click-Through Rate Prediction in E-commerce Search. WSDM 2022【电子商务搜索中点击率预测的用户情境化页面反馈建模】
-   Learning-To-Ensemble by Contextual Rank Aggregation in E-Commerce. WSDM 2022【通过电子商务中的上下文排名聚合学习集成】
-   CAN: Feature Co-Action Network for Click-Through Rate Prediction. WSDM 2022【用于点击率预测的特征协同网络】
-   Triangle Graph Interest Network for Click-through Rate Prediction. WSDM 2022【用于点击率预测的三角图兴趣网络】
-   OptEmbed: Learning Optimal Embedding Table for Click-through Rate Prediction. CIKM 2022 【点击率预测的最优嵌入表】
-   Multi-Interest Refinement by Collaborative Attributes Modeling for Click-Through Rate Prediction. CIKM 2022 【通过协作属性建模进行多兴趣细化的点击率预测】
-   GIFT: Graph-guIded Feature Transfer for Cold-Start Video Click-Through Rate Prediction. CIKM 2022 【GIFT：用于冷启动视频点击率预测的图引导特征迁移】
-   Graph Based Long-Term And Short-Term Interest Model for Click-Through Rate Prediction. CIKM 2022 【用于点击率预测的基于图的长期和短期兴趣模型】
-   Hierarchically Fusing Long and Short-Term User Interests for Click-Through Rate Prediction in Product Search. CIKM 2022 【分层融合长期和短期用户兴趣以进行产品搜索中的点击率预测】
-   Sparse Attentive Memory Network for Click-through Rate Prediction with Long Sequences. CIKM 2022 【用于长序列点击率预测的稀疏注意力记忆网络】
-   Towards Understanding the Overfitting Phenomenon of Deep Click-Through Rate Models. CIKM 2022 【了解深度点击率模型的过拟合现象】

### Conversational Recommender System

-   Popcorn: Human-in-the-loop Popularity Debiasing in Conversational Recommender Systems. CIKM 2021【采用人在回路方式进行对话推荐系统的流行度去偏】
-   A Neural Conversation Generation Model via Equivalent Shared Memory Investigation. CIKM 2021【对话生成】
-   C2-CRS: Coarse-to-Fine Contrastive Learning for Conversational Recommender System. WSDM 2022【对话式推荐系统的粗到细对比学习】
-   Rethinking Conversational Recommendations: Is Decision Tree All You Need? CIKM 2022 【重新思考对话推荐：决策树是否就是我们需要的？】
-   Two-level Graph Path Reasoning for Conversational Recommendation with User Realistic Preference. CIKM 2022 【具有用户现实偏好的会话推荐的两级图路径推理】

### Social Recommendations

-   Social Recommendation with Self-Supervised Metagraph Informax Network. CIKM 2021【使用自监督元图网络的社交推荐】
-   Ranking Friend Stories on Social Platforms with Edge-Contextual Local Graph Convolutions. WSDM 2022 【基于图卷积神经网络的社交排序】

### News Recommendation

-   WG4Rec: Modeling Textual Content with Word Graph for News Recommendation. CIKM 2021【使用Word Graph为新闻推荐建模文本内容】
-   Popularity-Enhanced News Recommendation with Multi-View Interest Representation. CIKM 2021【多视角兴趣学习的流行度增强的新闻推荐】
-   Prioritizing Original News on Facebook. CIKM 2021【applied paper，原创新闻优先级排序】
-   DeepVT: Deep View-Temporal Interaction Network for News Recommendatio. CIKM 2022 【新闻推荐的深度视图-时间交互网络】
-   Generative Adversarial Zero-Shot Learning for Cold-Start News Recommendation. CIKM 2022 【冷启动新闻推荐的生成对抗零样本学习】


### Text-aware Recommendations

-   Counterfactual Review-based Recommendation. CIKM 2021【基于评论的反事实推荐】
-   Review-Aware Neural Recommendation with Cross-Modality Mutual Attention. CIKM 2021【short paper，文本+RS+跨模态】

### POI

-   Answering POI-recommendation Questions using Tourism Reviews. CIKM 2021【使用旅游者评论回答POI问题】
-   SNPR: A Serendipity-Oriented Next POI Recommendation Model. CIKM 2021【面向偶然性的POI推荐】
-   ST-PIL: Spatial-Temporal Periodic Interest Learning for Next Point-of-Interest Recommendation. CIKM 2021【short paper，用于POI推荐的时空周期兴趣学习】

### Online Recommendations

-   Generative Inverse Deep Reinforcement Learning for Online Recommendation. CIKM 2021【用于在线推荐的生成式逆强化学习】
-   Long Short-Term Temporal Meta-learning in Online Recommendation. WSDM 2022【在线推荐中的长短期时间元学习】
-   A Cooperative-Competitive Multi-Agent Framework for Auto-bidding in Online Advertising. WSDM 2022【一种用于在线广告自动竞价的合作竞争多代理框架】
-   Knowledge Extraction and Plugging for Online Recommendation. CIKM 2022 【在线推荐的知识抽取与插入】
-   Real-time Short Video Recommendation on Mobile Devices. CIKM 2022 【移动端实时短视频推荐】
-   SASNet: Stage-aware sequential matching for online travel recommendation. CIKM 2022 【在线旅游推荐的阶段感知序列匹配】

### Group Recommendation

-   Double-Scale Self-Supervised Hypergraph Learning for Group Recommendation. CIKM 2021【用于群组推荐的自监督超图学习】
-   DeepGroup: Group Recommendation with Implicit Feedback. CIKM 2021【short paper，隐式反馈的群组推荐】

### Multi-task/Multi-behavior/Cross-domain Recommendations

-   Cross-Market Product Recommendation. CIKM 2021【跨市场产品推荐】
-   Expanding Relationship for Cross Domain Recommendation. CIKM 2021【扩展跨领域推荐的关系】
-   Learning Representations of Inactive Users: A Cross Domain Approach with Graph Neural Networks. CIKM 2021【short paper，跨领域方法结合图神经网络用于学习非活跃用户表示】
-   Low-dimensional Alignment for Cross-Domain Recommendation. CIKM 2021【short paper，跨领域推荐的低维对齐】
-   Multi-Sparse-Domain Collaborative Recommendation via Enhanced Comprehensive Aspect Preference Learning. WSDM 2022【通过增强的综合方面偏好学习的多稀疏域协作推荐】
-   Leaving No One Behind: A Multi-Scenario Multi-Task Meta Learning Approach for Advertiser Modeling. WSDM 2022【一种用于广告商建模的多场景多任务元学习方法】
-   RecGURU: Adversarial Learning of Generalized User Representations for Cross-Domain Recommendation. WSDM 2022【用于跨域推荐的广义用户表示的对抗性学习】
-   Personalized Transfer of User Preferences for Cross-domain Recommendation. WSDM 2022【跨域推荐用户偏好的个性化传输】
-   Adaptive Domain Interest Network for Multi-domain Recommendation. CIKM 2022 【多域推荐的自适应域兴趣网络】
-   Multi-Scale User Behavior Network for Entire Space Multi-Task Learning. CIKM 2022 【全空间多任务学习的多尺度用户行为网络】
-   Gromov-Wasserstein Guided Representation Learning for Cross-Domain Recommendation. CIKM 2022 【跨域推荐表示学习】
-   Contrastive Cross-Domain Sequential Recommendation. CIKM 2022 【对比跨域序列推荐】
-   Cross-domain Recommendation via Adversarial Adaptation. CIKM 2022【通过对抗性适应进行跨域推荐】
-   Dual-Task Learning for Multi-Behavior Sequential Recommendation. CIKM 2022 【多行为序列推荐的双任务学习】
-   FedCDR: Federated Cross-Domain Recommendation for Privacy-Preserving Rating Prediction. CIKM 2022 【FedCDR：用于隐私保护评级预测的联合跨域推荐】
-   Leveraging Multiple Types of Domain Knowledge for Safe and Effective Drug Recommendation. CIKM 2022 【利用多种领域知识进行安全有效的药物推荐】
-   Multi-Faceted Hierarchical Multi-Task Learning for Recommender Systems. CIKM 2022 【推荐系统的多方面分层多任务学习】
-   Review-Based Domain Disentanglement without Duplicate Users or Contexts for Cross-Domain Recommendation. CIKM 2022 【没有重复用户或上下文的基于审查的域解耦，用于跨域推荐】
-   Scenario-Adaptive and Self-Supervised Model for Multi-Scenario Personalized Recommendation. CIKM 2022 【多场景个性化推荐的场景自适应自监督模型】

### Other Task

-   Disentangling Preference Representations for Recommendation Critiquing with $\beta$-VAE. CIKM 2021【用于推荐的VAE偏好表示】
-   Top-N Recommendation with Counterfactual User Preference Simulation. CIKM 2021【反事实用户偏好模拟的Top-N推荐】
-   Learning An End-to-End Structure for Retrieval in Large-Scale Recommendations. CIKM 2021【在大规模推荐中学习一个端到端的结构用于检索】
-   USER: A Unified Information Search and Recommendation Model based on Integrated Behavior Sequence. CIKM 2021【基于集成行为序列的统一搜索与推荐模型】
-   Multi-hop Reading on Memory Neural Network with Selective Coverage for Medication Recommendation. CIKM 2021【药物推荐】
-   A Counterfactual Modeling Framework for Churn Prediction. WSDM 2022 【客户流失预测的反事实建模框架】
-   Show Me the Whole World: Towards Entire Item Space Exploration for Interactive Personalized Recommendations. WSDM 2022【面向交互式个性化推荐的整个商品空间探索】
-   Adapting Triplet Importance of Implicit Feedback for Personalized Recommendation. CIKM 2022 【在个性化推荐中调整隐式反馈的三元组重要性】
-   GRP: A Gumbel-based Rating Prediction Framework for Imbalanced Recommendation. CIKM 2022 【基于 Gumbel 的不平衡推荐评级预测框架】
-   Rank List Sensitivity of Recommender Systems to Interaction Perturbations. CIKM 2022 【推荐系统对交互扰动的排名列表敏感性】
-   CROLoss: Towards a Customizable Loss for Retrieval Models in Recommender Systems. CIKM 2022 【推荐系统中检索模型的可定制损失】
-   Towards Principled User-side Recommender Systems. CIKM 2022 【迈向有原则的用户侧推荐系统】
-   A Case Study in Educational Recommenders:Recommending Music Partitures at Tomplay. CIKM 2022 【在 Tomplay 推荐音乐片段】
-   Knowledge Enhanced Multi-Interest Network for the Generation of Recommendation Candidates. CIKM 2022 【用于生成推荐候选的知识增强多兴趣网络】
-   UDM: A Unified Deep Matching Framework in Recommender System. CIKM 2022 【推荐系统中的统一深度匹配框架】
-   User Recommendation in Social Metaverse with VR. CIKM 2022 【VR的用户推荐】


Topic
-----

### Debias in Recommender System

-   CauSeR: Causal Session-based Recommendations for Handling Popularity Bias. CIKM 2021【short paper，用于流行度去偏的因果关系序列推荐】
-   Mixture-Based Correction for Position and Trust Bias in Counterfactual Learning to Rank. CIKM 2021【位置和信任偏差】
-   Unbiased Filtering of Accidental Clicks in Verizon Media Native Advertising. CIKM 2021【applied paper，广告意外点击的无偏过滤】
-   It Is Different When Items Are Older: Debiasing Recommendations When Selection Bias and User Preferences are Dynamic. WSDM 2022【选择偏差和偏好偏差动态变化时的纠偏推荐系统】
-   Fighting Mainstream Bias in Recommender Systems via Local Fine Tuning. WSDM 2022【通过局部微调对抗推荐系统中的主流偏见】
-   Towards Unbiased and Robust Causal Ranking for Recommender Systems. WSDM 2022【推荐系统的无偏和稳健因果排名】
-   Quantifying and Mitigating Popularity Bias in Conversational Recommender Systems. CIKM 2022 【量化和减轻会话推荐系统中的流行度偏差】
-   Learning Unbiased User Behaviors Estimation with Hierarchical Recurrent Model on the Entire Space. CIKM 2022 【分层递归模型学习无偏用户行为估计】
-   Representation Matters When Learning From Biased Feedback in Recommendation. CIKM 2022 【从推荐中的有偏反馈中学习时，表征很重要】

### Fairness in Recommender System

-   SAR-Net: A Scenario-Aware Ranking Network for Personalized Fair Recommendation in Hundreds of Travel Scenarios. CIKM 2021【applied paper，用于个性化公平推荐的场景感知排名网络】
-   Enumerating Fair Packages for Group Recommendations. WSDM 2022【枚举组推荐的公平包】


### Attack in Recommender System

-   PipAttack: Poisoning Federated Recommender Systems for Manipulating Item Promotion. WSDM 2022【用于操纵项目促销的中毒联合推荐系统】

### Explanation in Recommender System

-   Counterfactual Explainable Recommendation. CIKM 2021【反事实可解释推荐】
-   On the Diversity and Explainability of Recommender Systems: A Practical Framework for Enterprise App Recommendation. CIKM 2021【applied paper，推荐系统的多样性和可解释性】
-   You Are What and Where You Are: Graph Enhanced Attention Network for Explainable POI Recommendation. CIKM 2021【applied paper，Attention图神经网络用于可解释推荐】
-   XPL-CF: Explainable Embeddings for Feature-based Collaborative Filtering. CIKM 2021【short paper，可解释Embedding用于基于特征的协同过滤】
-   Grad-SAM: Explaining Transformers via Gradient Self-Attention Maps. CIKM 2021【short paper，通过梯度Self-Attention解释Transformer】
-   Reinforcement Learning over Sentiment-Augmented Knowledge Graphs towards Accurate and Explainable Recommendation. WSDM 2022【对情感增强知识图的强化学习以实现准确和可解释的推荐】
-   Explanation Guided Contrastive Learning for Sequential Recommendation. CIKM 2022 【序列推荐的解释引导对比学习】

### Long-tail/Cold-start in Recommendations

-   CMML: Contextual Modulation Meta Learning for Cold-Start Recommendation. CIKM 2021【元学习+冷启动】
-   Reinforcement Learning to Optimize Lifetime Value in Cold-Start Recommendation. CIKM 2021【增强学习+冷启动】
-   Zero Shot on the Cold-Start Problem: Model-Agnostic Interest Learning for Recommender Systems. CIKM 2021【零样本学习+冷启动】
-   Memory Bank Augmented Long-tail Sequential Recommendation. CIKM 2022 【记忆库增强】

### Diversity in Recommendations

- Choosing the Best of All Worlds: Accurate, Diverse, and Novel Recommendations through Multi-Objective Reinforcement Learning. WSDM 2022【通过多目标强化学习的准确、多样化和新颖的推荐】

### Evaluation

- Evaluating Human-AI Hybrid Conversational Systems with Chatbot Message Suggestions. CIKM 2021【人机混合对话系统评估】
- POSSCORE: A Simple Yet Effective Evaluation of Conversational Search with Part of Speech Labelling. CIKM 2021【使用部分语音标签对会话搜索进行简单有效的评估】
- KuaiRec: A Fully-observed Dataset and Insights for Evaluating Recommender Systems. CIKM 2022 【用于评估推荐系统的完全观察数据集和见解】

### Resource

- DistRDF2ML - Scalable Distributed In-Memory Machine Learning Pipelines for RDF Knowledge Graphs. CIKM 2021 【用于RDF的分布式机器学习pipeline】
- Evaluating Graph Vulnerability and Robustness using TIGER. CIKM 2021 【评估图的脆弱性和鲁棒性】
- PyTorch Geometric Temporal: Spatiotemporal Signal Processing with Neural Machine Learning Models. CIKM 2021 【使用神经机器学习模型进行时空信号处理】
- LC: A Flexible, Extensible Open-Source Toolkit for Model Compression. CIKM 2021 【LC: 用于模型压缩的灵活、可扩展的开源工具包】
- LiteratureQA: A Qestion Answering Corpus with Graph Knowledge on Academic Literature. CIKM 2021 【LiteratureQA: 学术文献中具有图形知识的问答语料库】
- GAKG: A Multimodal Geoscience Academic Knowledge Graph. CIKM 2021 【GAKG：多模态地球科学学术知识图谱】
- TrUMAn: Trope Understanding in Movies and Animations. CIKM 2021 【TrUMAn: 电影和动画中的比喻理解】
- Machamp: A Generalized Entity Matching Benchmark. CIKM 2021 【Machamp：广义实体匹配基准】
- DL-Traff: Survey and Benchmark of Deep Learning Models for Urban Traffic Prediction. CIKM 2021 【DL-Traff：城市交通预测深度学习模型的调查和基准测试】
- TwiBot-20: A Comprehensive Twitter Bot Detection Benchmark. CIKM 2021 【TwiBot-20：全面的 Twitter 机器人检测基准】
- MOOCCubeX: A Large Knowledge-centered Repository for Adaptive Learning in MOOCs. CIKM 2021 【MOOCCubeX：以知识为中心的大型 MOOC 自适应学习存储库】
- ULTRA: An Unbiased Learning To Rank Algorithm Toolbox. CIKM 2021 【ULTRA：一个无偏学习排名算法工具箱】
- VerbCL: A Dataset of Verbatim Quotes for Highlight Extraction in Case Law. CIKM 2021 【VerbCL：判例法中用于突出显示的逐字引用数据集】
- QuaPy: A Python-Based Framework for Quantification. CIKM 2021 【QuaPy：基于 Python 的量化框架】
- SoMeSci— A 5 Star Open Data Gold Standard Knowledge Graph of Software Mentions in Scientific Articles. CIKM 2021 【SomeSci — 科学文章中软件提及的 5 星开放数据黄金标准知识图谱】
- PyTerrier: Declarative Experimentation in Python from BM25 to Dense Retrieval. CIKM 2021 【PyTerrier：从 BM25 到密集检索的 Python 声明性实验】
- ECEdgeNet: A Large Scale Edge Computing Dataset in the Field of E-commerce. CIKM 2021 【ECEdgeNet：电子商务领域的大规模边缘计算数据集】
- Pirá: A Bilingual Portuguese-English Dataset for Question-Answering about the Ocean. CIKM 2021 【Pirá：用于海洋问答的双语葡萄牙语-英语数据集】
- MS MARCO Chameleons: Challenging the MS MARCO Leaderboard with Extremely Obstinate Queries. CIKM 2021 【MS MARCO Chameleons：用极其顽固的查询挑战 MS MARCO 排行榜】
- CoST: An annotated Data Collection for Complex Search. CIKM 2021 【CoST：复杂搜索的注释数据集合】
- librec-auto: A Tool for Recommender Systems Experimentation. CIKM 2021 【librec-auto：推荐系统实验的工具】
- Matches Made in Heaven: Toolkit and Large-Scale Datasets for Supervised Query Reformulation. CIKM 2021 【监督查询重构的工具包和大规模数据集】
- VidLife: A Dataset for Life Event Extraction from Videos. CIKM 2021 【VidLife：从视频中提取生活事件的数据集】
- GeoVectors: A Linked Open Corpus of OpenStreetMap Embeddings on World Scale. CIKM 2021 【GeoVectors：世界范围内 OpenStreetMap 嵌入的链接开放语料库】
- WorldKG: A World-Scale Geographic Knowledge Graph. CIKM 2021 【WorldKG：世界级地理知识图谱】
- RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms. CIKM 2021 【RecBole：迈向统一、全面和高效的推荐算法框架】
- RecBole 2.0: Towards a More Up-to-Date Recommendation Library. CIKM 2022 【RecBole 2.0：与时俱进的推荐库】

Technique
---------

### Pre-training in Recommender System

-   Pre-training for Ad-hoc Retrieval: Hyperlink is Also You Need. CIKM 2021【Ad-hoc检索预训练】
-   Pulling Up by the Causal Bootstraps: Causal Data Augmentation for Pre-training Debiasing. CIKM 2021【用于预训练去偏的因果关系数据增强】
-   HORNET: Enriching Pre-trained Language Representations with Heterogeneous Knowledge Sources. CIKM 2021【异构知识来源的预训练】
-   WebKE: Knowledge Extraction from Semi-structured Web with Pre-trained Markup Language Model. CIKM 2021【知识抽取+预训练】
-   Natural Language Understanding with Privacy-Preserving BERT. CIKM 2021【NLU+BERT】
-   K-AID: Enhancing Pre-trained Language Models with Domain Knowledge for Question Answering. CIKM 2021【applied paper，QA+领域知识+预训练】
-   DialogueBERT: A Self-Supervised Learning based Dialogue Pre-training Encoder. CIKM 2021【short paper，自监督对话预训练】
-   BERT-QPP: Contextualized Pre-trained transformers for Query Performance Prediction. CIKM 2021【short paper，用于查询性能预测的上下文预训练】
-   CANCN-BERT: A Joint Pre-Trained Language Model for Classical and Modern Chinese. CIKM 2021【short paper，古典和现代中文的联合预训练】
-   Distilling Knowledge from BERT into Simple Fully Connected Neural Networks for Efficient Vertical Retrieval. CIKM 2021【applied paper，知识蒸馏+预训练+检索】
-   Adversarial Reprogramming of Pretrained Neural Networks for Fraud Detection. CIKM 2021【short paper，用于欺诈检测的预训练对抗再编程】
-   Adversarial Domain Adaptation for Cross-lingual Information Retrieval with Multilingual BERT. CIKM 2021【short paper，使用多语言 BERT 进行跨语言信息检索的对抗域自适应】
-   Multi-modal Dictionary BERT for Cross-modal Video Search in Baidu Advertising. CIKM 2021【applied paper，百度广告中用于跨模态视频搜索的多模态词典BERT】
-   RABERT: Relation-Aware BERT for Target-Oriented Opinion Words Extraction. CIKM 2021【short paper，用于词提取的关系感知BERT】
-   GBERT: Pre-training User representations for Ephemeral Group Recommendation. CIKM 2022 【为临时组推荐预训练用户表示】

### Reinforcement Learning in Recommendations

-   Explore, Filter and Distill: Distilled Reinforcement Learning in Recommendation. CIKM 2021【applied paper，推荐中的蒸馏强化学习】
-   A Peep into the Future: Adversarial Future Encoding in Recommendation. WSDM 2022【推荐中的对抗性未来编码】
-   Toward Pareto Efficient Fairness-Utility Trade-off in Recommendation through Reinforcement Learning. WSDM 2022【通过强化学习在推荐中实现帕累托高效的公平-效用权衡】
-   Supervised Advantage Actor-Critic for Recommender Systems. WSDM 2022【推荐系统的监督优势Actor-Critic】

### Knowledge Distillation in Recommendations

-   Graph Structure Aware Contrastive Knowledge Distillation for Incremental Learning in Recommender Systems. CIKM 2021【short paper，推荐系统中用于增量学习的图结构感知的对比知识蒸馏】
-   Target Interest Distillation for Multi-Interest Recommendation. CIKM 2022 【多兴趣推荐的目标兴趣蒸馏】

### NAS in Recommendations



### Federated Learning in Recommendations

- Differentially Private Federated Knowledge Graphs Embedding. CIKM 2021【差异化隐私联邦KG Embedding】


### GNN in Recommendations

- DSKReG: Differentiable Sampling on Knowledge Graph for Recommendation with Relational GNN. CIKM 2021【short paper，用于推荐的知识图谱采样】
- UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation. CIKM 2021【GNN+RS】
- How Powerful is Graph Convolution for Recommendation? CIKM 2021【GNN+RS】
- Concept-Aware Denoising Graph Neural Network for Micro-Video Recommendation. CIKM 2021【用于微视频推荐的去噪GNN】
- Graph Logic Reasoning for Recommendation and Link Prediction. WSDM 2022【用于推荐和链接预测的图逻辑推理】
- Heterogeneous Global Graph Neural Networks for Personalized Session-based Recommendation. WSDM 2022【用于个性化基于会话的推荐的异构全局图神经网络】
- Community Trend Prediction on Heterogeneous Graph in E-commerce. WSDM 2022【电子商务异构图的社区趋势预测】
- Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation. CIKM 2022 【大规模推荐的神经相似度度量下的近似最近邻搜索】
- Automatic Meta-Path Discovery for Effective Graph-Based Recommendation. CIKM 2022 【基于图的有效推荐的自动元路径发现】
- HySAGE: A Hybrid Static and Adaptive Graph Embedding Network for Context-Drifting Recommendations. CIKM 2022 【用于上下文漂移推荐的混合静态和自适应图嵌入网络】
- Multi-Aggregator Time-Warping Heterogeneous Graph Neural Network for Personalized Micro-video Recommendation. CIKM 2022 【用于个性化微视频推荐的多聚合器时间扭曲异构图神经网络】
- PlatoGL: Effective and Scalable Deep Graph Learning System for Graph-enhanced Real-Time Recommendation. CIKM 2022 【用于图增强实时推荐的有效且可扩展的深度图学习系统】
- SVD-GCN: A Simplified Graph Convolution Paradigm for Recommendation. CIKM 2022 【用于推荐的简化图卷积范式】
- The Interaction Graph Auto-encoder Network Based on Topology-aware for Transferable Recommendation. CIKM 2022 【基于拓扑感知的可迁移推荐交互图自动编码器网络】

### Transformer in Recommendations

- LiteGT: Efficient and Lightweight Graph Transformers. CIKM 2021【高效轻量化图Transformer】
- Block Access Pattern Discovery via Compressed Full Tensor Transformer. CIKM 2021【Transformer压缩】
- Mixed Attention Transformer for Leveraging Word-Level Knowledge to Neural Cross-Lingual Information Retrieval. CIKM 2021【用于跨语言信息检索的混合注意力Transformer】
- Match-Ignition: Plugging PageRank into Transformer for Long-form Text Matching. CIKM 2021【PageRank+Transformer】
- DCAP: Deep Cross Attentional Product Network for User Response Prediction. CIKM 2021【用于用户响应预测的交叉注意力产品网络】
- Storage-saving Transformer for Sequential Recommendations. CIKM 2022 【用于序列推荐的节省存储的Transformer】
- Contrastive Learning with Bidirectional Transformers for Sequential Recommendation. CIKM 2022 【用于序列推荐的双向 Transformer 对比学习】

### Contrastive Learning in Recommendations

- Contrastive Pre-Training of GNNs on Heterogeneous Graphs. CIKM 2021【图神经网络的对比预训练】
- Contrastive Curriculum Learning for Sequential User Behavior Modeling via Data Augmentation. CIKM 2021【applied paper，通过数据增强进行序列用户行为建模的对比课程学习】
- Semi-deterministic and Contrastive Variational Graph Autoencoder for Recommendation. CIKM 2021【用于推荐的半确定性和对比变分图自动编码器】
- Contrastive Learning for Representation Degeneration Problem in Sequential Recommendation. WSDM 2022【序列推荐中表征退化问题的对比学习】
- Prototypical Contrastive Learning And Adaptive Interest Selection for Candidate Generation in Recommendations. CIKM 2022 【候选生成的原型对比学习和自适应兴趣选择】
- Multi-level Contrastive Learning Framework for Sequential Recommendation. CIKM 2022 【序列推荐多层次对比学习框架】
- Improving Knowledge-aware Recommendation with Multi-level Interactive Contrastive Learning. CIKM 2022 【通过多层次交互式对比学习改进知识感知推荐】
- MIC:Model-agnostic Integrated Cross-channel Recommender. CIKM 2022 【与模型无关的集成跨渠道推荐器】
- Temporal Contrastive Pre-Training for Sequential Recommendation. CIKM 2022 【时序推荐的时间对比预训练】

### Multi-Modality in Recommendations

- Student Can Also be a Good Teacher: Extracting Knowledge from Vision-and-Language Model for Cross-Modal Retrieval. CIKM 2021【short paper，用于跨模态检索的知识提取】
- Supervised Contrastive Learning for Multimodal Unreliable News Detection in COVID-19 Pandemic. CIKM 2021【short paper，用于多模态不可靠新闻检测的有监督对比学习】
- AutoMARS: Searching to Compress Multi-Modality Recommendation Systems. CIKM 2022 【搜索压缩多模态推荐系统】
- Multimodal Meta-Learning for Cold-Start Sequential Recommendation. CIKM 2022 【冷启动序列推荐的多模态元学习】

### Data Augmentation in Recommendations

- Influence-guided Data Augmentation for Neural Tensor Completion. CIKM 2021【用于张量补全的数据增强】
- Learning to Augment Imbalanced Data for Re-ranking Models. CIKM 2021【用于再排序模型的数据增强】
- Action Sequence Augmentation for Early Graph-based Anomaly Detection. CIKM 2021【用于异常检测的动作序列增强】
- A Relevant and Diverse Retrieval-enhanced Data Augmentation Framework for Sequential Recommendation. CIKM 2022 【Applied Research Track, 用于顺序推荐的相关且多样化的检索增强数据增强框架】
- MARIO: Modality-Aware Attention and Modality-Preserving Decoders for Multimedia Recommendation. CIKM 2022 【用于多媒体推荐的模态感知注意力和模态保留解码器】

### Meta Learning in Recommendations

- Multimodal Graph Meta Contrastive Learning. CIKM 2021【short paper，多模态元图对比学习】
- Meta-Learning Based Hyper-Relation Feature Modeling for Out-of-Knowledge-Base Embedding. CIKM 2021【基于元学习的超关系特征建模】
- HetMAML: Task-Heterogeneous Model-Agnostic Meta-Learning for Few-Shot Learning Across Modalities. CIKM 2021【Meta Learning+Few-Shot Learning】
- Pruning Meta-Trained Networks for On-Device Adaptation. CIKM 2021【用于设备自适应的元训练网络剪枝】
- Meta Hyperparameter Optimization with Adversarial Proxy Subsets Sampling. CIKM 2021【元超参优化】
- Contrastive Meta Learning with Behavior Multiplicity for Recommendation. WSDM 2022【具有行为多样性的对比元学习推荐】

### Few-Shot Learning in Recommendations

- Behind the Scenes: An Exploration of Trigger Biases Problem in Few-Shot Event Classification. CIKM 2021【小样本学习中偏差问题的探讨】
- Learning Discriminative and Unbiased Representations for Few-Shot Relation Extraction. CIKM 2021【用于小样本关系提取的无偏表示学习】
- Multi-view Interaction Learning for Few-Shot Relation Classification. CIKM 2021【用于小样本关系分类的多视角交互学习】
- One-shot Transfer Learning for Population Mapping. CIKM 2021【单样本迁移学习】
- Boosting Few-shot Abstractive Summarization with Auxiliary Tasks. CIKM 2021【short paper，使用辅助任务提升小样本摘要】
- Multi-objective Few-shot Learning for Fair Classification. CIKM 2021【short paper，Few-Shot Learning+Classification】
- Few-shot Link Prediction in Dynamic Networks. WSDM 2022 【动态网络中的少样本链接预测研究】
- Tiger: Transferable Interest Graph Embedding for Domain-Level Zero-Shot Recommendation. CIKM 2022 【用于域级零样本推荐的可迁移兴趣图嵌入】


Analysis
--------

-   Profiling the Design Space for Graph Neural Networks based Collaborative Filtering. WSDM 2022【分析基于图神经网络的协同过滤的设计空间】

Others
-----

-   Joint Learning of E-commerce Search and Recommendation with A Unified Graph Neural Network. WSDM 2022【电子商务搜索和推荐与统一图神经网络的联合学习】
-   Hierarchical Imitation Learning via Subgoal Representation Learning for Dynamic Treatment Recommendation. WSDM 2022【基于动态治疗推荐的子目标表示学习的分层模仿学习】
-   PLdFe-RR:Personalized Long-distance Fuel-efficient Route Recommendation Based On Historical Trajectory. WSDM 2022【基于历史轨迹的个性化长途省油路线推荐】
-   The Datasets Dilemma: How Much Do We Really Know About Recommendation Datasets? WSDM 2022【数据集困境：我们对推荐数据集真正了解多少？】
-   On Sampling Collaborative Filtering Datasets. WSDM 2022【关于采样协同过滤数据集】
-   Leveraging World Events to Predict E-Commerce Consumer Demand under Anomaly. WSDM 2022【利用世界事件预测异常情况下的电子商务消费者需求】
-   Scope-aware Re-ranking with Gated Attention in Feed. WSDM 2022【在 Feed 中使用 Gated Attention 进行范围感知重新排名】
-   Best practices for top-N recommendation evaluation: Candidate set sampling and Statistical inference techniques. CIKM 2022 【候选集抽样和统计推断技术】
-   PROPN: Personalized Probabilistic Strategic Parameter Optimization in Recommendations. CIKM 2022 【推荐中的个性化概率策略参数优化】
-   Multi-granularity Fatigue in Recommendation. CIKM 2022 【推荐中的多粒度疲劳】
-   A Multi-Interest Evolution Story: Applying Psychology in Query-based Recommendation for Inferring Customer Intention. CIKM 2022 【在基于查询的推荐中应用心理学以推断客户意图】
-   Improving Text-based Similar Product Recommendation for Dynamic Product Advertising at Yahoo. CIKM 2022 【改进雅虎动态产品广告的基于文本的相似产品推荐】
-   E-Commerce Promotions Personalization via Online Multiple-Choice Knapsack with Uplift Modeling. CIKM 2022 【在线的电子商务促销个性化】
