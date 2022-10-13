## Catalog (目录)

**1 Sort by task (按照任务场景划分)**

- [CTR/CVR Prediction](#CTR/CVR-Prediction)
- [Collaborative Filtering](#Collaborative-Filtering)
- [Sequential/Session-based Recommendation](#Sequential/Session-based-Recommendation)
- [Conversational Recommender System](#Conversational-Recommender-System)
- [Knowledge-aware Recommendation](#Knowledge-aware-Recommendation)
- [Social Recommendation](#Social-Recommendation)
- [POI Recommendation](#POI-Recommendation)
- [News Recommendation](#News-Recommendation)
- [Online Recommendation](#Online-Recommendation)
- [Bundle Recommendation](#Bundle-Recommendation)
- [Music Recommendation](#Music-Recommendation)
- [Other task](#Other-task)

**2 Sort by main technique (按照主要技术划分)**

- [Factorization Machines](#Factorization-Machines)
- [GNN-based](#GNN-based)
- [Pre-training in Recommender
  System](#pre-training-in-recommender-system)
- [Contrastive Learning based](#Contrastive-Learning-based)
- [Adversarial Learning based](#Adversarial-Learning-based)
- [Autoencoder based](#Autoencoder-based)
- [Reinforcemnet Learning](#Reinforcemnet-Learning)
- [Bandit Algorithm](#Bandit-Algorithm)
- [Other technique](#Other-technique)

 **3 Sort by topic (按研究话题划分)**

- [Bias/Debias in Recommender System](#Bias/Debias-in-Recommender-System)
- [Explanation in Recommender System](#Explanation-in-Recommender-System)
- [Long-tail/Cold-start in Recommender System](#Long-tail/Cold-start-in-Recommender-System)
- [Fairness in Recommender System](#Fairness-in-Recommender-System)
- [Diversity in Recommender System](#Diversity-in-Recommender-System)
- [Denoising in Recommender System](#Denoising-in-Recommender-System)
- [Attack in Recommender System](#Attack-in-Recommender-System)
- [Privacy Protection in Recommender System](#Privacy-Protection-in-Recommender-System)
- [Evaluation of Recommender System](#Evaluation-of-Recommender-System)
- [Other topic](#Other-topic)

------



## Sort by task(按照任务场景划分)

### CTR /CVR Prediction

- CAEN: A Hierarchically Attentive Evolution Network for Item-Attribute-Change-Aware Recommendation in the Growing E-commerce Environment RecSys 2022 【物品属性变化感知的分层注意力动态网络】
- Dual Attentional Higher Order Factorization Machines RecSys 2022 【双注意高阶因式分解机】
- Position Awareness Modeling with Knowledge Distillation for CTR Prediction RecSys 2022 【LBR，位置感知的知识提取框架】

### Collaborative Filtering

- ProtoMF: Prototype-based Matrix Factorization for Effective and Explainable Recommendations RecSys 2022【基于原型的可解释协同过滤算法】
- Revisiting the Performance of iALS on Item Recommendation Benchmarks RecSys 2022【Reproducibility paper，隐式交替最小二乘法 iALS 基准表现的再思考】
- Scalable Linear Shallow Autoencoder for Collaborative Filtering RecSys 2022 【LBR，用于协同过滤的可扩展线性浅层自动编码器】

### Sequential/Session-based Recommendation

- Aspect Re-distribution for Learning Better Item Embeddings in Sequential Recommendation RecSys 2022 【序列推荐中物品向量的方面重分布】
- Context and Attribute-Aware Sequential Recommendation via Cross-Attention RecSys 2022 【交叉注意力机制实现上下文和属性感知的序列推荐】
- Defending Substitution-Based Profile Pollution Attacks on Sequential Recommenders RecSys 2022 【序列推荐中基于替代的对抗性攻击算法】
- Denoising Self-Attentive Sequential Recommendation RecSys 2022 【基于 Transformer 的序列推荐中自注意力机制的去噪】
- Don't recommend the obvious: estimate probability ratios RecSys 2022 【通过拟合逐点的互信息来改进序列推荐的流行度采样指标】
- Effective and Efficient Training for Sequential Recommendation using Recency Sampling RecSys 2022 【基于接近度采样进行高效的序列推荐】
- Global and Personalized Graphs for Heterogeneous Sequential Recommendation by Learning Behavior Transitions and User Intentions RecSys 2022 【异构序列推荐中通过全局和个性化的图建模学习行为转换和用户意图】
- Learning Recommendations from User Actions in the Item-poor Insurance Domain RecSys 2022 【保险领域使用循环神经网络的跨会话模型】
- A Systematic Review and Replicability Study of BERT4Rec for Sequential Recommendation RecSys 2022【Reproducibility paper，BERT4Rec 结果的系统回顾与可复现性研究】
- Streaming Session-Based Recommendation: When Graph Neural Networks meet the Neighborhood RecSys 2022【Reproducibility paper，图神经网络解决流会话推荐问题】
- M2TRec: Metadata-aware Multi-task Transformer for Large-scale and Cold-start free Session-based Recommendations RecSys 2022【LBR，基于元数据和多任务 Transformer 的冷启动会话推荐系统】

### Conversational Recommender System

- Bundle MCR: Towards Conversational Bundle Recommendation RecSys 2022 【马尔可夫决策进行对话式捆绑推荐】
- Self-Supervised Bot Play for Transcript-Free Conversational Recommendation with Rationales RecSys 2022 【自监督对话推荐】

### Knowledge-aware Recommendation

- TinyKG: Memory-Efficient Training Framework for Knowledge Graph Neural Recommender Systems RecSys 2022 【知识图神经推荐系统的内存高效训练框架】
- Knowledge-aware Recommendations Based on Neuro-Symbolic Graph Embeddings and First-Order Logical Rules RecSys 2022 【LBR，基于神经符号图表示的知识感知推荐框架】

### Social Recommendation

- Do recommender systems make social media more susceptible to misinformation spreaders? RecSys 2022 【LBR，错误信息传播者对社交推荐的影响】

### POI Recommendation

- Exploring the Impact of Temporal Bias in Point-of-Interest Recommendation RecSys 2022 【LBR，时间偏差对兴趣点推荐的影响】

### News Recommendation

- RADio – Rank-Aware Divergence Metrics to Measure Normative Diversity in News Recommendations RecSys 2022【衡量新闻推荐规范化多样性的度量框架】
- Reducing Cross-Topic Political Homogenization in Content-Based News Recommendation RecSys 2022【新闻推荐中减少跨主题的政治同质化】

### Online Recommendation

- A GPU-specialized Inference Parameter Server for Large-Scale Deep Recommendation Models RecSys 2022 【大规模在线推理模型基于 GPU 高速缓存的分布式框架】
- Modeling User Repeat Consumption Behavior for Online Novel Recommendation RecSys 2022 【在线小说推荐的用户重复消费行为建模】

### Bundle Recommendation

- BRUCE – Bundle Recommendation Using Contextualized item Embeddings RecSys 2022 【Transformer 建模上下文进行捆绑推荐】
- Bundle MCR: Towards Conversational Bundle Recommendation RecSys 2022 【马尔可夫决策进行对话式捆绑推荐】

### Music Recommendation

- A User-Centered Investigation of Personal Music Tours RecSys 2022 【以用户为中心的音乐巡演推荐】
- Exploiting Negative Preference in Content-based Music Recommendation with Contrastive Learning RecSys 2022 【利用对比学习挖掘基于内容的音乐推荐中的负面偏好】
- Exploring the longitudinal effect of nudging on users' genre exploration behavior and listening preference RecSys 2022 【探索轻推对用户听歌体裁偏好的纵向效应】
- Discovery Dynamics: Leveraging Repeated Exposure for User and Music CharacterizationRecSys 2022 【LBR，探索轻推对用户听歌体裁偏好的纵向效应】

### Other task

- Learning to Ride a Buy-Cycle: A Hyper-Convolutional Model for Next Basket Repurchase Recommendation RecSys 2022 【针对下一篮回购推荐问题的超卷积模型】
- MARRS: A Framework for multi-objective risk-aware route recommendation using Multitask-Transformer RecSys 2022 【利用多任务 Transformer 进行多目标的路线推荐】
- Modeling Two-Way Selection Preference for Person-Job Fit RecSys 2022 【建模双向选择偏好的人岗匹配模型】
- Multi-Modal Dialog State Tracking for Interactive Fashion Recommendation RecSys 2022 【交互式时装推荐的多模态注意网络】
- Towards Psychologically Grounded Dynamic Preference Models RecSys 2022 【基于心理学的动态偏好建模】

## Sort by main technique(按照主要技术划分)

### Factorization Machines

- Dual Attentional Higher Order Factorization Machines RecSys 2022 【双注意高阶因式分解机】
- You Say Factorization Machine, I Say Neural Network – It’s All in the Activation RecSys 2022 【通过激活函数建立因子分解机和神经网络的联系】

### GNN-based

- Global and Personalized Graphs for Heterogeneous Sequential Recommendation by Learning Behavior Transitions and User Intentions RecSys 2022 【异构序列推荐中通过全局和个性化的图建模学习行为转换和用户意图】
- TinyKG: Memory-Efficient Training Framework for Knowledge Graph Neural Recommender Systems RecSys 2022 【知识图神经推荐系统的内存高效训练框架】
- Streaming Session-Based Recommendation: When Graph Neural Networks meet the Neighborhood RecSys 2022【Reproducibility paper，图神经网络解决流会话推荐问题】

### Pre-training in Recommender System

- Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5) RecSys 2022 【通用的预训练、个性化提示和预测范式建模推荐任务】

### Contrastive Learning based

- Exploiting Negative Preference in Content-based Music Recommendation with Contrastive Learning RecSys 2022 【利用对比学习挖掘基于内容的音乐推荐中的负面偏好】

### Adversarial Learning based

- Adversary or Friend? An adversarial Approach to Improving Recommender Systems RecSys 2022 【对抗式方法促进推荐系统公平性】

### Autoencoder based

- Scalable Linear Shallow Autoencoder for Collaborative Filtering RecSys 2022 【LBR，用于协同过滤的可扩展线性浅层自动编码器】

### Reinforcement Learning

- Off-Policy Actor Critic for Recommender Systems RecSys 2022 【离线演员-评论家强化学习算法缓解分布偏差问题】
- Multiobjective Evaluation of Reinforcement Learning Based Recommender Systems RecSys 2022【LBR，基于强化学习的推荐系统的多目标评价】

### Bandit Algorithm

- Dynamic Global Sensitivity for Differentially Private Contextual Bandits RecSys 2022 【通过差分私有的上下文 Bandit 算法保护隐私】
- Identifying New Podcasts with High General Appeal Using a Pure Exploration Infinitely-Armed Bandit Strategy RecSys 2022 【通过 Bandit 策略进行播客推荐】

### Other technique

- Towards Recommender Systems with Community Detection and Quantum Computing RecSys 2022【LBR，利用量子计算进行社区检测】

## Sort by topic (按研究话题划分)

### Bias/Debias in Recommender System

- Countering Popularity Bias by Regularizing Score Differences RecSys 2022 【利用正则化分数差异减少流行度偏差】
- Off-Policy Actor Critic for Recommender Systems RecSys 2022 【离线演员-评论家强化学习算法缓解分布偏差问题】
- Exploring the Impact of Temporal Bias in Point-of-Interest Recommendation RecSys 2022 【LBR，时间偏差对兴趣点推荐的影响】

### Explanation in Recommender System

- ProtoMF: Prototype-based Matrix Factorization for Effective and Explainable Recommendations RecSys 2022【基于原型的可解释协同过滤算法】

### Long-tail/Cold-start in Recommender System

- Fast And Accurate User Cold-Start Learning Using Monte Carlo Tree Search RecSys 2022 【蒙特卡洛树搜索进行用户冷启动学习】
- M2TRec: Metadata-aware Multi-task Transformer for Large-scale and Cold-start free Session-based Recommendations RecSys 2022【LBR，基于元数据和多任务 Transformer 的冷启动会话推荐系统】

### Fairness in Recommender System

- Adversary or Friend? An adversarial Approach to Improving Recommender Systems RecSys 2022 【对抗式方法促进推荐系统公平性】
- Fairness-aware Federated Matrix Factorization RecSys 2022 【结合差异隐私技术的公平感知的联邦矩阵分解】
- Toward Fair Federated Recommendation Learning: Characterizing the Inter-Dependence of System and Data Heterogeneity RecSys 2022 【推荐中公平的联邦学习】

### Diversity in Recommender System

- Solving Diversity-Aware Maximum Inner Product Search Efficiently and Effectively RecSys 2022 【多样性感知的最大内部产品搜索】

### Denoising in Recommender System

- Denoising Self-Attentive Sequential Recommendation RecSys 2022 【基于 Transformer 的序列推荐中自注意力机制的去噪】

### Attack in Recommender System

- Defending Substitution-Based Profile Pollution Attacks on Sequential Recommenders RecSys 2022 【序列推荐中基于替代的对抗性攻击算法】

### Privacy Protection in Recommender System

- Dynamic Global Sensitivity for Differentially Private Contextual Bandits RecSys 2022 【通过差分私有的上下文 Bandit 算法保护隐私】
- EANA: Reducing Privacy Risk on Large-scale Recommendation Models RecSys 2022 【降低大规模推荐模型的隐私风险】
- Fairness-aware Federated Matrix Factorization RecSys 2022 【结合差异隐私技术的公平感知的联邦矩阵分解】
- Toward Fair Federated Recommendation Learning: Characterizing the Inter-Dependence of System and Data Heterogeneity RecSys 2022 【推荐中公平的联邦学习】

### Evaluation of Recommender System

- Don't recommend the obvious: estimate probability ratios RecSys 2022 【通过拟合逐点的互信息来改进序列推荐的流行度采样指标】
- RADio – Rank-Aware Divergence Metrics to Measure Normative Diversity in News Recommendations RecSys 2022【衡量新闻推荐规范化多样性的度量框架】
- A Systematic Review and Replicability Study of BERT4Rec for Sequential Recommendation RecSys 2022【Reproducibility paper，BERT4Rec 结果的系统回顾与可复现性研究】
- Revisiting the Performance of iALS on Item Recommendation Benchmarks RecSys 2022【Reproducibility paper，隐式交替最小二乘法 iALS 基准表现的再思考】
- Measuring Commonality in Recommendation of Cultural Content: Recommender Systems to Enhance Cultural Citizenship RecSys 2022【LBR，通用性作为文化内容推荐的度量】
- Multiobjective Evaluation of Reinforcement Learning Based Recommender Systems RecSys 2022【LBR，基于强化学习的推荐系统的多目标评价】

### Other topic

- Recommender Systems and Algorithmic Hate RecSys 2022 【LBR，对用户方案推荐系统算法的探究性工作】
- The Effect of Feedback Granularity on Recommender Systems Performance RecSys 2022 【LBR，评分和反馈粒度对推荐性能的影响】