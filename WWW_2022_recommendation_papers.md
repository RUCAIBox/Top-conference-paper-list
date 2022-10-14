## Catalog (目录)

**1 Sort by task (按照任务场景划分)**

- [Sequential/Session-based Recommendation](#Sequential/Session-based-Recommendation)
- [News Recommendation](#News-Recommendation)
- [Cross-domain Recommendation](#Cross-domain-Recommendation)


**2 Sort by main technique (按照主要技术划分)**

- [GNN based](#GNN-based)
- [Reinforcemnet Learning](#Reinforcemnet-Learning)


 **3 Sort by topic (按研究话题划分)**

- [Bias/Debias in Recommender System](#Bias/Debias-in-Recommender-System)
- [Explanation in Recommender System](#Explanation-in-Recommender-System)
- [Long-tail/Cold-start in Recommender System](#Long-tail/Cold-start-in-Recommender-System)
- [Fairness in Recommender System](#Fairness-in-Recommender-System)


------



## Sort by task(按照任务场景划分)


### Sequential/Session-based Recommendation

-   Disentangling Long and Short-Term Interests for Recommendation. WWW 2022 [利用自监督对比学习解耦长短期兴趣]

-   Efficient Online Learning to Rank for Sequential Music Recommendation. WWW 2022 [将搜索空间限制在此前表现不佳的搜索方向的正交补上]

-   Filter-enhanced MLP is All You Need for Sequential Recommendation. www 2022 [通过可学习滤波器对用户序列进行编码]

-   Generative Session-based Recommendation. WWW 2022 [构建生成器来模拟用户序列行为，从而改善目标序列推荐模型]

-   GSL4Rec: Session-based Recommendations with Collective
Graph Structure Learning and Next Interaction Prediction. WWW 2022 [图结构学习+推荐]

-   Intent Contrastive Learning for Sequential Recommendation. WWW 2022 [利用用户意图来增强序列推荐]

-   Learn from Past, Evolve for Future: Search-based Time-aware Recommendation with Sequential Behavior Data. WWW 2022 [检索相关历史行为并融合当前序列行为]

-   Sequential Recommendation via Stochastic Self-Attention. WWW 2022 [通过随机高斯分布和Wasserstein自注意力模块来引入不确定性]

-   Sequential Recommendation with Decomposed Item Feature Routing. WWW 2022 [解耦item特征，并分别利用软硬模型来路由最有特征序列]
  
-   Towards Automatic Discovering of Deep Hybrid Network Architecture for Sequential Recommendation. WWW 2022 [通过NAS来搜索每一层使用注意力/卷积模块]

-   Unbiased Sequential Recommendation with Latent Confounders. WWW 2022 [去除潜在混淆变量来实现无偏序列推荐]

-   Re4: Learning to Re-contrast, Re-attend, Re-construct for Multi-interest Recommendation. WWW 2022 [通过re-contrast,re-attend,re-construct来增强解耦用户多兴趣表示]

### News Recommendation

-   FeedRec: News Feed Recommendation with Various User Feedbacks. WWW 2022 [融入各类显示/隐式反馈来建模用户兴趣]

### Cross-domain Recommendation

-   Collaborative Filtering with Attribution Alignment for Review-based Non-overlapped Cross Domain Recommendation. WWW 2022 [通过属性对齐实现基于评论的跨域推荐]

-   Differential Private Knowledge Transfer for Privacy-Preserving Cross-Domain Recommendation. WWW 2022 [通过可微隐私知识迁移实现源域隐私保护的跨域推荐]

-   MetaBalance: Improving Multi-Task Recommendations via Adapting Gradient Magnitudes of Auxiliary Tasks. WWW 2022 [动态保持辅助任务和目标任务的梯度在同一个量级]



## Sort by main technique (按照主要技术划分)

### GNN based

    -   Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning. WWW 2022 [通过邻居节点间的对比学习来改善图协同过滤]

    -   Revisiting Graph based Social Recommendation: A Distillation Enhanced Social Graph Network. WWW 2022 [使用知识蒸馏来融入user-item交互图和user-user社交图的信息]

    -   STAM: A Spatiotemporal Aggregation Method for Graph Neural Network-based Recommendation. WWW 2022 [同时聚合时空领域信息]

    -   Hypercomplex Graph Collaborative Filtering. WWW 2022 [超复图协同过滤]

    -   Graph Neural Transport Networks with Non-local Attentions for Recommender Systems. WWW 2022 [使用非局部注意力来实现不加深GNN的同时捕捉节点间的长距离依赖]

    -   FIRE: Fast Incremental Recommendation with Graph Signal Processing. WWW 2022 [通过图信号处理来实现快速增量推荐]

    -   Large-scale Personalized Video Game Recommendation via Social-aware Contextualized Graph Neural Network. WWW 202 [同时考虑个性化，游戏上下文，社交联系]

### Reinforcement Learning

-   MINDSim: User Simulator for News Recommenders. WWW 2022 [用户模拟+新闻推荐]

-   Multi-level Recommendation Reasoning over Knowledge Graphs with Reinforcement Learning. WWW 2022

-   Multiple Choice Questions based Multi-Interest Policy Learning for Conversational Recommendation. WWW 2022

-   Off-policy Learning over Heterogeneous Information for Recommendation. WWW 2022

### NAS in Recommendations

- Towards Automatic Discovering of Deep Hybrid Network Architecture for Sequential Recommendation. WWW 2022 [通过NAS来搜索每一层使用注意力/卷积模块]



## Sort by topic (按研究话题划分)

### Bias/Debias in Recommender System

-   Causal Representation Learning for Out-of-Distribution Recommendation. WWW 2022 [利用因果模型解决用户特征变化问题]

-   A Model-Agnostic Causal Learning Framework for Recommendation using Search Data. WWW 2022 [将搜索数据作为工具变量，解耦推荐中的因果部分和非因果部分]

-   Causal Preference Learning for Out-of-Distribution Recommendation. WWW 2022 [从观察数据可用的正反馈中联合学习具有因果结构的不变性偏好，再用发现的不变性偏好继续做预测]
  
-   Learning to Augment for Casual User Recommendation. WWW 2022 [通过数据增强来增强对随机用户的推荐性能]

-   CBR: Context Bias aware Recommendation for Debiasing User Modeling and Click Prediction. WWW 2022 [去除由丰富交互造成的上下文偏差]

-   Cross Pairwise Ranking for Unbiased Item Recommendation. WWW 2022 [利用交叉成对损失来去偏]

-   Rating Distribution Calibration for Selection Bias Mitigation in Recommendations. WWW 2022 [通过校准评分分布来缓解选择偏差]

-   UKD: Debiasing Conversion Rate Estimation via Uncertainty-regularized Knowledge Distillation. WWW 2022

### Explanation in Recommender System

-   Graph-based Extractive Explainer for Recommendations. WWW 2022 [使用图注意力网络来实现可解释推荐]

-   ExpScore: Learning Metrics for Recommendation Explanation. WWW 2022 [可解释推荐评价指标]

-   Path Language Modeling over Knowledge Graphs for Explainable Recommendation. WWW 2022 [在知识图谱上学习语言模型，实现推荐和解释]

-   Accurate and Explainable Recommendation via Review Rationalization. WWW 2022 [提取评论中的因果关系]

-   AmpSum: Adaptive Multiple-Product Summarization towards Improving Recommendation Captions. WWW 2022 [生成商品标题]

-   Comparative Explanations of Recommendations. WWW 2022 [可比较的推荐解释]

-   Neuro-Symbolic Interpretable Collaborative Filtering for Attribute-based Recommendation. WWW 2022 [以模型为核心的神经符号可解释协同过滤]
-   
### Long-tail/Cold-start in Recommender System

-   Alleviating Cold-start Problem in CTR Prediction with A Variational Embedding Learning Framework. WWW 2022 [使用变分embedding学习框架缓解 CTR 预测中的冷启动问题]

-   PNMTA: A Pretrained Network Modulation and Task Adaptation Approach for User Cold-Start Recommendation. WWW 2022 [加入编码调制器和预测调制器，使得编码器和预测器可以自适应处理冷启动用户。]

-   KoMen: Domain Knowledge Guided Interaction Recommendation for Emerging Scenarios. WWW 2022 [元学习+图网络]

### Fairness in Recommender System

-   Link Recommendations for PageRank Fairness. WWW 2022 [PageRank算法链接预测中的公平性]
-   FairGAN: GANs-based Fairness-aware Learning for Recommendations with Implicit Feedback. WWW 2022 [将物品曝光公平性问题映射为隐式反馈数据中缺乏负反馈的问题]



### Privacy Protection in Recommender System

- Privacy Preserving Collaborative Filtering by Distributed Mediation RecSys 2021【隐私保护式协同过滤】
- A Payload Optimization Method for Federated Recommender Systems RecSys 2021 【联邦推荐的有效负载优化方法】
- Stronger Privacy for Federated Collaborative Filtering With Implicit Feedback RecSys 2021【隐式数据的联邦推荐系统】
- FR-FMSS: Federated Recommendation via Fake Marks and Secret Sharing RecSys 2021【LBR，跨用户联邦推荐框架】
- Horizontal Cross-Silo Federated Recommender Systems RecSys 2021【LBR，联邦推荐对多类利益相关者的影响研究】

