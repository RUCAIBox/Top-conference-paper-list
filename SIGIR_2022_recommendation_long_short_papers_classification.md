## Catalog (目录)

**1 Sort by task(按照任务场景划分)**

- [CTR/CVR Prediction](#CTR/CVR-Prediction)
- [Collaborative Filtering](#Collaborative-Filtering)
- [Sequential/Session-based Recommendation](#Sequential/Session-based-Recommendation)
- [Conversational Recommender System](#Conversational-Recommender-System)
- POI Recommendation(#POI-Recommendation)
- Cross-domain/Multi-behavior Recommendation(#Cross-domain/Multi-behavior-Recommendation)
- Knowledge-aware Recommendation(#Knowledge-aware-Recommendation)
- News Recommendation(#News-Recommendation)
- Other task(#other-task)

**2 Sort by main technique(按照主要技术划分)**

- GNN-based
- RL-based
- Contrastive Learning based
- AutoML-based
- Others

 **3 Sort by topic(按研究话题划分)**

- Bias/Debias in Recommender System
- Explanation in Recommender System
- Long-tail/Cold-start in Recommender System
- Fairness in Recommender System
- Diversity in Recommender System
- Attack/Denoise in Recommender System
- Others

 **4 Other Research topics(其他研究方向)**

- QA
- Knowledge Graph
- Conversation/ Dialog
- Summarization
- Multi-Modality
- Generation
- Representation Learning

------



## Sort by task(按照任务场景划分)

### CTR /CVR Prediction

- Enhancing CTR Prediction with Context-Aware Feature Representation Learning 【上下文相关的特征表示】
- HIEN: Hierarchical Intention Embedding Network for Click-Through Rate Prediction 【层次化意图嵌入网络】
- NAS-CTR: Efficient Neural Architecture Search for Click-Through Rate Prediction 【高效的网络结构搜索】
- NMO: A Model-Agnostic and Scalable Module for Inductive Collaborative Filtering 【模型无关的归纳式协同过滤模块】
- Neighbour Interaction based Click-Through Rate Prediction via Graph-masked Transformer 【图遮盖的Transformer】
- Neural Statistics for Click-Through Rate Prediction 【short paper，神经统计学】
- Smooth-AUC: Smoothing the Path Towards Rank-based CTR Prediction 【short paper，基于排序的CTR预估】
- DisenCTR: Dynamic Graph-based Disentangled Representation for Click-Through Rate Prediction 【基于图的解耦表示】
- Deep Multi-Representational Item Network for CTR Prediction 【short paper，多重表示商品网络】
- Gating-adapted Wavelet Multiresolution Analysis for Exposure Sequence Modeling in CTR prediction 【short paper，多分辨率小波分析】
- MetaCVR: Conversion Rate Prediction via Meta Learning in Small-Scale Recommendation Scenarios 【short paper，小规模推荐场景下的元学习】
- Adversarial Filtering Modeling on Long-term User Behavior Sequences for Click-Through Rate Prediction 【short paper，对抗过滤建模用户长期行为序列】
- Clustering based Behavior Sampling with Long Sequential Data for CTR Prediction 【short paper，长序列数据集基于聚类的行为采样】
- CTnoCVR: A Novelty Auxiliary Task Making the Lower-CTR-Higher-CVR Upper 【short paper，新颖度辅助任务】

### Collaborative Filtering

- Geometric Disentangled Collaborative Filtering 【几何解耦的协同过滤】
- Self-Augmented Recommendation with Hypergraph Contrastive Collaborative Filtering 【超图上的对比学习】
- Investigating Accuracy-Novelty Performance for Graph-based Collaborative Filtering 【图协同过滤在准确度和新颖度上的表现】
- Unify Local and Global Information for Top-N Recommendation 【综合局部和全局信息】
- Enhancing Top-N Item Recommendations by Peer Collaboration 【short paper ，同龄人协同】
- Evaluation of Herd Behavior Caused by Population-scale Concept Drift in Collaborative Filtering 【short paper】

### Sequential/Session-based Recommendation

- Decoupled Side Information Fusion for Sequential Recommendation 【融合边缘特征的序列推荐】
- On-Device Next-Item Recommendation with Self-Supervised Knowledge Distillation 【自监督知识蒸馏】
- Multi-Agent RL-based Information Selection Model for Sequential Recommendation 【多智能体信息选择】
- An Attribute-Driven Mirroring Graph Network for Session-based Recommendation 【特征驱动的反射图网络】
- When Multi-Level Meets Multi-Interest: A Multi-Grained Neural Model for Sequential Recommendation 【多粒度网络】
- Price DOES Matter! Modeling Price and Interest Preferences in Session-based Recommendation 【考虑价格和兴趣的推荐】
- AutoGSR: Neural Architecture Search for Graph-based Session Recommendation 【面向图会话推荐的网络结构搜索】
- Ada-Ranker: A Data Distribution Adaptive Ranking Paradigm for Sequential Recommendation 【数据分布自适应排序】
- Multi-Faceted Global Item Relation Learning for Session-Based Recommendation 【多面全局商品关系学习】
- ReCANet: A Repeat Consumption-Aware Neural Network for Next Basket Recommendation in Grocery Shopping 【考虑重复消费的网络】
- Determinantal Point Process Set Likelihood-Based Loss Functions for Sequential Recommendation 【基于DPP的损失函数】
- Positive, Negative and Neutral: Modeling Implicit Feedback in Session-based News Recommendation 【建模隐式反馈】
- Coarse-to-Fine Sparse Sequential Recommendation 【short paper，粗到细的稀疏序列化推荐】
- Dual Contrastive Network for Sequential Recommendation 【short paper，双对比网络】
- Explainable Session-based Recommendation with Meta-Path Guided Instances and Self-Attention Mechanism 【short paper， 基于元路径指导和自注意力机制的可解释会话推荐】 
- Item-Provider Co-learning for Sequential Recommendation 【short paper，商品-商家一同训练】
- RESETBERT4Rec: A Pre-training Model Integrating Time And User Historical Behavior for Sequential Recommendation 【short paper，融合时间和用户历史行为的预训练模型】
- Enhancing Hypergraph Neural Networks with Intent Disentanglement for Session-based Recommendation【short paper，意图解耦增强超图神经网络】
- CORE: Simple and Effective Session-based Recommendation within Consistent Representation Space 【short paper，在一致表示空间上的简单有效会话推荐】
- DAGNN: Demand-aware Graph Neural Networks for Session-based Recommendation 【short paper， 需求感知的图神经网络】
- Progressive Self-Attention Network with Unsymmetrical Positional Encoding for Sequential Recommendation 【short paper，使用非对称位置编码的自注意力网络】
- ELECRec: Training Sequential Recommenders as Discriminators 【short paper，训练序列推荐模型作为判别器】
- Exploiting Session Information in BERT-based Session-aware Sequential Recommendation 【short paper，在基于BERT的模型中利用会话信息】

### Conversational Recommender System

- Learning to Infer User Implicit Preference in Conversational Recommendation 【学习推测用户隐偏好】
- User-Centric Conversational Recommendation with Multi-Aspect User Modeling 【多角度用户建模】
- Variational Reasoning about User Preferences for Conversational Recommendation 【用户偏好的变分推理】
- Analyzing and Simulating User Utterance Reformulation in Conversational Recommender Systems 【对话推荐中模仿用户言论】
- Improving Conversational Recommender Systems via Transformer-based Sequential Modelling【short paper，基于Transformer的序列化建模】
- Conversational Recommendation via Hierarchical Information Modeling 【short paper，层次化信息建模】

### POI Recommendation

- Hierarchical Multi-Task Graph Recurrent Network for Next POI Recommendation 【多任务图循环网络】
- Learning Graph-based Disentangled Representations for Next POI Recommendation 【学习基于图的解耦表示】
- GETNext: Trajectory Flow Map Enhanced Transformer for Next POI Recommendation 【轨迹图加强的Transformer】
- Next Point-of-Interest Recommendation with Auto-Correlation Enhanced Multi-Modal Transformer Network 【short paper，自修正的多模态Transformer】
- Empowering Next POI Recommendation with Multi-Relational Modeling 【多重关系建模】

### Cross-domain/Multi-behavior Recommendation

- Co-training Disentangled Domain Adaptation Network for Leveraging Popularity Bias in Recommenders 【训练解耦的域适应网络来利用流行度偏差】
- DisenCDR: Learning Disentangled Representations for Cross-Domain Recommendation 【解耦表示】
- Doubly-Adaptive Reinforcement Learning for Cross-Domain Interactive Recommendation 【双重适应的强化学习】
- Exploiting Variational Domain-Invariant User Embedding for Partially Overlapped Cross Domain Recommendation 【域不变的用户嵌入】
- Multi-Behavior Sequential Transformer Recommender 【多行为序列化Transformer】

### Knowledge-aware Recommendation

- Knowledge Graph Contrastive Learning for Recommendation 【知识图谱上的对比学习】
- Multi-level Cross-view Contrastive Learning for Knowledge-aware Recommender System 【多级交叉视图的对比学习】
- Alleviating Spurious Correlations in Knowledge-aware Recommendations through Counterfactual Generator 【利用反事实生成器缓解假知识】
- HAKG: Hierarchy-Aware Knowledge Gated Network for Recommendation 【层次化知识门控网络】
- KETCH: Knowledge Graph Enhanced Thread Recommendation in Healthcare Forums 【医疗论坛上的知识图谱增强的推荐】

### News Recommendation

- ProFairRec: Provider Fairness-aware News Recommendation 【商家公平的新闻推荐】
- Positive, Negative and Neutral: Modeling Implicit Feedback in Session-based News Recommendation 【建模隐式反馈】
- FUM: Fine-grained and Fast User Modeling for News Recommendation 【short paper，细粒度快速的用户建模】
- Is News Recommendation a Sequential Recommendation Task? 【short paper，新闻推荐是序列化推荐吗】
- News Recommendation with Candidate-aware User Modeling 【short paper，候选感知的用户建模】
- MM-Rec: Visiolinguistic Model Empowered Multimodal News Recommendation 【short paper，视觉语言学增强的多模态新闻推荐】

### other task

- CAPTOR: A Crowd-Aware Pre-Travel Recommender System for Out-of-Town Users 【为乡村用户提供的旅游推荐】
- PERD: Personalized Emoji Recommendation with Dynamic User Preference 【short paper，个性化表情推荐】
- Item Similarity Mining for Multi-Market Recommendation 【short paper，多市场推荐中的商品相似度挖掘】
- A Content Recommendation Policy for Gaining Subscribers 【short paper，为提升订阅者的内容推荐策略】
- Thinking inside The Box: Learning Hypercube Representations for Group Recommendation 【超立方体表示用于组推荐】

## -按照主要技术划分

### -1 GNN-based

- Hierarchical Multi-Task Graph Recurrent Network for Next POI Recommendation 【多任务图循环网络】
- An Attribute-Driven Mirroring Graph Network for Session-based Recommendation 【特征驱动的反射图网络】
- Co-clustering Interactions via Attentive Hypergraph Neural Network 【超图神经网络聚类交互】
- Graph Trend Filtering Networks for Recommendation 【图趋势过滤网络】
- EFLEC: Efficient Feature-LEakage Correction in GNN based Recommendation Systems 【short paper，高效的特征泄露修正】
- DH-HGCN: Dual Homogeneity Hypergraph Convolutional Network for Multiple Social Recommendations 【short paper，双同质超图卷积网络】
- Enhancing Hypergraph Neural Networks with Intent Disentanglement for Session-based Recommendation【short paper，意图解耦增强超图神经网络】
- DAGNN: Demand-aware Graph Neural Networks for Session-based Recommendation 【short paper， 需求感知的图神经网络】

### -2 RL-based

- Locality-Sensitive State-Guided Experience Replay Optimization for Sparse-Reward in Online Recommendation 【在线推荐中的稀疏奖励问题】
- Multi-Agent RL-based Information Selection Model for Sequential Recommendation 【多智能体信息选择】
- Rethinking Reinforcement Learning for Recommendation: A Prompt Perspective 【从提示视角看用于推荐的强化学习】
- Doubly-Adaptive Reinforcement Learning for Cross-Domain Interactive Recommendation 【双重适应的强化学习】
- MGPolicy: Meta Graph Enhanced Off-policy Learning for Recommendations 【元图增强的离线策略学习】
- Value Penalized Q-Learning for Recommender Systems 【short paper，值惩罚的Q-Learning】
- Revisiting Interactive Recommender System with Reinforcement Learning 【short paper，回顾基于强化学习的交互推荐】

### -3 Contrastive Learning based

- A Review-aware Graph Contrastive Learning Framework for Recommendation 【考虑评论的图对比学习】
- Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation 【简单的图对比学习方法】
- Knowledge Graph Contrastive Learning for Recommendation 【知识图谱上的对比学习】
- Self-Augmented Recommendation with Hypergraph Contrastive Collaborative Filtering 【超图上的对比学习】
- Multi-level Cross-view Contrastive Learning for Knowledge-aware Recommender System 【多级交叉视图的对比学习】
- Dual Contrastive Network for Sequential Recommendation 【short paper，双对比网络】
- Improving Micro-video Recommendation via Contrastive Multiple Interests 【short paper，对比多兴趣提升短视频推荐】
- An MLP-based Algorithm for Efficient Contrastive Graph Recommendations 【short paper，基于MLP的算法实现高效图对比】
- Multi-modal Graph Contrastive Learning for Micro-video Recommendation 【short paper，多模态图对比学习】
- Towards Results-level Proportionality for Multi-objective Recommender Systems 【short paper，动量对比方法】
1- Socially-aware Dual Contrastive Learning for Cold-Start Recommendation 【short paper，社交感知的双重对比学习】

### -4 AutoML-based Recommender System

- Single-shot Embedding Dimension Search in Recommender System 【嵌入维度搜索】
- AutoLossGen: Automatic Loss Function Generation for Recommender Systems 【自动损失函数生成】
- NAS-CTR: Efficient Neural Architecture Search for Click-Through Rate Prediction 【高效的网络结构搜索】

### -5 Others

- Forest-based Deep Recommender 【深度森林】
- Deployable and Continuable Meta-Learning-Based Recommender System with Fast User-Incremental Updates 【基于元学习的可部署可拓展推荐系统】

## -按照研究话题划分

### -1 Bias/Debias in Recommender System

- Interpolative Distillation for Unifying Biased and Debiased Recommendation 
- Co-training Disentangled Domain Adaptation Network for Leveraging Popularity Bias in Recommenders 【训练解耦的域适应网络来利用流行度偏差】
- Bilateral Self-unbiased Recommender Learning from Biased Implicit Feedback 【双边去偏】
- Mitigating Consumer Biases in Recommendations with Adversarial Training 【short paper，对抗训练去偏】
- Neutralizing Popularity Bias in Recommendation Models 【short paper，中和流行度偏差】
- DeSCoVeR: Debiased Semantic Context Prior for Venue Recommendation 【short paper，在场所推荐中去除语义上下文先验】

### -2 Explanation in Recommender System

- Post Processing Recommender Systems with Knowledge Graphs for Recency, Popularity, and Diversity of Explanations 【使用知识图谱为推荐生成崭新的、多样的解释】
- PEVAE: A hierarchical VAE for personalized explainable recommendation. 【利用层次化VAE进行个性化可解释推荐】
- Explainable Session-based Recommendation with Meta-Path Guided Instances and Self-Attention Mechanism 【short paper， 基于元路径指导和自注意力机制的可解释会话推荐】 

### -3 Long-tail/Cold-start in Recommender System

- Socially-aware Dual Contrastive Learning for Cold-Start Recommendation 【short paper，社交感知的双重对比学习】
- Transform Cold-Start Users into Warm via Fused Behaviors in Large-Scale Recommendation 【short paper，通过融合行为转换冷启动用户】
- Generative Adversarial Framework for Cold-Start Item Recommendation 【short paper，针对冷启动商品的生成对抗框架】
- Improving Item Cold-start Recommendation via Model-agnostic Conditional Variational Autoencoder 【short paper，模型无关的自编码器提升商品冷启动推荐】

### -4 Fairness in Recommender System

- Joint Multisided Exposure Fairness for Recommendation 【综合考虑多边的曝光公平性】
- ProFairRec: Provider Fairness-aware News Recommendation 【商家公平的新闻推荐】
- CPFair: Personalized Consumer and Producer Fairness Re-ranking for Recommender Systems 【用户和商家公平的重排序】
- Explainable Fairness for Feature-aware Recommender Systems 【考虑特征的推荐系统中的可解释公平】
- Selective Fairness in Recommendation via Prompts 【short paper，通过提示保证可选的公平性】
- Regulating Provider Groups Exposure in Recommendations 【short paper，调整商家组曝光】

### -5 Diversity in Recommender System

- DAWAR: Diversity-aware Web APIs Recommendation for Mashup Creation based on Correlation Graph 【多样化Web API推荐】
- Mitigating the Filter Bubble while Maintaining Relevance: Targeted Diversification with VAE-based Recommender Systems 【short paper，定向多样化】
- Diversity vs Relevance: a practical multi-objective study in luxury fashion recommendations 【short paper，奢侈品推荐中的多目标研究】

### -6 Attack/Denoise in Recommender System

- Learning to Denoise Unreliable Interactions for Graph Collaborative Filtering 【数据去噪】
- Less is More: Reweighting Important Spectral Graph Features for Recommendation 【评估重要的图谱特征】
- Denoising Time Cycle Modeling for Recommendation 【short paper，去噪时间循环建模】
- Adversarial Graph Perturbations for Recommendations at Scale 【short paper，大规模推荐中的对抗图扰动】

### -7Others

- Privacy-Preserving Synthetic Data Generation for Recommendation 【隐私保护的仿真数据生成】
- User-Aware Multi-Interest Learning for Candidate Matching in Recommenders 【使用用户多兴趣学习进行候选匹配】
- User-controllable Recommendation Against Filter Bubbles 【用户可控的推荐】
- Rethinking Correlation-based Item-Item Similarities for Recommender Systems 【short paper，反思基于关系的商品相似度】
- ReLoop: A Self-Correction Learning Loop for Recommender Systems 【short paper，推荐系统中的自修正循环学习】
- Towards Results-level Proportionality for Multi-objective Recommender Systems 【short paper，结果均衡的多目标推荐系统】

## -其他研究方向

### -1 QA

- DGQAN: Dual Graph Question-Answer Attention Networks for Answer Selection 【双图注意力网络】
- Counterfactual Learning To Rank for Utility-Maximizing Query Autocompletion 【反事实学习】
- PTAU: Prompt Tuning for Attributing Unanswerable Questions 【提示微调】
- Conversational Question Answering on Heterogeneous Sources 【异质来源的问答】
- A Non-Factoid Question-Answering Taxonomy
- QUASER: Question Answering with Scalable Extractive Rationalization
- Detecting Frozen Phrases in Open-Domain Question Answering 【short paper 在开放域问答中检测固定短语】
- Answering Count Query with Explanatory Evidence 【short paper】

### -1 Knowledge Graph

- Hybrid Transformer with Multi-level Fusion for Multimodal Knowledge Graph Completion 【多模态知识图谱补全】
- Incorporating Context Graph with Logical Reasoning for Inductive Relation Prediction 【合并上下文图和逻辑推理进行归纳式关系预测】
- Meta-Knowledge Transfer for Inductive Knowledge Graph Embedding【元知识迁移解决归纳式知识图谱嵌入】
- Re-thinking Knowledge Graph Completion Evaluation from an Information Retrieval Perspective 【从信息检索视角思考知识图谱补全的评测】
- Relation-Guided Few-Shot Relational Triple Extraction 【short paper，关系指导的few-shot三元组抽取】

### -2 Conversation/ Dialog

- Unified Dialog Model Pre-training for Task-Oriented Dialog Understanding and Generation 【统一对话理解和生成的预训练模型】
- Interacting with Non-Cooperative User: A New Paradigm for Proactive Dialogue Policy 【主动对话策略的新范式】
- COSPLAY: Concept Set Guided Personalized Dialogue System 【概念集合指导的个性化对话系统】
- Understanding User Satisfaction with Task-Oriented Dialogue Systems 【理解用户满意度】
- A Multi-Task Based Neural Model to Simulate Users in Goal Oriented Dialogue Systems 【short paper 多任务模型仿真用户】
- Task-Oriented Dialogue System as Natural Language Generation 【short paper，自然语言生成的对话系统】

### -3 Summarization

- HTKG: Deep Keyphrase Generation with Neural Hierarchical Topic Guidance
- V2P: Vision-to-Prompt based Multi-Modal Product Summary Generation
- Unifying Cross-lingual Summarization and Machine Translation with Compression Rate  【使用压缩率统一跨语言总结和机器翻译】
- ADPL: Adversarial Prompt-based Domain Adaptation for Dialogue Summarization with Knowledge Disentanglement 【基于提示的对抗领域自适应】
- Summarizing Legal Regulatory Documents using Transformers 【short ，使用Transformers总结法律监管文档】
- QSG Transformer: Transformer with Query-Attentive Semantic Graph for Query-Focused Summarization 【short paper】
- MuchSUM: Multi-channel Graph Neural Network for Extractive Summarization 【short paper，多通道图神经网络】
- Lightweight Meta-Learning for Low-Resource Abstractive Summarization 【short paper， 轻量级元学习】
- Extractive Elementary Discourse Units for Improving Abstractive Summarization 【short paper】

### -4 Multi-Modality

- Tag-assisted Multimodal Sentiment Analysis under Uncertain Missing Modalities
- Progressive Learning for Image Retrieval with Hybrid-Modality Queries
- CenterCLIP: Token Clustering for Efficient Text-Video Retrieval
- Multimodal Entity Linking with Gated Hierarchical Fusion and Contrastive Training
- CRET: Cross-Modal Retrieval Transformer for Efficient Text-Video Retrieval
- Bit-aware Semantic Transformer Hashing for Multi-modal Retrieval 
- Video Moment Retrieval from Text Queries via Single Frame Annotation
- Multimodal Disentanglement Variational AutoEncoders for Zero-Shot Cross-Modal Retrieval
- A Multitask Framework for Sentiment, Emotion and Sarcasm aware Cyberbullying Detection in Multi-modal Code-Mixed Memes
- Animating Images to transfer CLIP for Video-Text Retrieval 【short paper， 使用CLIP进行视频-文本检索】
1- Image-Text Retrieval via Contrastive Learning with Auxiliary Generative Features and Support-set Regularization 【short paper】
1- An Efficient Fusion Mechanism for Multimodal Low-resource Setting 【short paper，在低资源下的一种高效融合机制】

### -5 Generation

- Mutual Disentanglement Learning for Joint Fine-Grained Sentiment Classification and Controllable Text Generation
- Target-aware Abstractive Related Work Generation with Contrastive Learning 【利用对比学习生成生成相关工作】
- Generating Clarifying Questions with Web Search Results 【利用Web搜索结果生成清晰问题】
- Choosing The Right Teammate For Cooperative Text Generation 【short paper 】

### -6 Representation Learning

- Structure and Semantics Preserving Document Representations 【保留结构和语义的文档表示】
- Unsupervised Belief Representation Learning with Information-Theoretic Variational Graph Auto-Encoders