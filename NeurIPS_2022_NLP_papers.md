# Catalog(目录)
- [Model 【模型】](#model-模型)
  - [Model Design 【模型设计】](#model-design-模型设计)
  - [Model Compression 【模型压缩】](#model-compression-模型压缩)
  - [Model Training 【模型训练】](#model-training-模型训练)
  - [Model Usage 【模型使用】](#model-usage-模型使用)
- [Interpretability, Analysis and Evaluation 【可解释性、分析、评测】](#interpretability-analysis-and-evaluation-可解释性分析评测)
- [Robustness and Safety 【鲁棒性与安全】](#robustness-and-safety-鲁棒性与安全)
- [knowledge and reasoning 【知识与推理】](#knowledge-and-reasoning-知识与推理)
- [Information Extraction 【信息抽取】](#information-extraction-信息抽取)
- [Information Retrieval 【信息检索】](#information-retrieval-信息检索)
- [Text Classification 【文本分类】](#text-classification-文本分类)
- [Text Generation 【文本生成】](#text-generation-文本生成)
- [Machine Translation and Multilinguality 【机器翻译与多语言】](#machine-translation-and-multilinguality-机器翻译与多语言)
- [Multimodality 【多模态】](#multimodality-多模态)
- [Special Tasks 【特殊任务】](#special-tasks-特殊任务)
  - [Code 【代码】](#code-代码)
  - [Mathematics 【数学】](#mathematics-数学)
  - [Others 【其他】](#others-其他)

## Model 【模型】
### Model Design 【模型设计】
- Recurrent Memory Transformer
- Jump Self-attention: Capturing High-order Statistics in Transformers
- Block-Recurrent Transformers
- Staircase Attention for Recurrent Processing of Sequences
- Non-Linguistic Supervision for Contrastive Learning of Sentence Embeddings
- Transcormer: Transformer for Sentence Scoring with Sliding Language Modeling
- Mixture-of-Experts with Expert Choice Routing
- On the Representation Collapse of Sparse Mixture of Experts
- Improving Transformer with an Admixture of Attention Heads
- Your Transformer May Not be as Powerful as You Expect
- Confident Adaptive Language Modeling
- Decoupled Context Processing for Context Augmented Language Modeling
- Unsupervised Cross-Task Generalization via Retrieval Augmentation
- Revisiting Neural Scaling Laws in Language and Vision
- Learning to Scaffold: Optimizing Model Explanations for Teaching

### Model Compression 【模型压缩】
- Information-Theoretic Generative Model Compression with Variational Energy-based Model
- Towards Efficient Post-training Quantization of Pre-trained Language Models
- Outlier Suppression: Pushing the Limit of Low-bit Transformer Language Models
- Deep Compression of Pre-trained Transformer Models
- LiteTransformerSearch: Training-free On-device Search for Efficient Autoregressive Language Models
- GPT3.int8(): 8-bit Matrix Multiplication for Transformers at Scale
- MorphTE: Injecting Morphology in Tensorized Embeddings
- Few-shot Task-agnostic Neural Architecture Search for Distilling Large Language Models
- A Fast Post-Training Pruning Framework for Transformers

### Model Training 【模型训练】
- Memorization Without Overfitting: Analyzing the Training Dynamics of Large Language Models
- Generating Training Data with Language Models: Towards Zero-Shot Language Understanding
- A Data-Augmentation Is Worth A Thousand Samples
- TokenMixup: Efficient Attention-guided Token-level Data Augmentation for Transformers
- The Stability-Efficiency Dilemma: Investigating Sequence Length Warmup for Training GPT Models
- Tempo: Accelerating Transformer-Based Model Training through Memory Footprint Reduction
- Training and Inference on Any-Order Autoregressive Models the Right Way
- Decentralized Training of Foundation Models in Heterogeneous Environments

### Model Usage 【模型使用】
- The Unreliability of Explanations in Few-Shot In-Context Learning
- What Can Transformers Learn In-Context? A Case Study of Simple Function Classes
- Decoupling Knowledge from Memorization: Retrieval-augmented Prompt Learning
- Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning
- Training language models to follow instructions with human feedback
- LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning
- How to talk to your model: Instructions, descriptions, and learning
- Data Distributional Properties Drive Emergent In-Context Learning in Transformers
- Sparse Structure Search for Parameter-Efficient Tuning
- Fine-Tuning Pre-Trained Language Models Effectively by Optimizing Subnetworks Adaptively
- Second Thoughts are Best: Learning to Re-Align With Human Values from Text Edits
- LIFT: Language-Interfaced FineTuning for Non-language Machine Learning Tasks
- Adapting to Domain Shift by Meta-Distillation from Mixture-of-Experts

## Interpretability, Analysis and Evaluation 【可解释性、分析、评测】
- CEBaB: Estimating the Causal Effects of Real-World Concepts on NLP Model Behavior
- Rule-Based but Flexible? Evaluating and Improving Language Models as Accounts of Human Moral Judgment
- Understanding the Failure of Batch Normalization for Transformers in NLP
- AttCAT: Explaining Transformers via Attentive Class Activation Tokens
- An empirical analysis of compute-optimal large language model training
- Why GANs are overkill for NLP
- Exploring Length Generalization in Large Language Models
- Capturing Failures of Large Language Models via Human Cognitive Biases
- Pre-Trained Model Reusability Evaluation for Small-Data Transfer Learning
- First is Better Than Last for Language Data Influence
- What are the best Systems? New Perspectives on NLP Benchmarking
- Characteristics of Harmful Text: Towards Rigorous Benchmarking of Language Models
- FETA: Towards Specializing Foundational Models for Expert Task Applications
- This is the way - lessons learned from designing and compiling LEPISZCZE, a comprehensive NLP benchmark for Polish
- Rethinking Knowledge Graph Evaluation Under the Open-World Assumption
- A Multi-Task Benchmark for Korean Legal Language Understanding and Judgement Prediction
  
## Robustness and Safety 【鲁棒性与安全】
- Active Learning Helps Pretrained Models Learn the Intended Task
- Improving Certified Robustness via Statistical Learning with Logical Reasoning
- Moderate-fitting as a Natural Backdoor Defender for Pre-trained Language Models
- BadPrompt: Backdoor Attacks on Continuous Prompts
- A Win-win Deal: Towards Sparse and Robust Pre-trained Language Models
- Exploring the Limits of Domain-Adaptive Training for Detoxifying Large-Scale Language Models
- AD-DROP: Attribution Driven Dropout for Robust Language Model Finetuning
- Large (robust) models from computational constraints
- Multitasking Models are Robust to Structural Failure: A Neural Model for Bilingual Cognitive Reserve
- A Unified Evaluation of Textual Backdoor Learning: Frameworks and Benchmarks
- Recovering Private Text in Federated Learning of Language Models
- LAMP: Extracting Text from Gradients with Language Model Priors
- SeqPATE: Differentially Private Text Generation via Knowledge Distillation
- Differentially Private Model Compression
- Federated Learning from Pre-Trained Models: A Contrastive Learning Approach

## knowledge and reasoning 【知识与推理】
- Learning to Sample and Aggregate: Few-shot Reasoning over Temporal Knowledge Graph
- Retaining Knowledge for Learning with Dynamic Definition
- Shadow Knowledge Distillation: Bridging Offline and Online Knowledge Transfer
- What Makes a "Good" Data Augmentation in Knowledge Distillation - A Statistical Perspective
- Learning to Reason with Neural Networks: Generalization, Unseen Data and Boolean Measures
- Roadblocks for Temporarily Disabling Shortcuts and Learning New Knowledge
- PALBERT: Teaching ALBERT to Ponder
- Locating and Editing Factual Associations in GPT
- OTKGE: Multi-modal Knowledge Graph Embeddings via Optimal Transport
- Large Language Models are Zero-Shot Reasoners
- STaR: Bootstrapping Reasoning With Reasoning
- Chain of Thought Prompting Elicits Reasoning in Large Language Models
- ELASTIC: Numerical Reasoning with Adaptive Symbolic Compiler
- Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering
- Inductive Logical Query Answering in Knowledge Graphs
- Formalizing Coherence and Consistency Applied to Transfer Learning in Neuro-Symbolic Autoencoders
- CoNSoLe: Convex Neural Symbolic Learning
- Deep Bidirectional Language-Knowledge Pretraining
- Neurosymbolic Deep Generative Models for Sequence Data with Relational Constraints
- Instance-based Learning for Knowledge Base Completion
- LogiGAN: Learning Logical Reasoning via Adversarial Pre-training
- Learning robust rule representations for abstract reasoning via internal inferences
- Solving Quantitative Reasoning Problems with Language Models
- Towards Better Evaluation for Dynamic Link Prediction
- Predictive Querying for Autoregressive Neural Sequence Models
- Semantic Probabilistic Layers for Neuro-Symbolic Learning
- End-to-end Symbolic Regression with Transformers
- A Unified Framework for Deep Symbolic Regression
- ZeroC: A Neuro-Symbolic Model for Zero-shot Concept Recognition and Acquisition at Inference Time

## Information Extraction 【信息抽取】
- Unifying Information Extraction with Latent Adaptive Structure-aware Generative Language Model
- TweetNERD - End to End Entity Linking Benchmark for Tweets
- METS-CoV: A Dataset of Medical Entity and Targeted Sentiment on COVID-19 Related Tweets

## Information Retrieval 【信息检索】
- Transformer Memory as a Differentiable Search Index
- Autoregressive Search Engines: Generating Substrings as Document Identifiers
- A Neural Corpus Indexer for Document Retrieval

## Text Classification 【文本分类】
- CascadeXML: End-to-end Multi-Resolution Learning for Extreme Multi-Label Text Classification
- Text Classification with Born's Rule
- Public Wisdom Matters! Discourse-Aware Hyperbolic Fourier Co-Attention for Social Text Classification

## Text Generation 【文本生成】
- CoNT: Contrastive Neural Text Generation
- A Character-Level Length Control Algorithm for Non-Autoregressive Sentence Summarization
- Towards Improving Faithfulness in Abstractive Summarization
- QUARK: Controllable Text Generation with Reinforced Unlearning
- Teacher Forcing Recovers Reward Functions for Text Generation
- Retrieve, Reason, and Refine: Generating Accurate and Faithful Patient Instructions
- A Contrastive Framework for Neural Text Generation
- Learning to Break the Loop: Analyzing and Mitigating Repetitions for Neural Text Generation
- COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics
- Diffusion-LM Improves Controllable Text Generation
- Factuality Enhanced Language Models for Open-Ended Text Generation
- Controllable Text Generation with Neurally-Decomposed Oracle
- InsNet: An Efficient, Flexible, and Performant Insertion-based Text Generation Model
- Relation-Constrained Decoding for Text Generation
- EHRSQL: A Practical Text-to-SQL Benchmark for Electronic Health Records
- TGEA 2.0: A Large-Scale Diagnostically Annotated Dataset with Benchmark Tasks for Text Generation of Pretrained Language Models

## Machine Translation and Multilinguality 【机器翻译与多语言】
- Exploring Non-Monotonic Latent Alignments for Non-Autoregressive Machine Translation
- A new dataset for multilingual keyphrase generation
- Less-forgetting Multi-lingual Fine-tuning
- Losses Can Be Blessings: Routing Self-Supervised Speech Representations Towards Efficient Multilingual and Multitask Speech Processing
- Refining Low-Resource Unsupervised Translation by Language Disentanglement of Multilingual Translation Model
- OccGen: Selection of Real-world Multilingual Parallel Data Balanced in Gender within Occupations
- Multilingual Abusive Comment Detection at Scale for Indic Languages
- The BigScience Corpus A 1.6TB Composite Multilingual Dataset
- Addressing Resource Scarcity across Sign Languages with Multilingual Pretraining and Unified-Vocabulary Datasets

## Multimodality 【多模态】
- REVIVE: Regional Visual Representation Matters in Knowledge-Based Visual Question Answering
- Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning
- GLIPv2: Unifying Localization and Vision-Language Understanding
- VLMo: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts
- A Differentiable Semantic Metric Approximation in Probabilistic Embedding for Cross-Modal Retrieval
- Egocentric Video-Language Pretraining
- Flamingo: a Visual Language Model for Few-Shot Learning
- Language Conditioned Spatial Relation Reasoning for 3D Object Grounding
- Multi-Granularity Cross-modal Alignment for Generalized Medical Visual Representation Learning
- Deep Multi-Modal Structural Equations For Causal Effect Estimation With Unstructured Proxies
- OmniVL: One Foundation Model for Image-Language and Video-Language Tasks
- Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models
- Visual Clues: Bridging Vision and Language Foundations for Image Paragraph Captioning
- TVLT: Textless Vision-Language Transformer
- Divert More Attention to Vision-Language Tracking
- CogView2: Faster and Better Text-to-Image Generation via Hierarchical Transformers
- Text-Adaptive Multiple Visual Prototype Matching for Video-Text Retrieval
- BMU-MoCo: Bidirectional Momentum Update For Continual Video-Language Modeling
- Expectation-Maximization Contrastive Learning for Compact Video-and-Language Representations
- What is Where by Looking: Weakly-Supervised Open-World Phrase-Grounding without Text Inputs
- Flamingo: a Visual Language Model for Few-Shot Learning
- Self-Supervised Multi-Granularity Map Learning for Vision-and-Language Navigation
- UniCLIP: Unified Framework for Contrastive Language-Image Pre-training
- Contrastive Language-Image Pre-Training with Knowledge Graphs
- PyramidCLIP: Hierarchical Feature Alignment for Vision-language Model Pretraining
- Enhancing and Scaling Cross-Modality Alignment for Contrastive Multimodal Pre-Training via Gradient Harmonization
- Mutual Information Divergence: A Unified Metric for Multimodal Generative Models
- Transferring Pre-trained Multimodal Representations with Cross-modal Similarity Matching
- MACK: Multimodal Aligned Conceptual Knowledge for Unpaired Image-text Matching
- HUMANISE: Language-conditioned Human Motion Generation in 3D Scenes
- CyCLIP: Cyclic Contrastive Language-Image Pretraining
- S-Prompts Learning with Pre-trained Transformers: An Occam’s Razor for Domain Incremental Learning
- Delving into OOD Detection with Vision-Language Representations
- Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding
- Language Models with Image Descriptors are Strong Few-Shot Video-Language Learners
- DetCLIP: Dictionary-Enriched Visual-Concept Paralleled Pre-training for Open-world Detection
- Multimodal Contrastive Learning with LIMoE: the Language-Image Mixture of Experts
- Coarse-to-Fine Vision-Language Pre-training with Fusion in the Backbone
- CoupAlign: Coupling Word-Pixel with Sentence-Mask Alignments for Referring Image Segmentation
- Relational Language-Image Pre-training for Human-Object Interaction Detection
- Fine-Grained Semantically Aligned Vision-Language Pre-Training
- Cross-Linked Unified Embedding for cross-modality representation learning
- Quality Not Quantity: On the Interaction between Dataset Design and Robustness of CLIP
- Kernel Multimodal Continuous Attention
- Paraphrasing Is All You Need for Novel Object Captioning
- Long-Form Video-Language Pre-Training with Multimodal Temporal Contrastive Learning
- CLIPDraw: Exploring Text-to-Drawing Synthesis through Language-Image Encoders
- One Model to Edit Them All: Free-Form Text-Driven Image Manipulation with Semantic Modulations
- LGDN: Language-Guided Denoising Network for Video-Language Modeling
- Zero-Shot Video Question Answering via Frozen Bidirectional Language Models
- WinoGAViL: Gamified Association Benchmark to Challenge Vision-and-Language Models
- VLMbench: A Compositional Benchmark for Vision-and-Language Manipulation
- ELEVATER: A Benchmark and Toolkit for Evaluating Language-Augmented Visual Models
- LAION-5B: An open large-scale dataset for training next generation image-text models
- Towards Video Text Visual Question Answering: Benchmark and Baseline
- TaiSu: A 166M Large-scale High-Quality Dataset for Chinese Vision-Language Pre-training
- Wukong: A 100 Million Large-scale Chinese Cross-modal Pre-training Benchmark
- Understanding Aesthetics with Language: A Photo Critique Dataset for Aesthetic Assessment
- Multi-modal Robustness Analysis Against Language and Visual Perturbations
- CLiMB: A Continual Learning Benchmark for Vision-and-Language Tasks
- OrdinalCLIP: Learning Rank Prompts for Language-Guided Ordinal Regression

## Special Tasks 【特殊任务】
### Code 【代码】
- CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning
- Fault-Aware Neural Code Rankers
- NS3: Neuro-symbolic Semantic Code Search
- Pyramid Attention For Source Code Summarization

### Mathematics 【数学】
- HyperTree Proof Search for Neural Theorem Proving
- NaturalProver: Grounded Mathematical Proof Generation with Language Models
- Autoformalization with Large Language Models
- Thor: Wielding Hammers to Integrate Language Models and Automated Theorem Provers

### Others 【其他】
- Measuring and Reducing Model Update Regression in Structured Prediction for NLP
- Learning to Follow Instructions in Text-Based Games
- WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents
- LISA: Learning Interpretable Skill Abstractions from Language
- Inherently Explainable Reinforcement Learning in Natural Language
- Using natural language and program abstractions to instill human inductive biases in machines
- Semantic Exploration from Language Abstractions and Pretrained Representations
- Pre-Trained Language Models for Interactive Decision-Making
- Knowledge-Aware Bayesian Deep Topic Model
- Improving Intrinsic Exploration with Language Abstractions
- Improving Policy Learning via Language Dynamics Distillation
- Meta-Complementing the Semantics of Short Texts in Neural Topic Models
- Pile of Law: Learning Responsible Data Filtering from the Law and a 256GB Open-Source Legal Dataset
- BigBio: A Framework for Data-Centric Biomedical Natural Language Processing