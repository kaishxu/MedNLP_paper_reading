# MedNLP paper reading
学习记录。

## Dataset

**[emrKBQA: A Clinical Knowledge-Base Question Answering Dataset](https://aclanthology.org/2021.bionlp-1.7/)** (BioNLP 2021 workshop)

设计有点复杂，脑子清醒再看。**（数据集未公开）**

**[MedDialog: Large-scale Medical Dialogue Datasets](https://aclanthology.org/2020.emnlp-main.743/)** (EMNLP 2020)

医疗对话数据集，包括中英文，中文340万条对话，英文25万条对话。数据集核心内容除多轮对话外，还包括病情描述、病人基本情况等信息。**（[代码](https://github.com/UCSD-AI4H/Medical-Dialogue-System)）**

**[MedDG: A Large-scale Medical Consultation Dataset for Building Medical Dialogue System](https://arxiv.org/abs/2010.07497)**

医疗对话数据集，仅包含中文，共1.7万条对话。其中每条对话都有实体标注，涉及诱因、疾病、症状、检测和药物5种类型共160个实体标注。**（[代码](https://github.com/lwgkzl/MedDG)）**

**[On the Generation of Medical Dialogues for COVID-19](https://arxiv.org/abs/2005.05442)** (ACL 2021)

关于COVID-19的医疗对话数据集，包括中英文，中文1万条对话，英文1千条对话。**（[代码](https://github.com/UCSD-AI4H/COVID-Dialogue)）**

## Pretraining

[**Conceptualized Representation Learning for Chinese Biomedical Text Mining**](https://arxiv.org/abs/2008.10813) (WSDM 2020 Health Day)

文章提出了一种显式融合医学实体和**医学短语**的预训练方法。整体思路同ERNIE1.0，在训练阶段对相关实体和短语进行mask。唯一的亮点是额外训练了一个短语分类器，用于区分医学转述（e.g.，“肚子有一点痛”==“胃疼”），并且训练时不仅融合医学实体，还融合医学短语。**（整体感觉是ERNIE的医学version，最大的贡献是发布了ChineseBLUE，[代码](https://github.com/alibaba-research/ChineseBLUE)）**

**[Improving Biomedical Pretrained Language Models with Knowledge](https://arxiv.org/abs/2104.10344)** (BioNLP 2021 workshop)

文章设计了一个KG-enhanced的预训练模型，主体结构为两个transformers堆叠，第一个transformers的输出与KG（用TransE）相融合，生成第二个transformers的输入。作者把KG信息attend到entity所对应的transformers输出上，对于任意entity，attend的KG涉及最近的k个entity（如何获得这k个entity没看懂，应该是图上最近的k个entity）。预训练任务除了MLM，还包括了entity的NER任务，以及KG和transformers表示的linking任务（使同一个entity在两者的表示相近）。**（效果一言难尽，因为最后测试的还是NER任务，并且实验未给出entity masking和KG融合哪个作用比较大，直觉是KG并没有发挥什么作用，[代码](https://github.com/GanjinZero/KeBioLM)）**

**[Infusing Disease Knowledge into BERT for Health Question Answering, Medical Inference and Disease Name Recognition](https://arxiv.org/abs/2010.03746)** (EMNLP 2020)

为了融合knowledge到已有的pretrain model中，文章设计了一个弱监督的训练方法（原文用weakly-supervised，类似于Prompt机制）并构建了基于Wikipedia的entity-passage对数据。其方法核心是在医学文本的前部插入辅助问句（e.g.，“What is the [Aspect] of [Disease]?”），在训练阶段将aspect和disease名称mask掉构建相应的分类任务，比较tricky的是loss不仅用了常规cross entropy，还补充了未经softmax下的相应输出的监督。实验结果非常好，该融合思路在大部分BERT系列模型上都取得了提升。**（重点是设计了Prompt策略，简单且有效，[代码](https://github.com/heyunh2015/diseaseBERT)）**

**[SMedBERT: A Knowledge-Enhanced Pre-trained Language Model with Structured Semantics for Medical Text Mining](https://aclanthology.org/2021.acl-long.457/)** (ACL 2021)

文章构建了一个预训练模型将结构化医疗语义知识融入BERT中。细节上，主要包含Mention-neighbor Hybrid Attention和Mention-neighbor Context Modeling，其中前者为hierarchical的attention，先按节点类型attend，再按具体节点attend，最后用gate除噪，后者为预训练辅助任务，包括Masked Neighbor Modeling（目前没看懂，涉及[TransR](https://dl.acm.org/doi/10.5555/2886521.2886624)）和Masked Mention Modeling（不理解为什么做这个任务）。另外context相关entity的获取还用到了PageRank。**（整体思路很明确，但是每个细节都很复杂，有些模块没什么依据，[代码](https://github.com/MatNLP/SMedBERT)）**

**[UmlsBERT: Clinical Domain Knowledge Augmentation of Contextual Embeddings Using the Unified Medical Language System Metathesaurus](https://arxiv.org/abs/2010.10391)** (NAACL 2021)

文章构建了一个医学知识增强的预训练模型，其通过引入UMLS语义类型（e.g.，心脏的语义类型为身体部位、器官）增强原有BERT对医学知识的表示，另外修改了softmax形式的loss为二元多标签loss。**（思路很常规，实验效果一般，[代码](https://github.com/gmichalo/UmlsBERT)）**

## Dialogue

**[End-to-End Knowledge-Routed Relational Dialogue System for Automatic Diagnosis](https://arxiv.org/abs/1901.10623)** (AAAI 2019)

涉及RL，待会看。**（[代码](https://github.com/HCPLab-SYSU/Medical_DS)）**

**[MIE: A Medical Information Extractor towards Medical Dialogues](https://aclanthology.org/2020.acl-main.576/)** (ACL 2020)

文章构建了一个window-to-information标注的医疗对话数据集，具体为某个sliding window内的对话内容与数个entity（包含状态，例如检测阳性：pos）相对应，不涉及entity内具体token的标注。文章设计了一个匹配模型以计算某个dialogue片段与任意entity的得分，其中dialogue和entity的encoder均为Bi-LSTM，两者用attention进行交互以获取融合后的表示，并通过sigmoid得到最终匹配分。**（亮点在于用ranking方法解决entity抽取，非常有趣，不过用的backbone很复古，[代码](https://github.com/nlpir2020/MIE-ACL-2020)）**

**[Graph-Evolving Meta-Learning for Low-Resource Medical Dialogue Generation](https://arxiv.org/abs/2012.11988)** (AAAI 2021)

文章主要提出了一个基于meta-learning的训练框架用于解决医疗场景下的low-resource问题，其中Graph-Evolving指将对话中的两个共现entity进行连接以弥补已有Commonsense Graph在医疗领域的不足。具体实现上，模型用了hierarchical的encoder分别编码句子和整段对话，句子编码单独构成graph并和entity graph进行融合，entity graph用GAT方式学习，而对话编码则用于decoder的初始化。最终decoder涉及词表和entity graph的条件概率。**（用meta-learning解决long-tail或者low-resource是个很有意思的点；graph-evole很tricky，如果只是共现，且和embedding无关，其实都不需要evolve，直接在最开始根据所有dialogue把图补全就好了，目前不知道作者具体怎么实现的；文章撰写极其糟糕，句子编码的graph如何融合到entity graph中完全没有说明，decoder输入以及输出的概率完全没有说明，[代码](https://github.com/ha-lins/GEML-MDG)）**

**[Extracting Appointment Spans from Medical Conversations](https://aclanthology.org/2021.nlpmc-1.6/)** (NLPMC 2021 workshop)

文章构建了一个弱监督的序列标注数据集，目的是从医疗对话文本中抽取appointment原因和时间。**（没什么实质内容，无代码）**

## Summarization

[**Attend to Medical Ontologies: Content Selection for Clinical Abstractive Summarization**](https://arxiv.org/abs/2005.00163) (ACL 2020)

为了更精确识别医学实体（文中用ontology term指代），文章构建了一个Bi-LSTM+BERT的content selector对输入文本中的ontology term进行0/1标注（表示term是否被复制到输出文本中）。模型包括原始输入文本和ontology term两个LSTM-based encoder，用ontology的最终state增强每个原始输入文本的state。另外decoder没说清楚。**（整体思路常规，无代码）**

[**A Gradually Soft Multi-Task and Data-Augmented Approach to Medical Question Understanding**](https://aclanthology.org/2021.acl-long.119/) (ACL 2021)

文章提出了一个gradually soft的多任务学习框架，该框架固定不同任务的encoder参数hard sharing（完全共享），而decoder参数gradually soft sharing（即模型层面上不共享，但通过loss约束每层的参数相似性，该文章期望层数越大，参数差异越大）。关于data augmentation，作者首先说明了summarization任务和recognizing question entailment（问题蕴含识别）任务的一致性，然后通过将S任务的数据改写成R任务的数据，将R任务的数据改写成S任务的数据，从而丰富各自任务的数据，构建领域一致的数据集。实验结果显示这两个改进都能带来一定程度提升。**（亮点在于gradually soft，而data augmentation属于锦上添花，并且方法通用性很强，并没有依赖medical相关知识，但是这文章写的实在是恶心，尤其数据增强部分，明明就很简单却要描述的那么绕，无[代码](https://github.com/KhalilMrini/Medical-Question-Understanding)）**

## Reading Comprehension

**[Knowledge-Empowered Representation Learning for Chinese Medical Reading Comprehension: Task, Model and Resources](https://aclanthology.org/2021.findings-acl.197.pdf)** (ACL Findings 2021)

文章设计了一个CMedMRC任务以及数据集（形式同SQuAD，e.g.，**Passage**-**Question**-**Answer**-**Support sentence**），总结了medical MRC的几个难点：长尾术语（出现频率非常低的token）、书面vs口头表述差异、术语组合以及术语转述（个人认为这几个难点算medical NLP的难点，不单单只是MRC）。 文章设计了一个BERT-based的answer extraction模型，核心是KB representation（用PTransE）和BERT representation的融合，在常规方法（把KB信息attend到BERT output上）上叠加一个过滤器（Gated Loop Layer），该过滤器以迭代的方式不断用KB信息精炼和融合当前BERT output。**（整体更像一个QA任务，比较有意思的点在于迭代精炼，但作者并没有做相关ablation实验，效果不得而知，无[代码](https://github.com/MatNLP/CMedMRC)）**

**[Towards Medical Machine Reading Comprehension with Structural Knowledge and Plain Text](https://aclanthology.org/2020.emnlp-main.111/)** (EMNLP 2020)

文章根据中国国家执业药师资格考试设计了一个问答数据集，形式为五选一的选择题。为有效融合医学知识到已有模型，文章主要采取了三种思路：1. 构建**intermediate-task**预训练，将CMeKG中的（h，r，t）三元组以“[CLS] [h] [SEP] [t]”的形式组成输入，并构建基于BERT的分类器区分不同的r；2. 将（h，r，t）转化成“The [h] of [r] is [t]”的形式，并与question中的entity计算word mover距离，筛选TOP-K的（h，r，t）三元组；3. 用GCN强化**思路2**中的entity表示（如果我没理解错，事实上原文中并没有详细描述该操作的实现）。另外，模型在输入阶段通过IR的方式获取了TOP-N的evidence进行输入增强。实验结果上显示，**思路1**的分类器和**思路3**的GCN强化效果最不显著。**（重点是IR的引入，在多个步骤中用ranking的方式获取更多信息从而增强输入，最后的GCN像是凑数的，没什么用，无代码，[demo](http://112.74.48.115:8157/)）**

## NER

**[A Neural Transition-based Joint Model for Disease Named Entity Recognition and Normalization](https://aclanthology.org/2021.acl-long.219/)** (ACL 2021)

文章提出了一个神经网络构建的基于状态转移的NER及entity normalization模型，目的在于模拟一种用于NER的状态转移算法（无需token标记）。文章将状态分为recognition（OUT，SHIFT，REDUCE，SEGMENT-t）和normalization（LINKING-c）两大类，并用不同模型进行表示，最终用softmax获取各状态的最大概率。**（虽然文章强调的贡献是end-to-end地解决NER和normalization两个任务，但用神经网络模拟状态转移比较有意思。文章有些地方说的不太清楚，e.g.，在norm中涉及多个且数量不一致的concept，最后如何计算softmax，在segment的时候需要输出entity类型，原文没有说明，无代码）**

**[An End-to-End Progressive Multi-Task Learning Framework for Medical Named Entity Recognition and Normalization](https://aclanthology.org/2021.acl-long.485/)** (ACL 2021)

文章提出了一个渐进式的多任务学习框架，该框架将NER和NEN任务分为low-level、mid-level和high-level子任务，分别为基于pretrain model的token序列标注、KG融合（用pretrain model生成entity表示）的粗粒度mention蕴含和KG融合的细粒度mention token序列标注。**（整体思路比较简单顺滑，[代码](https://github.com/zhoubaohang/E2EMERN)）**

## Others

**[Counterfactual Supporting Facts Extraction for Explainable Medical Record Based Diagnosis with Graph Network](https://aclanthology.org/2021.naacl-main.156/)** (NAACL 2021)

没看懂，正二刷。**（无[代码](https://github.com/CKRE/CMGE)）**

**[Rationalizing Medical Relation Prediction from Corpus-level Statistics](https://arxiv.org/abs/2005.00889)** (ACL 2020)

文章参照人类记忆的认知过程，构建了Recall、Recognition和Pred三模块的医学关系预测模型。Recall模块学习语料中的entity共现，Recognition模块对给定的head和tail及其相关entity pair（e.g.，i和j），首先学习i和j潜在relation的加权表示，再汇总得到节点和关系的综合表示eij，Pred模块将通过不同相关entity pair的表示进行融合得到最终表示，并通过sigmoid得到head和tail的相关性预测。**（文章亮点一方面在于利用语料库的entity共现补全原有KG的信息，另一方面在于提出了一种可解释性的关系预测框架，可提供待预测关系关联与否的相关关系依据，[代码](https://github.com/zhenwang9102/X-MedRELA)）**

**[Zero-shot Medical Entity Retrieval without Annotation: Learning From Rich Knowledge Graph Semantics](https://arxiv.org/abs/2105.12682)** (ACL 2021 short)

文章提出了一个基于medical KG的zero-shot实体提取框架。核心思路是利用KG中涉及的相同实体的同义表达，以及实体间的图形结构，构建双塔结构（Bi-encoder）的对比学习模型，从而对任意未标注的mention获取top-k相关的KG entity。文章用到了ICD-10、SNOMED和UMLS三种medical KG，ICD-10为树结构，parent和child拼接构成entity，SNOMED和UMLS为有向图，涉及多种关系。**（整体思路挺常规，和句子相似匹配、dense passage retrieval相近，无代码）**

**[Self-Alignment Pretraining for Biomedical Entity Representations](https://arxiv.org/abs/2010.11784)** (NAACL 2021)

文章整体思路同[Zero-shot Medical Entity Retrieval without Annotation: Learning From Rich Knowledge Graph Semantics](https://arxiv.org/abs/2105.12682)，核心是利用对比学习模型，从KG中的相同实体的同义表达中学习该实体的向量表示。其中正负例选择采用online hard pair mining，将较容易正例剔除，loss采用multi-similarity。**（整体思路和我被拒的文章相似，不过这个multi-similarity会有奇效吗？[代码](https://github.com/cambridgeltl/sapbert)）**

[**Clustering-based Inference for Biomedical Entity Linking**](https://aclanthology.org/2021.naacl-main.205/) (NAACL 2021)

文章提出了一种基于clustering的entity linking推理机制。首先用两个BERT模型分别构建mention-mention和mention-entity的评价模型，之后通过single-linkage层次聚类算法将最相关的mention-mention或mention-entity进行连接，并且每个cluster只能包含一个entity。**（文章的想法挺有意思，相比于直接判断mention和entity的关系，利用mention之间的联系简介link到entity可以充分利用context的信息，不过有个明显的问题是两个BERT-based的评价模型如何一致（两者分数可能会有较大差异，无法平衡），无代码）**

**[FedED: Federated Learning via Ensemble Distillation for Medical Relation Extraction](https://aclanthology.org/2020.emnlp-main.165/)** (EMNLP 2020)

**（无代码）**

## Be still updating...

