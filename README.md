# MedNLP paper reading
学习记录（纯主观）。

## Pretraining

[**Conceptualized Representation Learning for Chinese Biomedical Text Mining**](https://arxiv.org/abs/2008.10813) (WSDM 2020 Health Day)

文章提出了一种显式融合医学实体和**医学短语**的预训练方法。整体思路同ERNIE1.0，在训练阶段对相关实体和短语进行mask。唯一的亮点是额外训练了一个短语分类器，用于区分医学转述（e.g.，“肚子有一点痛”==“胃疼”），并且训练时不仅融合医学实体，还融合医学短语。**（整体感觉是ERNIE的医学version，最大的贡献是发布了ChineseBLUE，[代码](https://github.com/alibaba-research/ChineseBLUE)）**

**[Improving Biomedical Pretrained Language Models with Knowledge](https://arxiv.org/abs/2104.10344)** (BioNLP 2021 workshop)

文章设计了一个KG-enhanced的预训练模型，主体结构为两个transformers堆叠，第一个transformers的输出与KG（用TransE）相融合，生成第二个transformers的输入。作者把KG信息attend到entity所对应的transformers输出上，对于任意entity，attend的KG涉及最近的k个entity（如何获得这k个entity没看懂，应该是图上最近的k个entity）。预训练任务除了MLM，还包括了entity的NER任务，以及KG和transformers表示的linking任务（使同一个entity在两者的表示相近）。**（效果一言难尽，因为最后测试的还是NER任务，并且实验未给出entity masking和KG融合哪个作用比较大，直觉是KG并没有发挥什么作用，[代码](https://github.com/GanjinZero/KeBioLM)）**

**[Infusing Disease Knowledge into BERT for Health Question Answering, Medical Inference and Disease Name Recognition](https://arxiv.org/abs/2010.03746)** (EMNLP 2020)

为了融合knowledge到已有的pretrain model中，文章设计了一个弱监督的训练方法（原文用weakly-supervised，类似于Prompt机制）并构建了基于Wikipedia的entity-passage对数据。其方法核心是在医学文本的前部插入辅助问句（e.g.，“What is the [Aspect] of [Disease]?”），在训练阶段将aspect和disease名称mask掉构建相应的分类任务，比较tricky的是loss不仅用了常规cross entropy，还补充了未经softmax下的相应输出的监督。实验结果非常好，该融合思路在大部分BERT系列模型上都取得了提升。**（重点是设计了Prompt策略，简单且有效，[代码](https://github.com/heyunh2015/diseaseBERT)）**

## Dataset

**[emrKBQA: A Clinical Knowledge-Base Question Answering Dataset](https://aclanthology.org/2021.bionlp-1.7/)** (BioNLP 2021 workshop)

设计有点复杂，脑子清醒再看。**（数据集未公开）**

**[MedDialog: Large-scale Medical Dialogue Datasets](https://aclanthology.org/2020.emnlp-main.743/)** (EMNLP 2020)

医疗对话数据集，包括中英文，中文340万条对话，英文25万条对话。数据集核心内容除多轮对话外，还包括病情描述、病人基本情况等信息。**（[代码](https://github.com/UCSD-AI4H/Medical-Dialogue-System)）**

**[MedDG: A Large-scale Medical Consultation Dataset for Building Medical Dialogue System](https://arxiv.org/abs/2010.07497)**

医疗对话数据集，仅包含中文，共1.7万条对话。其中每条对话都有实体标注，涉及诱因、疾病、症状、检测和药物5种类型共160个实体标注。**（[代码](https://github.com/lwgkzl/MedDG)）**

## Dialogue

**[End-to-End Knowledge-Routed Relational Dialogue System for Automatic Diagnosis](https://arxiv.org/abs/1901.10623)** (AAAI 2019)

涉及RL，待会看。**（[代码](https://github.com/HCPLab-SYSU/Medical_DS)）**

**[MIE: A Medical Information Extractor towards Medical Dialogues](https://aclanthology.org/2020.acl-main.576/)** (ACL 2020)

文章构建了一个window-to-information标注的医疗对话数据集，具体为某个sliding window内的对话内容与数个entity（包含状态，例如检测阳性：pos）相对应，不涉及entity内具体token的标注。文章设计了一个匹配模型以计算某个dialogue片段与任意entity的得分，其中dialogue和entity的encoder均为Bi-LSTM，两者用attention进行交互以获取融合后的表示，并通过sigmoid得到最终匹配分。**（亮点在于用ranking方法解决entity抽取，非常有趣，不过用的backbone很复古，[代码](https://github.com/nlpir2020/MIE-ACL-2020)）**

**[Graph-Evolving Meta-Learning for Low-Resource Medical Dialogue Generation](https://arxiv.org/abs/2012.11988)** (AAAI 2021)

文章主要提出了一个基于meta-learning的训练框架用于解决医疗场景下的low-resource问题，其中Graph-Evolving指将对话中的两个共现entity进行连接以弥补已有Commonsense Graph在医疗领域的不足。具体实现上，模型用了hierarchical的encoder分别编码句子和整段对话，句子编码单独构成graph并和entity graph进行融合，entity graph用GAT方式学习，而对话编码则用于decoder的初始化。最终decoder涉及词表和entity graph的条件概率。**（用meta-learning解决long-tail或者low-resource是个很有意思的点；graph-evole很tricky，如果只是共现，且和embedding无关，其实都不需要evolve，直接在最开始根据所有dialogue把图补全就好了，目前不知道作者具体怎么实现的；文章撰写极其糟糕，句子编码的graph如何融合到entity graph中完全没有说明，decoder输入以及输出的概率完全没有说明，[代码](https://github.com/ha-lins/GEML-MDG)）**

**[Extracting Appointment Spans from Medical Conversations](https://aclanthology.org/2021.nlpmc-1.6/)** (NLPMC 2021 workshop)

文章构建了一个弱监督的序列标注数据集，目的是从医疗对话文本中抽取appointment原因和时间。**(没什么实质内容，无代码)**

## Summarization

[**Attend to Medical Ontologies: Content Selection for Clinical Abstractive Summarization**](https://arxiv.org/abs/2005.00163) (ACL 2020)

为了更精确识别医学实体（文中用ontology term指代），文章构建了一个Bi-LSTM+BERT的content selector对输入文本中的ontology term进行0/1标注（表示term是否被复制到输出文本中）。模型包括原始输入文本和ontology term两个LSTM-based encoder，用ontology的最终state增强每个原始输入文本的state。另外decoder没说清楚。**（整体思路常规，无代码）**

[**A Gradually Soft Multi-Task and Data-Augmented Approach to Medical Question Understanding**](https://aclanthology.org/2021.acl-long.119/) (ACL 2021)

文章提出了一个gradually soft的多任务学习框架，该框架固定不同任务的encoder参数hard sharing（完全共享），而decoder参数gradually soft sharing（即模型层面上不共享，但通过loss约束每层的参数相似性，该文章期望层数越大，参数差异越大）。关于data augmentation，作者首先说明了summarization任务和recognizing question entailment（问题蕴含识别）任务的一致性，然后通过将S任务改写成R任务，将R任务改写成S任务，从而构建领域一致的数据集。实验结果显示这两个改进都能带来一定程度提升。**（亮点在于gradually soft，而data augmentation属于锦上添花，并且方法通用性很强，并没有依赖medical相关知识，但是这文章写的实在是恶心，尤其数据增强部分，明明就很简单却要描述的那么绕，无[代码](https://github.com/KhalilMrini/Medical-Question-Understanding)）**

## Reading Comprehension

**[Knowledge-Empowered Representation Learning for Chinese Medical Reading Comprehension: Task, Model and Resources](https://aclanthology.org/2021.findings-acl.197.pdf)** (ACL Findings 2021)

文章设计了一个CMedMRC任务以及数据集（形式同SQuAD，e.g.，**Passage**-**Question**-**Answer**-**Support sentence**），总结了medical MRC的几个难点：长尾术语（出现频率非常低的token）、书面vs口头表述差异、术语组合以及术语转述（个人认为这几个难点算medical NLP的难点，不单单只是MRC）。 文章设计了一个BERT-based的answer extraction模型，核心是KB representation（用PTransE）和BERT representation的融合，在常规方法（把KB信息attend到BERT output上）上叠加一个过滤器（Gated Loop Layer），该过滤器以迭代的方式不断用KB信息精炼和融合当前BERT output。**(整体更像一个QA任务，比较有意思的点在于迭代精炼，但作者并没有做相关ablation实验，效果不得而知，无[代码](https://github.com/MatNLP/CMedMRC))**

**[Towards Medical Machine Reading Comprehension with Structural Knowledge and Plain Text](https://aclanthology.org/2020.emnlp-main.111/)** (EMNLP 2020)

文章根据中国国家执业药师资格考试设计了一个问答数据集，形式为五选一的选择题。为有效融合医学知识到已有模型，文章主要采取了三种思路：1. 构建**intermediate-task**预训练，将CMeKG中的（h，r，t）三元组以“[CLS] [h] [SEP] [t]”的形式组成输入，并构建基于BERT的分类器区分不同的r；2. 将（h，r，t）转化成“The [h] of [r] is [t]”的形式，并与question中的entity计算word mover距离，筛选TOP-K的（h，r，t）三元组；3. 用GCN强化**思路2**中的entity表示（如果我没理解错，事实上原文中并没有详细描述该操作的实现）。另外，模型在输入阶段通过IR的方式获取了TOP-N的evidence进行输入增强。实验结果上显示，**思路1**的分类器和**思路3**的GCN强化效果最不显著。**（重点是IR的引入，在多个步骤中用ranking的方式获取更多信息从而增强输入，最后的GCN像是凑数的，没什么用，无代码，[demo](http://112.74.48.115:8157/)）**

## NER

**[A Neural Transition-based Joint Model for Disease Named Entity Recognition and Normalization](https://aclanthology.org/2021.acl-long.219/)** (ACL 2021)



## Others

**[Counterfactual Supporting Facts Extraction for Explainable Medical Record Based Diagnosis with Graph Network](https://aclanthology.org/2021.naacl-main.156/)** (NAACL 2021)

没看懂，正二刷。**（无[代码](https://github.com/CKRE/CMGE)）**

**[Rationalizing Medical Relation Prediction from Corpus-level Statistics](https://arxiv.org/abs/2005.00889)** (ACL 2020)



## Be still updating...

