# MedNLP paper reading
学习记录（纯主观）。

## Pretraining

[**Conceptualized Representation Learning for Chinese Biomedical Text Mining**](https://arxiv.org/abs/2008.10813) (WSDM 2020 Health Day)

文章提出了一种显式融合医学实体和**医学短语**的预训练方法。整体思路同ERNIE1.0，在训练阶段对相关实体和短语进行mask。唯一的亮点是额外训练了一个短语分类器，用于区分医学转述（e.g.，“肚子有一点痛”==“胃疼”），并且训练时不仅融合医学实体，还融合医学短语。**（整体感觉是ERNIE的医学version，最大的贡献是发布了ChineseBLUE，[代码](https://github.com/alibaba-research/ChineseBLUE)）**

**[Improving Biomedical Pretrained Language Models with Knowledge](https://arxiv.org/abs/2104.10344)** (BioNLP 2021 workshop)

文章设计了一个KG-enhanced的预训练模型，主体结构为两个transformers堆叠，第一个transformers的输出与KG（用TransE）相融合，生成第二个transformers的输入。作者把KG信息attend到entity所对应的transformers输出上，对于任意entity，attend的KG涉及最近的k个entity（如何获得这k个entity没看懂，应该是图上最近的k个entity）。预训练任务除了MLM，还包括了entity的NER任务，以及KG和transformers表示的linking任务（使同一个entity在两者的表示相近）。**（效果一言难尽，因为最后测试的还是NER任务，并且实验未给出entity masking和KG融合哪个作用比较大，直觉是KG并没有发挥什么作用，[代码](https://github.com/GanjinZero/KeBioLM)）**

**[Infusing Disease Knowledge into BERT for Health Question Answering, Medical Inference and Disease Name Recognition](https://arxiv.org/abs/2010.03746)** (EMNLP 2020)

为了融合knowledge到已有的pretrain model中，文章设计了一个弱监督的训练方法（原文用weakl-supervised，类似于Prompt机制）并构建了基于Wikipedia的entity-passage对数据。其方法核心是在医学文本的前部插入辅助问句（e.g.，“What is the [Aspect] of [Disease]?”），在训练阶段将aspect和disease名称mask掉构建相应的分类任务，比较tricky的是loss不仅用了常规cross entropy，还补充了未经softmax下的相应输出的监督。实验结果非常好，该融合思路在大部分BERT系列模型上都取得了提升。**（重点是设计了Prompt策略，简单且有效，[代码](https://github.com/heyunh2015/diseaseBERT)）**

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

## Summarization

[**Attend to Medical Ontologies: Content Selection for Clinical Abstractive Summarization**](https://arxiv.org/abs/2005.00163) (ACL 2020)

为了更精确识别医学实体（文中用ontology term指代），文章构建了一个bi-LSTM+BERT的content selector对输入文本中的ontology term进行0/1标注（表示term是否被复制到输出文本中）。模型包括原始输入文本和ontology term两个LSTM-based encoder，用ontology的最终state增强每个原始输入文本的state。另外decoder没说清楚。**（整体思路常规，无代码）**

## Reading Comprehension

**[Knowledge-Empowered Representation Learning for Chinese Medical Reading Comprehension: Task, Model and Resources](https://aclanthology.org/2021.findings-acl.197.pdf)** (ACL Findings 2021)

文章设计了一个CMedMRC任务以及数据集（形式同SQuAD，e.g.，**Passage**-**Question**-**Answer**-**Support sentence**），总结了medical MRC的几个难点：长尾术语（出现频率非常低的token）、书面vs口头表述差异、术语组合以及术语转述（个人认为这几个难点算medical NLP的难点，不单单只是MRC）。 文章设计了一个BERT-based的answer extraction模型，核心是KB representation（用PTransE）和BERT representation的融合，在常规方法（把KB信息attend到BERT output上）上叠加一个过滤器（Gated Loop Layer），该过滤器以迭代的方式不断用KB信息精炼和融合当前BERT output。**(整体更像一个QA任务，比较有意思的点在于迭代精炼，但作者并没有做相关ablation实验，效果不得而知，无[代码](https://github.com/MatNLP/CMedMRC))**

**[Towards Medical Machine Reading Comprehension with Structural Knowledge and Plain Text](https://aclanthology.org/2020.emnlp-main.111/)** (EMNLP 2020)



## Others

**[Counterfactual Supporting Facts Extraction for Explainable Medical Record Based Diagnosis with Graph Network](https://aclanthology.org/2021.naacl-main.156/)** (NAACL 2021)

没看懂，正二刷。**（无[代码](https://github.com/CKRE/CMGE)）**

## Be still updating...

