# MedNLP paper reading
学习记录（纯主观）。

## Pretraining

[**Conceptualized Representation Learning for Chinese Biomedical Text Mining**](https://arxiv.org/abs/2008.10813) (WSDM 2020 Health Day)

文章提出了一种显式融合医学实体和**医学短语**的预训练方法。整体思路同ERNIE1.0，在训练阶段对相关实体和短语进行mask。唯一的亮点是额外训练了一个短语分类器，用于区分医学转述（e.g.，“肚子有一点痛”==“胃疼”），并且训练时不仅融合医学实体，还融合医学短语。**（整体感觉是ERNIE的医学version，最大的贡献是发布了ChineseBLUE，[代码](https://github.com/alibaba-research/ChineseBLUE)）**

**[Improving Biomedical Pretrained Language Models with Knowledge](https://arxiv.org/abs/2104.10344)** (BioNLP 2021 workshop)

文章设计了一个KG-enhanced的预训练模型，主体结构为两个transformers堆叠，第一个transformers的输出与KG（用TransE）相融合，生成第二个transformers的输入。作者把KG信息attend到entity所对应的transformers输出上，对于任意entity，attend的KG涉及最近的k个entity（如何获得这k个entity没看懂，应该是图上最近的k个entity）。预训练任务除了MLM，还包括了entity的NER任务，以及KG和transformers表示的linking任务（使同一个entity在两者的表示相近）。**（效果一言难尽，实验未给出entity masking和KG融合哪个作用比较大，直觉是KG并没有发挥什么作用，[代码](https://github.com/GanjinZero/KeBioLM)）**

## Dataset

**[emrKBQA: A Clinical Knowledge-Base Question Answering Dataset](https://aclanthology.org/2021.bionlp-1.7/)** (BioNLP 2021 workshop)

数据集未公开，设计有点复杂，脑子清醒再看。

## Summarization

[**Attend to Medical Ontologies: Content Selection for Clinical Abstractive Summarization**](https://arxiv.org/abs/2005.00163) (ACL 2020)

为了更精确识别医学实体（文中用ontology term指代），文章构建了一个bi-LSTM+BERT的content selector对输入文本中的ontology term进行0/1标注（表示term是否被复制到输出文本中）。模型包括原始输入文本和ontology term两个LSTM-based encoder，用ontology的最终state增强每个原始输入文本的state。另外decoder没说清楚。**（整体思路常规，无代码）**

## Reading Comprehension

**[Knowledge-Empowered Representation Learning for Chinese Medical Reading Comprehension: Task, Model and Resources](https://aclanthology.org/2021.findings-acl.197.pdf)** (ACL Findings 2021)

文章设计了一个CMedMRC任务以及数据集（形式同SQuAD，e.g.，**Passage**-**Question**-**Answer**-**Support sentence**），总结了medical MRC的几个难点：长尾术语（出现频率非常低的token）、书面vs口头表述差异、术语组合以及术语转述（个人认为这几个难点算medical NLP的难点，不单单只是MRC）。 文章设计了一个BERT-based的answer extraction模型，核心是KB representation（用PTransE）和BERT representation的融合，在常规方法（把KB信息attend到BERT output上）上叠加一个过滤器（Gated Loop Layer），该过滤器以迭代的方式不断用KB信息精炼和融合当前BERT output。**(比较有意思的点在于迭代精炼，但作者并没有做相关ablation实验，效果不得而知，无[代码](https://github.com/MatNLP/CMedMRC))**

## Be still updating...

