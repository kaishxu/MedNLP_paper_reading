# MedNLP paper reading
纯主观，只是学习记录。

## Summarization

[**Attend to Medical Ontologies: Content Selection for Clinical Abstractive Summarization (ACL 2020)**](https://arxiv.org/abs/2005.00163)

为了更精确识别医学实体（文中用ontology指代），该文章构建了一个bi-LSTM+BERT的content selector对输入文本中的ontology term进行0/1标注（表示term是否被复制到输出文本中）。模型包括原始输入文本和ontology term两个encoder，用ontology的最终state增强每个原始输入文本的state，decoder没说清楚。**（整体思路常规，描述不是很清楚，无代码）**



###### Be still updating...

