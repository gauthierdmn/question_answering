# Question Answering with SQuAD

The goal of this project was two-fold: getting familiar with the Question Answering task, a very active topic in NLP research, but also experiment the BERT pre-training model released by Google earlier this year, a very promising transfer learning approach.

To this end, I implemented a Bidirectional Attention Flow neural network as a baseline, starting from Chris Chute implemenation and adding word-character inputs. Then I used the op-for-op implementation of BERT for PyTorch released by the HuggingFace team.

# Resources

* SQuAD dataset: https://arxiv.org/abs/1606.05250
* Bidirectional Attention Flow for Machine Comprehension"
by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi : https://arxiv.org/abs/1611.01603
* BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova : https://arxiv.org/abs/1810.04805
* BiDAF baseline model: https://github.com/chrischute/squad/blob/master/model.py
* PyTorch pretrained BERT: https://github.com/huggingface/pytorch-pretrained-BERT
* GloVE: https://nlp.stanford.edu/projects/glove/