# Question Answering with SQuAD

The goal of this project was for me to get familiar with the Question Answering task, a very active topic in NLP research.

To this end, I implemented a Bidirectional Attention Flow neural network as a baseline, improving Chris Chute's model [implementation](https://github.com/chrischute/squad/models.py), adding word-character inputs as described in the original paper.

I found this project very useful from a learning perspective so I highly recommend you to dig into the code and work on improving this baseline.

# Model Architecture

![BiDAF Architecture](bidaf-architecture.png)

Source: [BiDAF paper](https://arxiv.org/abs/1611.01603)

# Code Organization

    ├── config.py          <- Configuration file with dataset directories and hyper-paramters to train the model
    ├── data_loader.py     <- Define an iterator who collects batches of data to train the model
    ├── eval.py            <- Evaluate the model on a new pair of (context, question)
    ├── layers.py          <- Define the various layers to be used by the main BiDAF model
    ├── make_dataset.py    <- Download the SquAD dataset and pre-process the data for training
    ├── model.py.          <- Define the BiDAF model architecture
    ├── requirements.txt   <- Required Python libraries to build the project
    ├── test.py            <- Test the performance of a trained model on the DEV dataset
    ├── train.py           <- Train a model using the TRAIN dataset only
    ├── utils.py           <- Group a bunch of useful functions to process the data

# Results

# Training Set-Up

# Next Step

- [ ] experiment pre-training encoding such as BERT (SOTA result as of writing).
- [ ] set up a multi-task learning pipeline to jointly learn to answer questions on SQuAD together with another closely related NLP task.

# Resources

* SQuAD dataset: https://arxiv.org/abs/1606.05250
* Bidirectional Attention Flow for Machine Comprehension"
by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi : https://arxiv.org/abs/1611.01603
* Authors' TensorFlow implementation: https://allenai.github.io/bi-att-flow/
* BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova : https://arxiv.org/abs/1810.04805
* BiDAF baseline model: https://github.com/chrischute/squad
* PyTorch pretrained BERT: https://github.com/huggingface/pytorch-pretrained-BERT
* GloVE: https://nlp.stanford.edu/projects/glove/
