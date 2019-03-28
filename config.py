# experiment ID
exp = "exp3"

# data directories
data_dir = "/AZVOL/question_answering/data/"
train_dir = data_dir + "train/"
dev_dir = data_dir + "dev/"

# model paths
spacy_en = "/Users/gdamien/Data/spacy/en_core_web_sm-2.0.0/en_core_web_sm/en_core_web_sm-2.0.0"
glove = "/AZVOL/question_answering/data/glove.6B/glove.6B.{}d.txt"

# preprocessing values
max_words = 40000
word_embedding_size = 300
char_embedding_size = 8
max_len_context = 400
max_len_question = 50
max_len_word = 25

# training hyper-parameters
num_epochs = 15
batch_size = 64
valid_size = 0.01
learning_rate = 0.5
drop_prob = 0.2
hidden_size = 100
char_channel_width = 5
char_channel_size = 100
cuda = True
pretrained = False
