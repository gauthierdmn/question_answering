# experiment ID
exp = "exp2"

# data directories
data_dir = "/Users/gdamien/Data/squad/"
train_dir = data_dir + "train/"
dev_dir = data_dir + "dev/"

# model paths
spacy_en = "/Users/gdamien/Data/spacy/en_core_web_sm-2.0.0/en_core_web_sm/en_core_web_sm-2.0.0"
glove = "/Users/gdamien/Data/glove.6B/glove.6B.{}d.txt"

# preprocessing values
max_words = 40000
word_embedding_size = 300
char_embedding_size = 8
char_channel_width = 5
char_channel_size = 100
max_len_context = 400
max_len_question = 50
max_len_word = 25

# training hyper-parameters
num_epochs = 10
batch_size = 64
valid_size = 0.01
learning_rate = 0.5
drop_prob = 0.2
hidden_size = 100
output_dim = 100
cuda = False
pretrained = False
