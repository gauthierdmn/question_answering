# data directories
data_dir = '/Users/gdamien/Data/squad/'
train_dir = data_dir + 'train/'
dev_dir = data_dir + 'dev/'

# model paths
spacy_en = '/Users/gdamien/Data/spacy/en_core_web_sm-2.0.0/en_core_web_sm/en_core_web_sm-2.0.0'
glove = '/Users/gdamien/Data/glove.6B/glove.6B.{}d.txt'

# preprocessing values
max_words = 40000
embedding_size = 50
max_len_context = 100

# training hyper-parameters
num_epochs = 10
batch_size = 32
valid_size = 0.01
learning_rate = 0.1
drop_prob = 0.2
hidden_size = 100
output_dim = 100
cuda = False
pretrained = False
