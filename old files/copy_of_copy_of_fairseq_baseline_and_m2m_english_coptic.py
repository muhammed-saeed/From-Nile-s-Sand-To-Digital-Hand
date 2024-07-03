#

import torch
#the print has to be T4 GPU to ensure that we are using the GPU
torch.cuda.get_device_name(0)


# !wget -qq "https://dl.fbaipublicfiles.com/m2m_100/spm.128k.model"
# !wget -qq "https://dl.fbaipublicfiles.com/m2m_100/data_dict.128k.txt"
# !wget -qq "https://dl.fbaipublicfiles.com/m2m_100/model_dict.128k.txt"
# !wget -qq "https://dl.fbaipublicfiles.com/m2m_100/language_pairs_small_models.txt"
# !wget "https://dl.fbaipublicfiles.com/m2m_100/418M_last_checkpoint.pt"

# ! pip install sentencepiece -q

import pandas as pd
import torch
import numpy as np
import os
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(7)

# #mkdir /local/musaeed/coptic-translator/TrainTestCSV/



# Define directory paths
base_path = "/local/musaeed/coptic-translator/"
train_test_csv_dir = base_path + "TrainTestCSV/"
fairseq_train_data_dir = base_path + "dataPreperationForFairseqTrain/"
bpe_dict_path = base_path + "bpe_dict_path/"

# Additional directories
train_bpe_dir = fairseq_train_data_dir + "train_bpe/"
test_bpe_dir = fairseq_train_data_dir + "test_bpe/"
val_bpe_dir = fairseq_train_data_dir + "val_bpe/"
BPE_train_dir = fairseq_train_data_dir + "BPE_train/"

# Ensure directories exist, create them if necessary
os.makedirs(train_test_csv_dir, exist_ok=True)
os.makedirs(fairseq_train_data_dir, exist_ok=True)
os.makedirs(bpe_dict_path, exist_ok=True)
os.makedirs(train_bpe_dir, exist_ok=True)
os.makedirs(test_bpe_dir, exist_ok=True)
os.makedirs(val_bpe_dir, exist_ok=True)
os.makedirs(BPE_train_dir, exist_ok=True)



import pandas as pd

# Load the preprocessed data
train = pd.read_csv("/local/musaeed/coptic-translator/dataset/processed/copticEnglish.csv")

# Remove any possible duplicates based on Coptic and English columns
train = train.drop_duplicates(subset=["Coptic", "English"])

# Lowercase and remove trailing spaces
train["Coptic"] = train["Coptic"].str.strip().str.lower()
train["English"] = train["English"].str.strip().str.lower()

# Split data: 2200 for validation and the rest for training
np.testing = train.sample(n=2200)
train = train.drop(index=np.testing.index)

# Optional: Save these dataframes to CSV if needed
train.to_csv("/local/musaeed/coptic-translator/TrainTestCSV/train.csv", index=False)
np.testing.to_csv("/local/musaeed/coptic-translator/TrainTestCSV/test.csv", index=False)

import pandas as pd

PATH_TO_DATASET = "./"  # Where you stored the dataset

# Load the datasets
train = pd.read_csv("/local/musaeed/coptic-translator/TrainTestCSV/train.csv")
test = pd.read_csv("/local/musaeed/coptic-translator/TrainTestCSV/test.csv")

# Remove duplicates from train
train = train.drop_duplicates(subset=["Coptic", "English"])

# Function to preprocess the data
def preprocess_data(df):
  #lower case al the data
    df["Coptic"] = df["Coptic"].astype(str).apply(lambda x: x.strip().lower())
    df["English"] = df["English"].astype(str).apply(lambda x: x.lower())
    df = df[["Coptic", "English"]]
    df.columns = ["input_text", "target_text"]
    return df.astype(str)

train = preprocess_data(train)
test = preprocess_data(test)

# Train 95% / Validation 5% Split from the training dataset
validation = train.sample(frac=0.05)
train = train.drop(index=validation.index)

train_txt = "\n".join(train.input_text.values.tolist())

#create a file and then store the training portion on this file
file = open("/local/musaeed/coptic-translator/dataPreperationForFairseqTrain/train.coptic", "w")
file.write(train_txt)
file.close()


train_target_txt = "\n".join(train.target_text.values.tolist())

file = open("/local/musaeed/coptic-translator/dataPreperationForFairseqTrain/train.en", "w")
file.write(train_target_txt)
file.close()

validation_txt = "\n".join(validation.input_text.values.tolist())

file = open("/local/musaeed/coptic-translator/dataPreperationForFairseqTrain/val.coptic", "w")
file.write(validation_txt)
file.close()


validation_target_txt = "\n".join(validation.target_text.values.tolist())

file = open("/local/musaeed/coptic-translator/dataPreperationForFairseqTrain/val.en", "w")
file.write(validation_target_txt)
file.close()

test_txt = "\n".join(test.input_text.values.tolist())

file = open("/local/musaeed/coptic-translator/dataPreperationForFairseqTrain/test.coptic", "w")
file.write(test_txt)
file.close()


test_target_txt = "\n".join(test.target_text.values.tolist())

file = open("/local/musaeed/coptic-translator/dataPreperationForFairseqTrain/test.en", "w")
file.write(test_target_txt)
file.close()

"""##Word Based Method"""


"""##BPE Based"""

#mkdir /local/musaeed/coptic-translator/bpe_dict_path
#mkdir /local/musaeed/coptic-translator/dataPreperationForFairseqTrain/
#mkdir /local/musaeed/coptic-translator/dataPreperationForFairseqTrain/train_bpe/
#mkdir /local/musaeed/coptic-translator/dataPreperationForFairseqTrain/test_bpe/
#mkdir /local/musaeed/coptic-translator/dataPreperationForFairseqTrain/val_bpe/

import pandas as pd

# Load the preprocessed data
train = pd.read_csv("/local/musaeed/coptic-translator/dataset/processed/copticEnglish.csv")

# Assuming your CSV has columns named 'coptic' and 'english'
coptic_texts = train['Coptic'].tolist()
english_texts = train['English'].tolist()

# Save Coptic texts to a text file
with open('/local/musaeed/coptic-translator/coptic.txt', 'w', encoding='utf-8') as f:
    for text in coptic_texts:
        if isinstance(text, str):  # Check if the value is a string
            f.write(text + '\n')
        else:
            f.write('\n')  # Write an empty line for non-string values

# Save English texts to a text file
with open('/local/musaeed/coptic-translator/english.txt', 'w', encoding='utf-8') as f:
    for text in english_texts:
        if isinstance(text, str):  # Check if the value is a string
            f.write(text + '\n')
        else:
            f.write('\n')  # Write an empty line for non-string values

# !pip install sentencepiece
import sentencepiece as spm



dict_path = "/local/musaeed/coptic-translator/bpe_dict_path"
#dictionary "create directory and name it as you like"
coptic_train_input = "/local/musaeed/coptic-translator/dataPreperationForFairseqTrain/train.coptic"
coptic_train_bpe_output = "/local/musaeed/coptic-translator/dataPreperationForFairseqTrain/train_bpe/train.bpe.coptic"
en_train_input = "/local/musaeed/coptic-translator/dataPreperationForFairseqTrain/train.en"
en_train_bpe_output = "/local/musaeed/coptic-translator/dataPreperationForFairseqTrain/train_bpe/train.bpe.en"

coptic_val_input = "/local/musaeed/coptic-translator/dataPreperationForFairseqTrain/val.coptic"
coptic_val_bpe_output = "/local/musaeed/coptic-translator/dataPreperationForFairseqTrain/val_bpe/val.bpe.coptic"
en_val_input = "/local/musaeed/coptic-translator/dataPreperationForFairseqTrain/val.en"
en_val_bpe_output = "/local/musaeed/coptic-translator/dataPreperationForFairseqTrain/val_bpe/val.bpe.en"

coptic_test_input = "/local/musaeed/coptic-translator/dataPreperationForFairseqTrain/test.coptic"
coptic_test_bpe_output = "/local/musaeed/coptic-translator/dataPreperationForFairseqTrain/test_bpe/test.bpe.coptic"
en_test_input = "/local/musaeed/coptic-translator/dataPreperationForFairseqTrain/test.en"
en_test_bpe_output = "/local/musaeed/coptic-translator/dataPreperationForFairseqTrain/test_bpe/test.bpe.en"

vocab_size = 12000


sw_source_file = "/local/musaeed/coptic-translator/coptic.txt"
ar_source_file = "/local/musaeed/coptic-translator/english.txt"

# joint_vocab_file = "/local/musaeed/coptic-translator/joint_train.txt"




#using entire text to found the dictonary for the BPE
def train_coptic(vocab_size):
  model_prefix = dict_path+"/coptic_" + "_vocab_" + str(vocab_size)
  spm.SentencePieceTrainer.train(input=ar_source_file
      , model_prefix=model_prefix
      , vocab_size=vocab_size
      , character_coverage = 0.9995
      , num_threads=60
      , model_type = "bpe"
      , train_extremely_large_corpus=True
  )
train_coptic(vocab_size)

def train_en(vocab_size):
  model_prefix = dict_path + "/en_" + "_vocab_" + str(vocab_size)
  spm.SentencePieceTrainer.train(input=sw_source_file
      , model_prefix=model_prefix
      , vocab_size=vocab_size
      , character_coverage = 0.9995
      , num_threads=60
      ,model_type = "bpe"
      , train_extremely_large_corpus=True
  )
train_en(vocab_size)

coptic_tokenizer = spm.SentencePieceProcessor(model_file="/local/musaeed/coptic-translator/bpe_dict_path/coptic__vocab_12000.model")
en_tokenizer = spm.SentencePieceProcessor(model_file="/local/musaeed/coptic-translator/bpe_dict_path/en__vocab_12000.model")

# trianiing 22.3K (coptic/english)
# testing 2.2K
# validation 0.05* 22.3K


#from the above the BPE models are trained
####################################33


#lines uses the english BPE model to tokenize the english training data
with open(coptic_train_input, "r", encoding="utf-8") as rf, open(coptic_train_bpe_output, "w", encoding="utf-8") as wf:
    output_lines = []
    for line in rf.readlines():
        wf.write(' '.join(coptic_tokenizer.encode(line, out_type=str)))
        # output_lines.append(tokenizer.encode(input = line, out_type = str))
        wf.write("\n")

    # wf.writelines(str(output_lines))

with open(en_train_input, "r", encoding="utf-8") as rf, open(en_train_bpe_output, "w", encoding="utf-8") as wf:
    output_lines = []
    for line in rf.readlines():
        wf.write(' '.join(en_tokenizer.encode(line, out_type=str)))
        # output_lines.append(tokenizer.encode(input = line, out_type = str))
        wf.write("\n")

    # wf.writelines(str(output_lines))



with open(coptic_test_input, "r", encoding="utf-8") as rf, open(coptic_test_bpe_output, "w", encoding="utf-8") as wf:
    output_lines = []
    for line in rf.readlines():
        wf.write(' '.join(coptic_tokenizer.encode(line, out_type=str)))
        # output_lines.append(tokenizer.encode(input = line, out_type = str))
        wf.write("\n")

    # wf.writelines(str(output_lines))


with open(en_test_input, "r", encoding="utf-8") as rf, open(en_test_bpe_output, "w", encoding="utf-8") as wf:
    output_lines = []
    for line in rf.readlines():
        wf.write(' '.join(en_tokenizer.encode(line, out_type=str)))
        # output_lines.append(tokenizer.encode(input = line, out_type = str))
        wf.write("\n")

    # wf.writelines(str(output_lines))



with open(coptic_val_input, "r", encoding="utf-8") as rf, open(coptic_val_bpe_output, "w", encoding="utf-8") as wf:
    output_lines = []
    for line in rf.readlines():
        wf.write(' '.join(coptic_tokenizer.encode(line, out_type=str)))
        # output_lines.append(tokenizer.encode(input = line, out_type = str))
        wf.write("\n")

    # wf.writelines(str(output_lines))



with open(en_val_input, "r", encoding="utf-8") as rf, open(en_val_bpe_output, "w", encoding="utf-8") as wf:
    output_lines = []
    for line in rf.readlines():
        wf.write(' '.join(en_tokenizer.encode(line, out_type=str)))
        # output_lines.append(tokenizer.encode(input = line, out_type = str))
        wf.write("\n")

    # wf.writelines(str(output_lines))
