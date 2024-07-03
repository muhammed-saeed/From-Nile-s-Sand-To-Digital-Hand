
SOURCE_LANGUAGE="coptic"
TARGET_LANGUAGE="en"
TRAIN_PREF="/local/musaeed/coptic-translator/dataPreperationForFairseqTrain/train"
VALID_PREF="/local/musaeed/coptic-translator/dataPreperationForFairseqTrain/val"
TEST_PREF="/local/musaeed/coptic-translator/dataPreperationForFairseqTrain/test"
COPTIC_EN_DEST_DIR="/local/musaeed/coptic-translator/fairseqwordBased/coptic_en.tokenized.coptic-en"
EN_COPTIC_DEST_DIR="/local/musaeed/coptic-translator/fairseqwordBased/en_coptic.tokenized.en-coptic"
SRC_THRES=0
TGT_THRES=0

fairseq-preprocess     --source-lang $SOURCE_LANGUAGE     --target-lang $TARGET_LANGUAGE     --trainpref  $TRAIN_PREF     --validpref $VALID_PREF    --testpref $TEST_PREF    --destdir  $COPTIC_EN_DEST_DIR     --thresholdsrc $SRC_THRES     --thresholdtgt $TGT_THRES
#now create the data_bin file for the second direction in which the target now is coptic and the source is english
fairseq-preprocess     --source-lang $TARGET_LANGUAGE     --target-lang $SOURCE_LANGUAGE     --trainpref  $TRAIN_PREF     --validpref $VALID_PREF    --testpref $TEST_PREF    --destdir  $EN_COPTIC_DEST_DIR     --thresholdsrc $TGT_THRES     --thresholdtgt $SRC_THRES
