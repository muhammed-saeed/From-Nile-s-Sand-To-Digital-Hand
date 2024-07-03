SOURCE_LANGUAGE=coptic
TARGET_LANGUAGE=en
TRAIN_PREF="/local/musaeed/coptic-translator/dataset/sanityCheck/train"
VALID_PREF="/local/musaeed/coptic-translator/dataset/sanityCheck/val"
TEST_PREF="/local/musaeed/coptic-translator/dataset/sanityCheck/test"
PCM_EN_DEST_DIR="/local/musaeed/coptic-translator/fairseqCheckpoints/sanityCheck/coptic_en.tokenized.coptic-en"
EN_PCM_DEST_DIR="/local/musaeed/coptic-translator/fairseqCheckpoints/sanityCheck/en_coptic.tokenized.en-coptic"
SRC_THRES=0
TGT_THRES=0

fairseq-preprocess     --source-lang $SOURCE_LANGUAGE     --target-lang $TARGET_LANGUAGE     --trainpref  $TRAIN_PREF     --validpref $VALID_PREF    --testpref $TEST_PREF    --destdir  $PCM_EN_DEST_DIR     --thresholdsrc $SRC_THRES     --thresholdtgt $TGT_THRES
