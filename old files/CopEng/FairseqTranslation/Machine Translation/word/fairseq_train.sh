DROPOUT=0.3
ATTENTION_DROPOUT=0
ACTIVATION_DROPOUT=0.3
EMBEDDING_SIZE=300
ENC_FFNN=1200
ENCODER_LAYERS=4
ENCODER_ATTENTION_HEADS=10
DECODER_LAYERS=4
DECODER_ATTENTION_HEADS=10
DEC_FFNN=1200
EPOCH=550
BATCH_SIZE=64
ENCODER_LAYER_DROPOUT=0.2
DECODER_LAYER_DROPOUT=0.2
SOURCE_LANGUAGE=coptic
TARGET_LANGUAGE=en
LABEL_SMOOTHING=0.1
SAVE_DIR=/local/musaeed/coptic-translator/fairseqCheckpoints/sanityCheck/
LABEL_CROSS_ENTROPY=label_smoothed_cross_entropy
WARMUP_UPDATES=4000
lEARNING_POLICY=inverse_sqrt
WAND_PROJECT_NAME="Coptic2EN Translation Sanity Check"
# name can contain only "_" as non characters

fairseq-train "/local/musaeed/coptic-translator/fairseqCheckpoints/sanityCheck/coptic_en.tokenized.coptic-en" \
    --arch transformer \
    --dropout $DROPOUT \
    --attention-dropout 0     --encoder-embed-dim $EMBEDDING_SIZE \
    --encoder-ffn-embed-dim $ENC_FFNN  \
    --encoder-layers $ENCODER_LAYERS  \
    --encoder-attention-heads $ENCODER_ATTENTION_HEADS \
    --encoder-learned-pos  \
    --decoder-embed-dim $EMBEDDING_SIZE \
    --decoder-ffn-embed-dim $DEC_FFNN  \
    --decoder-layers $DECODER_LAYERS \
    --decoder-attention-heads $DECODER_ATTENTION_HEADS \
    --decoder-learned-pos   \
    --max-epoch $EPOCH  \
    --optimizer adam \
    --lr 5e-4 \
    --batch-size $BATCH_SIZE \
    --seed 1     --encoder-layerdrop $ENCODER_LAYER_DROPOUT     --decoder-layerdrop $DECODER_LAYER_DROPOUT \
    --criterion $LABEL_CROSS_ENTROPY     --warmup-updates $WARMUP_UPDATES \
    --source-lang $SOURCE_LANGUAGE     --label-smoothing $LABEL_SMOOTHING \
    --lr-scheduler $lEARNING_POLICY   --save-dir $SAVE_DIR \
    --find-unused-parameters  \
    --target-lang $TARGET_LANGUAGE \
    --activation-dropout $ACTIVATION_DROPOUT  \
    --ddp-backend=no_c10d \
    --no-epoch-checkpoints --wandb-project "$WAND_PROJECT_NAME" \
    --log-format=json --log-interval=10 2>&1    |  tee  "PATH_TO_LOG_FILE.log"
