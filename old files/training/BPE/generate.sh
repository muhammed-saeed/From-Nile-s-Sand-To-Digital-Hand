
BATCH_SIZE=128
BEAM=5
SEED=1
SCORING="bleu"
CHECKPOINT_PATH="/local/musaeed/coptic-translator/BPE_training/checkpoint_last.pt"

fairseq-generate "/local/musaeed/coptic-translator/dataPreperationForFairseqTrain/BPE_coptic_en.tokenized.coptic-en" \
    --batch-size $BATCH_SIZE \
    --beam $BEAM \
    --path $CHECKPOINT_PATH \
    --seed $SEED \
    --scoring "bleu" > "/local/musaeed/coptic-translator/bpe_cotpic_to_english_translaton_results.txt"

