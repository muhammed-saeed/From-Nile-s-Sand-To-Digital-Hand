BATCH_SIZE=128
BEAM=5
SEED=1
SCORING="bleu"
CHECKPOINT_PATH="/local/musaeed/coptic-translator/checkpointWordBasedPCM/English2CopticWordBasedSeq2SeqTransformer/checkpoint_last.pt"

fairseq-generate "/local/musaeed/coptic-translator/fairseqwordBased/en_coptic.tokenized.en-coptic" \
    --batch-size $BATCH_SIZE \
    --beam $BEAM \
    --path $CHECKPOINT_PATH \
    --seed $SEED \
    --scoring "bleu" > "/local/musaeed/coptic-translator/training/wordBased/results/english to coptic word based generation results.txt"
