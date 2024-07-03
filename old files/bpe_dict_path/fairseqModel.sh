cut -f1 "/local/musaeed/coptic-translator/bpe_dict_path/coptic__vocab_12000.vocab" | tail -n +4 | sed "s/$/ 100/g" > "/local/musaeed/coptic-translator/bpe_dict_path/fairseq.coptic.vocab"
cut -f1 "/local/musaeed/coptic-translator/bpe_dict_path/en__vocab_12000.vocab" | tail -n +4 | sed "s/$/ 100/g" > "/local/musaeed/coptic-translator/bpe_dict_path/fairseq.en.vocab"
