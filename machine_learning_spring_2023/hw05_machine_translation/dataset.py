import fairseq
from fairseq.tasks.translation import TranslationConfig, TranslationTask

def setup_task(data_dir: str, source_lang: str, target_lang: str) -> TranslationTask:
    """# Dataloading

    ## We borrow the TranslationTask from fairseq
    * used to load the binarized data created above
    * well-implemented data iterator (dataloader)
    * built-in task.source_dictionary and task.target_dictionary are also handy
    * well-implemented beach search decoder

    # sample = task.dataset("valid")[1]
    # pprint(sample)
    # pprint(
    #     "Source: " + \
    #     task.source_dictionary.string(
    #         sample['source'],
    #         config.post_process,
    #     )
    # )
    # pprint(
    #     "Target: " + \
    #     task.target_dictionary.string(
    #         sample['target'],
    #         config.post_process,
    #     )
    # )
    """

    config = TranslationConfig(
        data=data_dir, source_lang=source_lang, target_lang=target_lang,
        train_subset="train", required_seq_len_multiple=8, dataset_impl="mmap",
        upsample_primary=1,
    )

    return TranslationTask.setup_task(config)

def get_data_iterator_kwargs(task: TranslationTask, max_tokens: int = 4000,
                             num_workers: int = 1, seed: int = 33, cached=True):
    """
    * Controls every batch to contain no more than N tokens, which optimizes GPU memory efficiency
    * Shuffles the training set for every epoch
    * Ignore sentences exceeding maximum length
    * Pad all sentences in a batch to the same length, which enables parallel computing by GPU
    * Add eos and shift one token
        - teacher forcing: to train the model to predict the next token based on prefix, we feed the right shifted target sequence as the decoder input.
        - generally, prepending bos to the target would do the job (as shown below)
          ![seq2seq](https://i.imgur.com/0zeDyuI.png)
        - in fairseq however, this is done by moving the eos token to the begining. Empirically, this has the same effect. For instance:
        ```
        # output target (target) and Decoder input (prev_output_tokens):
                    eos = 2
                    target = 419,  711,  238,  888,  792,   60,  968,    8,    2
        prev_output_tokens = 2,  419,  711,  238,  888,  792,   60,  968,    8
        ```
    """

    # API description:
    # https://fairseq.readthedocs.io/en/latest/tasks.html#fairseq.tasks.FairseqTask.get_batch_iterator
    #
    # Set "distable_iterator_cache" to False to speed up. However, if set to False, changing
    # max_tokens beyond first call of this method has no effect.
    return {
        "max_tokens": max_tokens,
        "max_sentences": None,
        "max_positions": fairseq.utils.resolve_max_positions(
            task.max_positions(),
            max_tokens,
        ),
        "ignore_invalid_inputs": True,
        "seed": seed,
        "num_workers": num_workers,
        "disable_iterator_cache": not cached,
    }
