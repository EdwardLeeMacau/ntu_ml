# Report

## Question Answering

1. Make a brief introduction about one of the variant of Transformer, and use an image of the structure of the model to help explain.

    [Reformer](https://arxiv.org/pdf/2001.04451.pdf), a variant of Transformer, which addresses time and space complexity under extra long input sequence, is proposed in ICLR2020. Transformer helps solving translation, summarization and even text generation. However, the author points out that the number of attention score to be calculated is growing with speed of $L^2$, where $L$ is the length of the sequence to be processed, this limits the power of Transformer to solve NLP tasks which have extra long sequence, such as understanding the content of a book, etc.

    To alleviate this problem, author suggests that it is not necessary to calculate all pairs of the attention score, because the attention score for each embedding should be dominated by several embedding only, the others are near to $0$ thus negligible. To save computation, some modification on transformer is addressed.
    - Use $Q = K$, which means the weights to calculate $Q$ and $K$ are shared.
    - Use locality-sensitive hash (LSH) to separate $Q$ into several buckets.
    - Calculate attention score for the embedding inside the same bucket. Attention score is $0$ when two embedding vectors are from different buckets.

    Here is the figure captured from [paper](https://arxiv.org/pdf/2001.04451.pdf), which explains the mechanism of LSH.

    ![](reformer.png)

2. Briefly explain what’s the advantages of this variant under certain situations.

    Reformer aims to make transformer possible to handle (also, accelerate processing speed on) extra long sequence.

## Reference

1. [Arxiv - Reformer](https://arxiv.org/pdf/2001.04451.pdf)
2. [[TA 補充課] Transformer and its variant (由助教紀伯翰同學講授)](https://www.youtube.com/watch?v=lluMBz5AoOg&ab_channel=Hung-yiLee)
3. [Google Research - Reformer: The Efficient Transformer](https://ai.googleblog.com/2020/01/reformer-efficient-transformer.html)
4. [Median - Illustrating the Reformer](https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0)
5. [https://www.youtube.com/watch?v=yHoAq1IT_og&ab_channel=Hung-yiLee](https://www.youtube.com/watch?v=yHoAq1IT_og&ab_channel=Hung-yiLee)
