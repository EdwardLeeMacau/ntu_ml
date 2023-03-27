# Report

## Question Answering

1. Make a brief introduction about one of the variant of Transformer, and use an image of the structure of the model to help explain.

    [Reformer](https://arxiv.org/pdf/2001.04451.pdf), a variant of Transformer, which addresses computation complexity for extra long sequence, is proposed in ICLR2020. In this paper, the author points out that the attention score is growing with speed of $L^2$, where $L$ is the length of the sequence to be processed.

    To alleviate this problem, author suggests that it is not necessary to calculate all pairs of the attention score, because the attention score for each embedding should be dominated by several embedding only, the others are near to $0$ thus negligible. To save computation, some modification on transformer is addressed.
    - Use $Q = K$, which means the weights to calculate $Q$ and $K$ are shared.
    - Use locality-sensitive hash (LSH) to separate $Q$ into several buckets.
    - Calculate attention score for the embedding inside the same bucket.

    ![](reformer.png)

2. Briefly explain what’s the advantages of this variant under certain
situations.

    Reformer aims to accelerate processing speed on long input sequence instances.

## Reference

1. https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0
