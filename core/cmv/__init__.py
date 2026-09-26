"""ChangeMyView (Experiment 2): Tan's paired persuasion task, end to end.

The pieces, in pipeline order:

* :mod:`data`          ConvoKit Winning-Args corpus -> the 4,263 matched pair units
* :mod:`items`         (OP view, challenger reply) items for the comparative run
* :mod:`graphs`        comparison graphs with forbidden same-thread edges
* :mod:`taxonomy`      the cited-dimension listing the taxonomy prompt embeds
* :mod:`bt`            per-feature Bradley-Terry with a FROZEN-anchor scorer for new items
* :mod:`bt_halves`     the same fit on disjoint halves of the comparisons -- the BT arms'
                       independent replicate, which the noise correction needs
* :mod:`features`      Tan et al.'s interplay / BOW / POS featurizers
* :mod:`embeddings`    the text-embedding-3-large baseline
* :mod:`pairtask`      the signed pair-difference task and its shared seed-42 sign deck
* :mod:`designs`       arm designs + the L1 paired logistic regression and its bootstrap
* :mod:`paired`        shared-resample accuracy intervals across arms, and the two-run PW mean
* :mod:`matched_k`     accuracy of criterion panels at a matched number of criteria k
* :mod:`dimensionality` participation ratio / effective rank, disattenuated for scoring noise
* :mod:`win_rates`     per-criterion effectiveness with Benjamini-Hochberg control
* :mod:`sample_elicit` the data-informed (S) panel
"""
