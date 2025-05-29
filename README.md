# bert-sst2-experiments

**Experiment Design Strategy**

In the first experiment (Trained BERT), the code was deliberately designed to process one layer at a time. This design choice was made to facilitate faster testing, ensure reproducibility, and allow for layer-by-layer analysis. The third experiment (No punctuation) follows the same principle.

**Handling Randomness in Untrained Models**

In contrast, the second experiment (Untrained BERT) involved a randomly initialized (untrained) model. Since the weights were randomly generated, results varied with each run—even when using the same random seed. Therefore, it was not feasible to isolate the analysis by layer. For this reason, the second experiment uses a global script that processes all layers and all iterations in a unified manner.
