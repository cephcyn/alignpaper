# Paper text excerpt alignment

High-level summary of approach:
- Collect a list of text excerpts from paper abstracts (e.g. phrases describing study population) (collected in [analyze.ipynb](#TODO_LINK))
- Align the text excerpts in a table-like format (executed in [multiple_alignment.ipynb](multiple_alignment.md))
  - Initialize the alignment by splitting each excerpt into individual tokens, ordering the excerpts, and running Smith-Waterman alignment to build an initial alignment table.
  - Refine the alignment by defining several operations and performing a stochastic gradient descent search in the alignment space using an alignment quality heuristic.
    - Operations: merge column, split column (TODO this could be better defined), shift cell (TODO this could be better defined)
