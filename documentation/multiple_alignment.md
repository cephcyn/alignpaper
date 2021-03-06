# multiple_alignment.ipynb

[High-level description.](main.md) This notebook performs the alignment and alignment space state search functions.

Contains code to import from the saved format generated by [analyze_ipynb](), align text, perform transformations on the alignment, score alignments, and search through the alignment space.

## Formats

### <a id="formatImport"></a>Import format

TODO elaborate?

### <a id="formatAlignment"></a>Alignment format

DataFrame with index=source text ID, columns titled txt0...txtN, and each cell is a tuple:

```
(text_of_cell, pos_of_entire_text, [list_of_pos_of_each_cell_word/token])
```

TODO elaborate?

## Functions

### <a id="alignRowMajorLocal"></a>Initial / "Full" text alignment (alignRowMajorLocal)

An implementation of Smith-Waterman for text alignment, using a [modified version of Smith-Waterman scoring function](#alignRowMajorLocal_scoringFunction)

TODO pseudocode

Inputs:
- align_a, align_b: [Alignment DataFrames](#formatAlignment) that we want to align together.
- use_types: Parameter for the scoring function, iff True then phrases must have identical phrase POS to match
- remove_empty_cols: iff True then strip out empty columns from the input alignment DFs (so forcibly remove gaps from the inputs)
- debug_print: ignore this, this only exists for debugging reasons

#### <a id="alignRowMajorLocal_scoringFunction"></a>alignRowMajorLocal scoring function (S-W scoring)

TODO elaborate

#### ~~<a id="buildAlignmentBeamSearch"></a>How to get the ordering of when to add each row into the "full alignment" for alignRowMajorLocal?~~

This isn't important, I implemented this for curiosity's sake.

~~buildAlignmentBeamSearch: perform a beam search.~~

Pseudocode:

```
initialize an empty beam
seed with N=10 (by default) starting rows to try building the alignment from
while all the elements in the beam don't contain all of the rows:
  initialize an empty 'next' beam
  for each of the elements/alignments in the beam:
    try adding each of M=10 (by default) random columns that aren't already in the alignment onto the alignment; put the result in the 'next' beam
  sort the 'next' beam by alignment score and filter out all but the top F=5 (by default) alignments
  set beam = 'next' beam
return the alignment (alignment ordering) of the first/best element/alignment in the beam
```

~~I'm not sure that this search actually works well, especially when the S/W align function isn't *too* great to start with... TODO, I want to test if this works better with the S-W alignment score or the full-alignment score. I feel like intuitively it would work better with the full-alignment score.~~

### <a id="tempScoreVector"></a>Alignment scoring (scoreAlignment)

Calculates an overall score for a given [alignment table]((#formatAlignment). It does this by calculating a weighted sum of several sub-scores, each of which target different desired traits for a good text alignment.

There are way more sub-score types than are actually used in the overall score.

There are a few candidates for overall score weighting:
- -1 * [(number of columns)](#score_numcolumns) + -1 * [(text embedding variance)](score_colptxtembed) (**this is the method I'm primarily using**)
- -1 * [(number of columns)](#score_numcolumns) + -1 * [(distinct tokens w/ variance)](score_coltoknvarcount)
- -1 * [(number of columns)](#score_numcolumns) + -1 * [(distinct entity TUIs w/ variance)](score_colttuivarcount)

#### Weighting schemes

#### <a id="weight_alignmentTerms"></a>Weights: term weighting

For scores that are term-specific (are calculated on a specific set of terms), we need a way to weight each term. Roughly, this weighting scheme collects all of the unique tokens in an alignment, removes stopwords, selects a group of POS tags that it especially cares about and squares the count of words that are most frequently tagged as those POSs. The final (adjusted) count is equal to either the number of rows a token appears in, or that number squared (if it's one of the 'priority' POS tags)

The priority POS list by default: NN, (NNS, NNP), JJ, RB

How I calculate these weights:

```
Collect list of tokens, POS per token
Reduce list of tokens to a dict mapping token to how many rows it appears in
Remove tokens that only appear in a single row
Count how many times each POS is applied to each token
Map each token to the POS it is most frequently tagged as
Square the row counts of tokens that have POSs within a priority POS list
```

#### <a id=""></a>Weights: column weighting

For scores that are column-specific (can be calculated per-column and then somehow summed together to represent the entire alignment), we need a way to weight each column / judge how "relevant" each column is. There are multiple possible ways to do this:
- num tokens (**this is the method I'm primarily using**): each column is weighted by the total number of tokens it contains (e.g. if a column contains 2 filled rows with 3 words each, then the total number of tokens is 6).
- row representation: each column is weighted by the number of rows filled within it (e.g. if a column contains 2 filled rows, the row representation is 2)

I've also experimented with adjusting both of these weighting schemes for distinct entity type or distinct entity TUI scoring so that columns that contain 0 identified entities are excluded from the weighting entirely. As of 2021/01/13, these adjusted schemes aren't used in the codebase at all.

#### Score components

##### <a id="score_numcolumns"></a>Score: number of columns (scoreNumColumns)

Number of columns divided by the largest number of tokens present in any single given row.
- Division is because the initial number of columns is determined in part by maximum token count (every single token occupies its own cell)

##### <a id="score_coltextcount"></a>Score: number of distinct phrases per column (scoreColumnTextCount)

The number of unique cells / texts / phrases in a given column.

##### <a id="score_colptxtembed"></a>Score: phrase embedding variance per column (scoreColumnPhraseEmbedVariance)

The trace of the covariance matrix containing phrase embeddings for all of the phrases within a given column.
- Trace(covariance(embeddings)) is because https://stats.stackexchange.com/questions/225434/a-measure-of-variance-from-the-covariance-matrix
- The word embedding for a phrase is calculated as a sum of word2vec embeds (for tokens that have embeds)
- If a column has no tokens that have valid word2vec embeds, then it returns 0.

##### <a id="score_coltokncount"></a>Score: number of distinct tokens per column (scoreColumnTokenCount)

The number of unique tokens / words in a given column.

<a id="score_coltoknvarcount"></a>
Also can apply a variation count: the number of unique tokens / words in a given column that *don't appear in all of the rows* (scoreColumnTokenVariationCount)

##### <a id="score_coltentcount"></a>Score: number of distinct entities per column (scoreColumnTokenEntityCount)

The number of unique entities in a given column.

<a id="score_coltentvarcount"></a>
Also can apply a variation count: the number of unique entities in a given column that *don't appear in all of the rows* (scoreColumnTokenEntityVariationCount)

##### <a id="score_colttuicount"></a>Score: number of distinct entity types per column (scoreColumnTokenEntityCount)

The number of unique entity types in a given column. This is different from the unique entities because there are some entities that end up grouping into the same entity type: for example, "patient" and "outpatient" may be the same entity type but have unique entity IDs.

<a id="score_colttuivarcount"></a>
Also can apply a variation count: the number of unique entity types in a given column that *don't appear in all of the rows* (scoreColumnTokenEntityVariationCount)

##### <a id="score_colpposcount"></a>Score: number of distinct phrase POS per column (scoreColumnPhrasePOSCount)

The number of unique phrase POSs (e.g. NP, VP) in a given column. (The phrase POS itself is determined by constituency parse and is often very variable for parse tree structure itself.)

##### <a id="score_coltposcount"></a>Score: number of distinct token POS per column (scoreColumnPOSCount)

The number of unique token POSs (e.g. NN, NNS, JJ, CD) in a given column. (The token POS is determined by constituency parse, which is relatively stable for token POS tagging.)

<a id="score_coltposvarcount"></a>
Also can apply a variation count: the number of unique token POSs in a given column that *don't appear in all of the rows* (scoreColumnPOSVariationCount)

##### <a id="score_colrepresent"></a>Score: number of rows that are represented per column (scoreColumnRepresentation)

The fraction of rows that are represented (have any cells with content in them) in a given column.

##### <a id="scoreRowAlignment"></a>Score: [gap/mismatch score](#alignRowMajorLocal_scoringFunction) between a target row and the alignment (scoreRowAlignment)

The [SW alignment score](#alignRowMajorLocal_scoringFunction) of an alignment between a given target row and the full alignment. This is intended to allow a user to specify a target row or pattern to match to, and provide additional emphasis on how well the alignment itself matches that.

##### <a id="score_termcolcount"></a>Score: weighted sum of how frequently target terms are repeated per column (scoreTermListColumnCount / scoreTermColumnCount)

A score intended to discourage terms from being spread across a large number of columns in an alignment. It calculates the number of columns that a given term (or terms) is present in within the full alignment, then sums them together (equally by default, or otherwise with a given set of weights for all of the terms).

I generally use this with a [term / token weighting scheme](weight_alignmentTerms)

##### <a id="scoreRowLayoutCount"></a>Score: number of distinct row layouts (sequence of phrase-filled/gap) present in an alignment (scoreRowLayoutCount)

The number of distinct row layouts / sequences of filled cells and gap cells in a full alignment.

### <a id="alignment-search"></a>Alignment search

Stochastic hill-climbing: either perform the operation that produces greatest full-alignment score increase, or randomly perform an operation.

Operations that the search algorithm explores:
- [Split a column on word tree level boundaries](splitCol)
- [Merge two columns](mergeCol)

#### <a id="splitCol"></a>Alignment column splitting (splitCol)

Split an [alignment column](#formatAlignment) using a word tree (either left-to-right or right-to-left)

Pseudocode:

```
Build word trie: - not in-depth as this is relatively standard trie building
  Initialize empty trie / prefix tree with an empty root node representing "phrase/list start"
  For each phrase (list of tokens) in the column:
    If performing right-to-left split, reverse the list of tokens
    Add list of tokens to trie, where each trie node is a single token
Collapse word trie:
  (Outermost execution: Set current trie node to root node)
  If current trie node has children:
    Collapse each of the child nodes
  If current trie node has exactly one child node:
    Combine the text of the current trie node and child node (if left-to-right, append; otherwise prepend)
    Assign each of the child node's children as the current trie node's children
  Return collapsed trie (the trie with current node as root is now fully collapsed)
Split into columns:
  N = height/depth of the collapsed word trie
  Create N new alignment table columns to replace the original column-to-be-split
  For each phrase in the old column:
    Get the list corresponding to tracing the phrase out in the collapsed word trie, padding the tail end of the list with blank strings (if right-to-left, pad the head of the list)
    Write list element X into column X
```

#### <a id="mergeCol"></a>Alignment column merging (mergeCol)

Merge two [alignment columns](#formatAlignment) together.

Pseudocode:

```
For each row in the columns:
  Combine the raw text (append col2 to col1)
  Combine the phrase POS (uses the phrase POS of text in col1 if there is one, otherwise uses phrase POS of col2)
  Combine the list of token POSs (append col2 to col1)
Remove the extra column
```

#### <a id="shiftCell"></a>Alignment cell shifting (shiftCell)

Shift the specified cell (or set of cells) N spaces to the left or to the right. It will (not shift | only shift the maximum possible number of spaces without overwriting any other cells) if the specified shift distance N overlaps with other cell contents or with the alignment boundaries.

TODO elaborate on which sets of cells for shifting the search algorithm actually considers

Pseudocode:
```
Calculate the maximum possible distance the specified segment can be shifted...
  - without overwriting non-empty cell contents
  - without exceeding the given shift distance
Remove the specified segment (overwrite with empty)
Write the specified segment content into the new location
```
