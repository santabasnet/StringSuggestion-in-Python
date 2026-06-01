# WordSuggestion-in-Python

Word Suggestion with N-Grams and Cosine Similarity implemented in Python from Scratch.

This project demonstrates a fundamental approach to building a word suggestion or autocomplete system without relying on external machine learning libraries. It leverages character-level n-grams and cosine similarity to find the closest matching words from a dictionary.

## Algorithm Overview

The algorithm is implemented entirely in [suggestion.py](suggestion.py). The `main()` function orchestrates **12 sequential steps**, from reading raw data to printing ranked word suggestions. Each step maps directly to a dedicated function.

---

### Step 1 — Read All Words from File
```python
word_list = read_all_words()
```
Opens `dictionary.txt` and reads every line into a Python list. Each line is treated as a single word entry. The file may contain tens of thousands of words.

---

### Step 2 — Sample and Sort 1000 Words
```python
sorted_sampled_words = sample_1000_words(word_list)
```
- **Filters** out any word shorter than `MINIMUM_LENGTH = 3` characters (e.g., `"a"`, `"to"` are discarded).
- **Randomly samples** 1000 words from the filtered list using `random.sample()`.
- **Sorts** the 1000 words alphabetically.

> *Example*: From 80,000+ words in the dictionary, 1000 are selected, e.g., `['abandon', 'ability', 'above', ...]`.

---

### Step 3 — Build the N-Gram Dictionary
```python
word_ngram_dictionary = build_ngram_dictionary(sorted_sampled_words)
```
For every sampled word, generates all character-level n-grams of sizes `NGRAM_SIZES = [3, 4, 5]` by calling `multi_grams_of(word)`, which in turn calls `ngram_of(word, size)` for each size. The result is a dictionary mapping each word to its list of n-grams.

> *Example for* `"blue"`:
> - 3-grams → `['blu', 'lue']`
> - 4-grams → `['blue']`
> - 5-grams → `[]` *(word is too short)*
> - Combined → `['blu', 'lue', 'blue']`

```python
{ 'blue': ['blu', 'lue', 'blue'], 'bluebell': ['blu', 'lue', 'ueb', 'ebe', 'bel', 'ell', 'blue', 'lueb', ...], ... }
```

---

### Step 4 — Build N-Gram Frequencies
```python
ngram_frequencies = build_ngram_frequencies(word_ngram_dictionary)
```
Iterates over every n-gram from every word and counts how many times each unique n-gram appears across the entire sampled dictionary. This creates the global n-gram frequency table.

> *Example*: If `'ing'` appears in 200 of the 1000 words, then `ngram_frequencies['ing'] = 200`.

```python
{ 'blu': 3, 'lue': 5, 'blue': 3, 'ing': 200, 'tion': 150, ... }
```

---

### Step 5 — Calculate Vector Dimensions
```python
dimension = len(ngram_frequencies.keys())
```
The total number of unique n-grams across all sampled words defines the **dimensionality** of the vector space. Every word will be represented as a vector of this length.

> *Example*: If there are 4500 unique n-grams, every word vector has 4500 dimensions.

---

### Step 6 — Sort N-Gram Positions
```python
ngram_positions = list(ngram_frequencies.keys())
ngram_positions.sort()
```
Extracts all unique n-grams and sorts them alphabetically. This sorted list determines the **fixed ordering** of dimensions in every word vector, ensuring consistency.

> *Example (partial)*: `['abl', 'abla', 'ablan', 'able', 'ably', 'blu', 'blue', ...]`

---

### Step 7 — Build the Position Dictionary
```python
position_dictionary = build_position_dictionary(ngram_positions)
```
Maps each unique n-gram to a fixed integer index (its position in the vector). This lookup table is used during vectorization to know which index of the vector a given n-gram corresponds to.

> *Example*:
> ```python
> { 'abl': 0, 'abla': 1, 'ablan': 2, 'able': 3, ..., 'blu': 45, 'blue': 46, ... }
> ```

---

### Step 8 — Build All Word Vectors
```python
ngram_vectors = build_word_vectors(word_ngram_dictionary, position_dictionary, ngram_frequencies)
```
For each word in the dictionary, calls `vector_of()` → `ngram_vectorizer()`. The vectorizer creates a zero-filled array of `dimension` length, then for each n-gram the word contains it places the **global n-gram frequency** at the corresponding index position.

> *Example for* `"blue"` *(simplified, dimension = 4500)*:
> ```
> Index:  0    1    2    3   ...   45   46   ...  4499
> Value: [0,   0,   0,   0,  ...,   3,   3,  ...,   0]
>                                  ^^^  ^^^
>                                 'blu' 'blue'
>                           (frequency = 3 each)
> ```

---

### Step 9 — Read the Input Word and Vectorize It
```python
input_word = input("\nGive a word : ")
input_vector = vector_of(input_word, word_ngram_dictionary, position_dictionary, ngram_frequencies)
```
Prompts the user for a word. The input word is converted into a vector using the **same** position dictionary and n-gram frequencies. If the word is already in the dictionary, its pre-built n-grams are reused; otherwise, n-grams are generated on the fly.

> *Example*: User types `"bluet"`.
> - N-grams: `['blu', 'lue', 'uet', 'blue', 'luet', 'bluet', 'lueto', ...]`
> - Vectorized against the same 4500-dimension space.

---

### Step 10 — Calculate Cosine Similarity with All Dictionary Vectors
```python
angles_with_input = calculate_angles_with(input_vector, ngram_vectors)
```
Iterates over every word in `ngram_vectors` and calls `cos_theta_of(v, w)`, which computes:

$$\cos(\theta) = \frac{v \cdot w}{\|v\| \cdot \|w\|}$$

using `dot_product_of()` and `magnitude_of()` implemented from scratch. The result is a dictionary of `{ word: similarity_score }` for all 1000 words.

> *Example (partial)*:
> ```python
> { 'blue': 0.93, 'bluebell': 0.81, 'bluet': 1.0, 'blur': 0.45, 'abandon': 0.0, ... }
> ```

---

### Step 11 — Rank Words by Similarity Score
```python
ranked_words = ranked_words_of(angles_with_input)
```
Sorts all words by their cosine similarity score in **descending order** using Python's `sorted()`. Selects the **top `SUGGESTION_SIZE = 10`** results.

> *Example (top 5)*:
> ```python
> { 'bluet': 1.0, 'blue': 0.93, 'bluebell': 0.81, 'bluets': 0.79, 'blur': 0.45 }
> ```

---

### Step 12 — Filter Zero-Scored Words and Print Results
```python
final_suggestions = filter_zero_scored_words(ranked_words)
```
Removes any words with a similarity score of `0.0` (no shared n-grams at all). The remaining words are printed as the final suggestions.

> *Example output for input* `"bluet"`:
> ```text
> Give a word : bluet
>
> Final Suggestions :
>     >> bluet:          1.0
>     >> blue:           0.93
>     >> bluebell:       0.81
>     >> bluets:         0.79
>     >> blur:           0.45
> ```

---

## Workflow Diagram

```text
      (Input Word)                            [Read dictionary.txt]
           |                                            |
           v                                            v
[Generate N-Grams for Input]               [Filter & Sample 1000 words]
           |                                            |
           |                                            v
           |                             [Generate Multi-N-Grams n=3,4,5]
           |                                            |
           |                                            v
           |                                [Build N-Gram Frequencies]
           |                                            |
           |                                            v
           |                              [Create Vector Space Dictionary]
           |                                        /      \
           v                                       /        \
 [Vectorize Input Word] <-------------------------+          \
           |                                                  v
           |                                      [Vectorize Sampled Words]
           |                                                  |
           +--------------------------------------------------+
                                   |
                                   v
                     [Calculate Cosine Similarity]
                                   |
                                   v
                         [Rank Words by Score]
                                   |
                                   v
                [Filter Zero Scores & Select Top 10]
                                   |
                                   v
                         ((( Final Suggestions )))
```

## Mathematical Background

**Cosine Similarity** measures the cosine of the angle between two non-zero vectors of an inner product space. It defines how similar two words are based on their n-gram vector representations.

Given two word vectors $v$ and $w$:

$$ \text{Cosine Similarity}(v, w) = \cos(\theta) = \frac{v \cdot w}{\Vert v \Vert \Vert w \Vert} $$

Where:
*   $v \cdot w$ is the dot product of the vectors: $\sum_{i=1}^{n} v_i w_i$
*   $\Vert v \Vert$ is the magnitude (Euclidean norm) of vector $v$: $\sqrt{\sum_{i=1}^{n} v_i^2}$

A score of `1.0` implies the words are identical (vectors point in the exact same direction), while `0.0` means they share no common n-grams (vectors are orthogonal).

## Usage

### Prerequisites
*   Python 3.x (Uses standard libraries `math` and `random`, no external dependencies required).
*   A dictionary file named `dictionary.txt` in the same directory. Each word should ideally be on a new line.

### Running the Application

Execute the Python script from your terminal:

```bash
python suggestion.py
```

### Example Interaction

```text
Give a word : python

Final Suggestions : 
	>> python:			1.0
	>> pythonic:			0.85
	>> pythons:			0.78
```
*(Note: Actual output depends on the contents of your `dictionary.txt` and the random sampling)*

## Project Structure
*   `suggestion.py`: The main script containing the n-gram and cosine similarity implementation.
*   `dictionary.txt`: The corpus of words used to build the suggestion engine.

## Future Enhancements / Suggestions

Currently, the implementation is a foundational demonstration of N-grams and Cosine Similarity. There is **no text normalization** implemented in the script. Adding the following features would significantly improve the robustness of the suggestion engine:

#### 1. Text Normalization (Case Folding)
The system currently treats uppercase and lowercase characters differently (e.g., "Apple" and "apple" produce completely different N-grams). Normalizing all dictionary words and user inputs to lowercase (`word.lower()`) would solve case-sensitivity mismatches.

#### 2. TF-IDF Weighting
Currently, the system uses raw N-gram frequencies. Implementing Term Frequency-Inverse Document Frequency (TF-IDF) would decrease the weight of overly common N-grams and highlight unique, distinguishing N-grams for better accuracy. The weighting can be calculated as:

$$ \text{TF-IDF}(t, d, D) = \text{TF}(t, d) \cdot \log\left(\frac{N}{\text{DF}(t)}\right) $$

Where $t$ is the N-gram, $d$ is the specific word, $N$ is the total number of words in the dictionary corpus, and $\text{DF}(t)$ is the number of words containing the N-gram $t$.

#### 3. Special Character Filtering
Stripping out punctuation, numbers, or special characters to ensure the algorithm only compares alphabetical characters.

#### 4. Optimized Vector Operations
The vector dimensions map to all unique N-grams across the dataset, creating sparse arrays. Using sparse matrix libraries (like SciPy) or NumPy could greatly improve performance for larger dictionaries.

### Alternative Similarity Metrics

While Cosine Similarity works well for measuring the angular distance between frequency vectors, other metrics could be offered as alternatives depending on the specific use case:

#### 1. Jaccard Similarity
Instead of looking at frequencies, Jaccard similarity measures the size of the intersection divided by the size of the union of two sets of n-grams. It's excellent for cases where simply the *presence* or *absence* of n-grams matters more than their frequency count.

$$ J(A, B) = \frac{\vert A \cap B \vert}{\vert A \cup B \vert} = \frac{\vert A \cap B \vert}{\vert A \vert + \vert B \vert - \vert A \cap B \vert} $$

Where $A$ and $B$ are the sets of n-grams for the input word and dictionary word, respectively.

#### 2. Euclidean Distance
Measures the straight-line distance between two points in vector space. While less robust for varying length strings compared to Cosine Similarity, it can be useful when magnitude is as important as the angle.

$$ d(v, w) = \sqrt{\sum_{i=1}^{n} (v_i - w_i)^2} $$

#### 3. Levenshtein Distance (Edit Distance)
Rather than using vector representations, this metric calculates the minimum number of single-character edits (insertions, deletions, or substitutions) required to change one word into the other. It is the gold standard for many traditional spell-checkers.

The distance is typically calculated using a Dynamic Programming algorithm. It constructs a matrix $D$ where $D[i,j]$ represents the edit distance between the first $i$ characters of word $A$ and the first $j$ characters of word $B$:

$$
D[i,j] = \begin{cases} 
  i & \text{if } j = 0 \\
  j & \text{if } i = 0 \\
  D[i-1, j-1] & \text{if } A[i] = B[j] \\
  1 + \min \left( D[i-1, j], D[i, j-1], D[i-1, j-1] \right) & \text{otherwise}
\end{cases}
$$
### Vector Storage & Database Integration

Currently, the vector space and all dictionary word vectors are computed in-memory every time the script runs. For a production environment with a massive dictionary, this approach is not scalable. 

A critical enhancement is to pre-compute these vectors and store them in a database:
*   **Vector Databases**: Utilize specialized vector databases like **Milvus**, **Pinecone**, **Qdrant**, or **Weaviate**. These are optimized for high-dimensional vectors and perform blazing fast Approximate Nearest Neighbor (ANN) searches (e.g., using HNSW indexing) instead of brute-force Cosine Similarity across the whole dataset.
*   **Standard Databases**: Alternatively, store the vectors in **PostgreSQL** (leveraging the `pgvector` extension) or **Redis** for fast in-memory lookups.

By doing this, when a user provides an input, the system only vectorizes the *single* input word and queries the database for the closest matches, reducing response times to milliseconds.
