# WordSuggestion-in-Python

Word Suggestion with N-Grams and Cosine Similarity implemented in Python from Scratch.

This project demonstrates a fundamental approach to building a word suggestion or autocomplete system without relying on external machine learning libraries. It leverages character-level n-grams and cosine similarity to find the closest matching words from a dictionary.

## Algorithm Overview

The core algorithm operates on the principle of representing words as numerical vectors based on their n-gram frequencies and then calculating the distance (similarity) between these vectors.

1.  **Data Preparation**:
    *   Reads a corpus of words from `dictionary.txt`.
    *   Filters out words shorter than a minimum length (3 characters).
    *   Randomly samples a subset of words (1000 words) for performance, simulating a constrained dictionary.
2.  **N-Gram Generation**:
    *   For every word in the dictionary, generates multi-grams of varying sizes (typically 3, 4, and 5-grams). 
    *   For example, 3-grams of "apple" are `['app', 'ppl', 'ple']`.
3.  **Vectorization**:
    *   Builds a frequency map of all unique n-grams across the entire sampled dictionary to determine the vector space dimension.
    *   Maps each word to a vector where each index represents a specific n-gram, and the value represents its frequency in that word.
4.  **Similarity Calculation (Cosine Similarity)**:
    *   When an input word is provided, it's converted into a vector using the same n-gram space.
    *   The system computes the Cosine Similarity between the input vector and all dictionary word vectors.
5.  **Ranking**:
    *   Words are ranked in descending order of their similarity score.
    *   The top $k$ (default 10) words with scores $> 0$ are returned as suggestions.

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

$$ \text{Cosine Similarity}(v, w) = \cos(\theta) = \frac{v \cdot w}{\|v\| \|w\|} $$

Where:
*   $v \cdot w$ is the dot product of the vectors: $\sum_{i=1}^{n} v_i w_i$
*   $\|v\|$ is the magnitude (Euclidean norm) of vector $v$: $\sqrt{\sum_{i=1}^{n} v_i^2}$

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

1.  **Text Normalization (Case Folding)**: The system currently treats uppercase and lowercase characters differently (e.g., "Apple" and "apple" produce completely different N-grams). Normalizing all dictionary words and user inputs to lowercase (`word.lower()`) would solve case-sensitivity mismatches.
2.  **TF-IDF Weighting**: Currently, the system uses raw N-gram frequencies. Implementing Term Frequency-Inverse Document Frequency (TF-IDF) would decrease the weight of overly common N-grams and highlight unique, distinguishing N-grams for better accuracy. The weighting can be calculated as:
    
    $$ \text{TF-IDF}(t, d, D) = \text{TF}(t, d) \cdot \log\left(\frac{N}{\text{DF}(t)}\right) $$
    
    Where $t$ is the N-gram, $d$ is the specific word, $N$ is the total number of words in the dictionary corpus, and $\text{DF}(t)$ is the number of words containing the N-gram $t$.
3.  **Special Character Filtering**: Stripping out punctuation, numbers, or special characters to ensure the algorithm only compares alphabetical characters.
4.  **Optimized Vector Operations**: The vector dimensions map to all unique N-grams across the dataset, creating sparse arrays. Using sparse matrix libraries (like SciPy) or NumPy could greatly improve performance for larger dictionaries.

### Alternative Similarity Metrics

While Cosine Similarity works well for measuring the angular distance between frequency vectors, other metrics could be offered as alternatives depending on the specific use case:

1.  **Jaccard Similarity**: Instead of looking at frequencies, Jaccard similarity measures the size of the intersection divided by the size of the union of two sets of n-grams. It's excellent for cases where simply the *presence* or *absence* of n-grams matters more than their frequency count.
    
    $$ J(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|} $$
    
    Where $A$ and $B$ are the sets of n-grams for the input word and dictionary word, respectively.

2.  **Euclidean Distance**: Measures the straight-line distance between two points in vector space. While less robust for varying length strings compared to Cosine Similarity, it can be useful when magnitude is as important as the angle.
    
    $$ d(v, w) = \sqrt{\sum_{i=1}^{n} (v_i - w_i)^2} $$

3.  **Levenshtein Distance (Edit Distance)**: Rather than using vector representations, this metric calculates the minimum number of single-character edits (insertions, deletions, or substitutions) required to change one word into the other. It is the gold standard for many traditional spell-checkers.
    
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
