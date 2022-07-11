# NGram based word suggestion system implemented from 
# Scratch. 
#
# All libraries goes here.
# Written by Santa Basnet
# Date: 2021-07-10
#
# Event: Teaching Python for Undergraduate student,
# with implementation of cosine similarity from 
# scratch (It uses python dictionary to store frequency
# of n-grams).
# 
# -------------------------------------------------
import random
import math

# We define a dictionary input file.
INPUT_FILE      = "dictionary.txt"
MINIMUM_LENGTH  = 3
NGRAM_SIZES     = [3, 4, 5]
SUGGESTION_SIZE = 10

# Perform read operation of all words from the defined
# file name.
def read_all_words():
    dictionary_words = []
    with open(INPUT_FILE) as input_file:
        dictionary_words = input_file.readlines()
    return dictionary_words

# Generates sampled words.
def sample_1000_words(all_words):
    # 1. Filter unwanted characters and the given length threshold.
    filtered_words = []
    for i in all_words:
        t_word = i.strip()
        if len(t_word) >= MINIMUM_LENGTH:
            filtered_words.append(t_word)

    # 2. Perform sampling here to select 1000 words.
    sampled_words = random.sample(filtered_words, 1000)

    # 3. Sort all the words.
    sampled_words.sort()
    return sampled_words

# A function that generates n-grams of a word with the given size.
def ngram_of(word, size):
    word_grams = []
    for i in range(len(word) - size + 1):
        word_grams.append(word[i:i + size])
    return word_grams

# We define a function that takes a word and generates multi-grams with
# different sizes.
def multi_grams_of(word):
    ngrams = []    
    for i in NGRAM_SIZES:
        ngrams.extend(ngram_of(word, i))
    return ngrams

# Build n-grams of respective word with given sizes n = 3, 4, 5.
def build_ngram_dictionary(sampled_words):
    ngram_dictionary = {}
    for word in sampled_words:
        ngram_dictionary[word] = multi_grams_of(word)
    return ngram_dictionary

# Build the ngram frequency table.
def build_ngram_frequencies(ngram_dictionary):
    generated_ngrams = ngram_dictionary.values()
    ngram_frequencies = {}
    for ngrams in generated_ngrams:
        for gram in ngrams:
            ngram_frequencies[gram] = ngram_frequencies.get(gram, 0) + 1
    return ngram_frequencies

# Build position dictionary.
def build_position_dictionary(ngram_positions):
    position_dictionary = {}
    count = 0 
    for gram in ngram_positions:
        position_dictionary[gram] = count
        count = count + 1
    return position_dictionary# define a function that converts list of ngrams to the equvalent vector representation.

# We have list1 => list2
# ['blu', 'lue', 'uet', 'eto', 'too', 'oot', 'oth', 'blue', 'luet', 'ueto', 'etoo', 'toot', 'ooth', 'bluet', 'lueto', 'uetoo', 'etoot', 'tooth']
# => [0, 3, 4, ....]
def ngram_vectorizer(word_grams, position_dictionary, ngram_frequencies):
    dimension = len(ngram_frequencies.keys())
    word_vector = [0] * dimension
    for gram in word_grams:
        if gram in position_dictionary:
            position = position_dictionary.get(gram)
            frequency = ngram_frequencies.get(gram, 0)
            word_vector[position] = frequency
    return word_vector    
    
# Generates vector of the given word.
def vector_of(word, ngram_dictionary, position_dictionary, ngram_frequencies):
    if word in ngram_dictionary:
        return ngram_vectorizer(ngram_dictionary[word], position_dictionary, ngram_frequencies)
    else:
        return ngram_vectorizer(multi_grams_of(word), position_dictionary, ngram_frequencies) 

# Generates all the word vectors.
def build_word_vectors(ngram_dictionary, position_dictionary, ngram_frequencies):
    ngram_vectors = {}
    for word in ngram_dictionary:
        ngram_vectors[word] = vector_of(word, ngram_dictionary, position_dictionary, ngram_frequencies)
    return ngram_vectors

# We want calculate dot product.
def dot_product_of(v1, v2): 
    sum = 0
    for i in range(min(len(v1), len(v2))):
        sum += v1[i] * v2[i]
    return sum

# Magnitude of a vector
def magnitude_of(v):
    sum = 0
    for i in range(len(v)):
        sum += v[i] * v[i]
    return math.sqrt(sum)

# Define a function that calculates the cosine angle between two vectors.
def cos_theta_of(v, w):
    denominator = (magnitude_of(v) * magnitude_of(w))
    if(denominator == 0.0): 
        return 0.0
    else: 
        return (dot_product_of(v, w) / denominator) 

# It calculates and list all the angles between word vectors with input vector.
def calculate_angles_with(input_vector, ngram_vectors):
    angles_with_input = {}
    for word in ngram_vectors:
        angles_with_input[word] = cos_theta_of(ngram_vectors[word], input_vector)
    return angles_with_input

# Generates all the ranked suggested words of defined size.
def ranked_words_of(angles_with_input):
    ranked_dictionary = {}
    sorted_key_scores = sorted(angles_with_input, key = angles_with_input.get, reverse=True)
    for word in sorted_key_scores:
        ranked_dictionary[word] = angles_with_input[word]

    # Select top k words.
    result = dict(list(ranked_dictionary.items())[0: SUGGESTION_SIZE])
    return result

# Filter all the zero scored words.
def filter_zero_scored_words(suggestions):
    final_suggestions = {}
    for word in suggestions:
        if (suggestions[word] > 0.0):
            final_suggestions[word] = suggestions[word]
    return final_suggestions

# Program starts from here.
def main():
    # 1. Read all the words from the file.
    word_list = read_all_words()

    # 2. Initialize sample words with sorted values.
    sorted_sampled_words = sample_1000_words(word_list)

    # 3. Initialize the word and their ngram dictionary.
    word_ngram_dictionary = build_ngram_dictionary(sorted_sampled_words)

    # 4. Calculate ngram frequencies.
    ngram_frequencies = build_ngram_frequencies(word_ngram_dictionary)

    # 5.Initialize total dimensions.
    dimension = len(ngram_frequencies.keys())

    # 6. Calculate n_gram positions.
    ngram_positions =  (list(ngram_frequencies.keys()))
    ngram_positions.sort()

    # 7. Initialize ngram position dictionary.
    position_dictionary = build_position_dictionary(ngram_positions)

    # 8. Initialize all the word vectors from dictionary.
    ngram_vectors = build_word_vectors(word_ngram_dictionary, position_dictionary, ngram_frequencies)
    
    # 9. Define input vector for a word.
    input_word = input("\nGive a word : ")
    input_vector = vector_of(input_word, word_ngram_dictionary, position_dictionary, ngram_frequencies)

    # 10. Calculated angles with input vector.
    angles_with_input = calculate_angles_with(input_vector, ngram_vectors)

    # 11. Get ranked words.
    ranked_words = ranked_words_of(angles_with_input)
    
    # 12. Filter all the words in the ranking with zero scores.
    final_suggestions = filter_zero_scored_words(ranked_words)

    print("\nFinal Suggestions : \n")
    for word in final_suggestions:
        print("\t>> " + word + ":\t\t\t" + str(final_suggestions[word]))
    #--- End of Main ---#

if __name__ == "__main__":
    main()
    
