# Review of Transformers from Scratch
Erika Duan
2025-04-12

- [Original uses of transformers](#original-uses-of-transformers)
- [One-hot encoding](#one-hot-encoding)
- [The dot product](#the-dot-product)
- [Matrix multiplication](#matrix-multiplication)
- [First order sequence models](#first-order-sequence-models)
- [Second order sequence
  probabilities](#second-order-sequence-probabilities)
- [Second order sequence model with
  skips](#second-order-sequence-model-with-skips)
- [Masking](#masking)
- [Rest Stop and an Off Ramp](#rest-stop-and-an-off-ramp)
- [Attention as matrix
  multiplication](#attention-as-matrix-multiplication)
- [Key messages](#key-messages)
- [Other resources](#other-resources)

``` python
# Import Python libraries ------------------------------------------------------
import numpy as np
```

This is a review of the brilliant [transformers from scratch
tutorial](https://e2eml.school/transformers.html) by Brandon Rohrer.

It is useful to think of words as the fundamental units of natural
language processing tasks. In practice, sub-words or **tokens** are used
due to their increased representation flexibility (covers punctuation
and typos) and increased processing efficiency (reduced vocabulary
size).

# Original uses of transformers

Transformers were originally used for:

- Sequence transduction - converting one sequence of tokens into another
  sequence in another language.  
- Sequence completion - given a starting prompt, output a sequence of
  tokens in the same style.

Sequence transduction requires:  
+ A **vocabulary** - the set of unique tokens that form our language.
The vocabulary is created from the training data set.  
+ A method to **convert** unique tokens into unique numbers
(specifically vectors).  
+ A method of **encoding token context** - ensuring that the word `bank`
in `river bank` has a different numerical representation to the word
`bank` in `bank mortgage`.

# One-hot encoding

The simplest method of representing a vocabulary of words is using
**one-hot encoding**.

- Each unique word is represented by a unique 1D vector of mostly 0s and
  a single 1. This is called a **one-hot vector**.  
- The vector length is the length of the vocabulary.

``` python
# Create a simple vocabulary from "find my local files" ------------------------
# Position of 1 in each vector is assigned by sorting words via alphabetical
# order.    

files = [1, 0, 0, 0] 
find = [0, 1, 0, 0] 
local = [0, 0, 1, 0]
my = [0, 0, 0, 1]
  
[find, my, local, files]
```

    [[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]]

# The dot product

The [dot product](https://en.wikipedia.org/wiki/Dot_product) is a
mathematical operation where
![a\times b](https://latex.codecogs.com/svg.latex?a%5Ctimes%20b "a\times b")
returns a 1-dimensional or scalar number given that
![a](https://latex.codecogs.com/svg.latex?a "a") and
![b](https://latex.codecogs.com/svg.latex?b "b") have the same vector
length.

Let
![a = \[a_1, a_2, \cdots, a_n\]](https://latex.codecogs.com/svg.latex?a%20%3D%20%5Ba_1%2C%20a_2%2C%20%5Ccdots%2C%20a_n%5D "a = [a_1, a_2, \cdots, a_n]")
and
![b = \[b_1, b_2, \cdots, b_n\]](https://latex.codecogs.com/svg.latex?b%20%3D%20%5Bb_1%2C%20b_2%2C%20%5Ccdots%2C%20b_n%5D "b = [b_1, b_2, \cdots, b_n]").  

![a \cdot b = \sum^n\_{i=1} a_ib_i = (a_1+b_1) + (a_2+b_2) + \cdots + (a_n + b_n)](https://latex.codecogs.com/svg.latex?a%20%5Ccdot%20b%20%3D%20%5Csum%5En_%7Bi%3D1%7D%20a_ib_i%20%3D%20%28a_1%2Bb_1%29%20%2B%20%28a_2%2Bb_2%29%20%2B%20%5Ccdots%20%2B%20%28a_n%20%2B%20b_n%29 "a \cdot b = \sum^n_{i=1} a_ib_i = (a_1+b_1) + (a_2+b_2) + \cdots + (a_n + b_n)")

Dot products are useful summation and information retrieval tools for
one-hot vectors:

- The dot product of any one-hot vector with itself is 1.  
- The dot product of any one-hot vector with a different one-hot vector
  is 0.  
- The dot product of a one-hot vector and a vector of word or token
  weights is useful for calculating how strongly a word is represented.

``` python
# Create function to identify if a one-hot vector is the same or different -----
def identify_ohe_vector(vector_1: np.ndarray, vector_2: np.ndarray) -> None: 
  
  # Check if both inputs are numpy arrays
  if not isinstance(vector_1, np.ndarray) or not isinstance(vector_2, np.ndarray):
    raise TypeError("Both inputs must be numpy arrays")
  
  # Check if both arrays are 1D
  if vector_1.ndim != 1 or vector_2.ndim != 1:
    raise ValueError("Both numpy arrays must be 1D")
  
  # Check that both arrays have the same length
  if len(vector_1) != len(vector_2):
    raise ValueError("Numpy arrays must have the same length")
    
  # Dot product of any one-hot vector with itself is 1
  if vector_1 @ vector_2 == 1:
    return f"One-hot encoded vectors are the same"
  
  # Dot product of any one-hot vector with a different vector is 0  
  if vector_1 @ vector_2 == 0:
    return f"One-hot encoded vectors are different"

# Test function on simple vocabulary -------------------------------------------
identify_ohe_vector(
  np.array(find),
  np.array(find)
)
```

    'One-hot encoded vectors are the same'

``` python
identify_ohe_vector(
  np.array(find),
  np.array(my)
)
```

    'One-hot encoded vectors are different'

``` python
# Calculate the dot product of a one-hot vector and a vector of word weights ---
# The vector of word weights has the same word order as the one-hot vector
# Weight = word freq/total words but we will assign pretend weights below

weights = np.array([
  0.1, # files
  0.2, # find
  0.3, # my
  0.4  # please
  ]) 

# Extract pretend vector weights for "find"
# find = [0, 1, 0, 0]
np.array(find) @ weights
```

    np.float64(0.2)

# Matrix multiplication

The dot product is the building block of [matrix
multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication). If
A is
![m \times n](https://latex.codecogs.com/svg.latex?m%20%5Ctimes%20n "m \times n")
and B is
![n \times p](https://latex.codecogs.com/svg.latex?n%20%5Ctimes%20p "n \times p"),
then AB is
![m \times p](https://latex.codecogs.com/svg.latex?m%20%5Ctimes%20p "m \times p")
in dimensions.

Matrix multiplication can be used as a lookup table:

- Each row contains all the vector weights for a unique one-hot
  vector.  
- Each column is a vector of all the word or token weights generated
  from a unique corpus.

``` python
# Use matrix multiplication as a lookup table ----------------------------------
# Create the vector query matrix 
# Each row contains a unique one-hot vector of interest
simple_query = np.array([
  files,
  my
  ]) 

# Create the entire word weights matrix 
# The first column contains all vector weights from corpus 1 and so forth 
# A row contains all word weights for a unique one-hot vector 
# Weights matrix row order follows the same word order as the one-hot vector
weights = np.array([
  [0.1, 0.5], # files
  [0.3, 0.2], # find
  [0.9, 0.1], # local
  [0.2, 0.2], # my
])

# The dot product retrieves token weights for vectors in simple_query  
simple_query @ weights
```

    array([[0.1, 0.5],
           [0.2, 0.2]])

# First order sequence models

Language involves understanding a sequence rather than bag of words. We
can represent small word sequences using a transition model.

Imagine if our corpus contained 3 commands:

- Show me my directories please.  
- Show me my files please.  
- Show me my photos please.

``` python
# Explore simple commands vocabulary -------------------------------------------
command_1 = ["show", "me", "my", "directories", "please"]
command_2 = ["show", "me", "my", "files", "please"]  
command_3 = ["show", "me", "my", "photos", "please"]   

command_set = set(command_1)
command_set.update(command_2, command_3)

command_vocab = sorted(command_set)
command_vocab_length = len(command_set) 

# Output is wrapped inside print() for tidy print statement in Quarto
print(
  f"""
  Our new vocabulary size is {command_vocab_length}.
  Our new vocabulary comprises {command_vocab}.  
  """
  )
```


      Our new vocabulary size is 7.
      Our new vocabulary comprises ['directories', 'files', 'me', 'my', 'photos', 'please', 'show'].  
      

A bag of words approach does not retain information about word sequence
(the likelihood of word X being followed by word Y versus Z). A
transition model is helpful for representing word sequences.

![](https://e2eml.school/images/transformers/markov_chain.png)

The transition model above is a **first order Markov chain**, as the
probabilities for the next word depend on the single most recent word.

The first order Markov chain can be expressed in matrix form:

- Each row represents a unique word in the vocabulary.  
- Each column stores the probability of a specific word occurring after
  the word of interest.  
- Individual rows sum up to 1 as probabilities always sum up to 1.  
- All values fall between 0 and 1 as each value represents a
  probability.  
- The transition matrix above is a sparse matrix as there is only one
  place in the Markov chain where branching happens.

![](https://e2eml.school/images/transformers/transition_matrix.png)

``` python
# Pull out transition probabilities for word of interest -----------------------  
command_t_matrix = np.array(
  [
    [0, 0, 0, 0, 0, 1, 0], # directories
    [0, 0, 0, 0, 0, 1, 0], # files
    [0, 0, 0, 1, 0, 0, 0], # me
    [0.2, 0.3, 0, 0, 0.5, 0, 0], # my
    [0, 0, 0, 0, 0, 1, 0], # photos
    [0, 0, 0, 0, 0, 0, 0], # please
    [0, 0, 1, 0, 0, 0, 0] # show
  ]
)

# Look up a word of interest via its one-hot vector  
my = np.array([0, 0, 0, 1, 0, 0, 0])

# Extract probabilities of next word via matrix multiplication  
# 1x7 vector query @ 7x7 transition matrix to output row of interest
my @ command_t_matrix
```

    array([0.2, 0.3, 0. , 0. , 0.5, 0. , 0. ])

# Second order sequence probabilities

Our prediction of the next word improves when we predict based on the
last **two** words before our word of interest. This is called a
**second order sequence model**.

Imagine a corpus of the 2 computer commands:

- Check whether the battery ran down please  
- Check whether the program ran please

A second order sequence model improves the certainty of our predictions
compared to a first order sequence model. This can be seen in the second
order Markov chain and the second order transition matrix.

![](https://e2eml.school/images/transformers/transition_matrix_second_order.png)

The second order transition matrix has more 1s and fewer values between
0 and 1 than the first order transition matrix. In terms of
probabilities, we have increased the certainty of predicting the next
word.

A caveat is that our vocabulary size has also increased, from
![N](https://latex.codecogs.com/svg.latex?N "N") to
![N^2](https://latex.codecogs.com/svg.latex?N%5E2 "N^2") rows in the
transition matrix.

# Second order sequence model with skips

Sometimes we may have to look further than 2 words back to predict which
word comes next.

Imagine another corpus of 2 longer computer commands:  
+ Check the program log and find out whether it ran please  
+ Check the battery log and find out whether it ran down please

To predict the word after `ran`, we would have to look back 7 words
behind `ran`. Generating an eighth order transition matrix with
![N^8](https://latex.codecogs.com/svg.latex?N%5E8 "N^8") rows is too
cumbersome. To capture **long range word dependencies**, we can use a
second order sequence model that captures the 2-word combinations of a
word and the ![Nth](https://latex.codecogs.com/svg.latex?Nth "Nth")
words that come before it.

``` python
# Generate list of 2-word combinations using a selected word and skip range ----
def list_combinations(sentence: str, word: str, num: int) -> list[tuple[str, str]]:
    # Split the sentence into individual words and check for word presence
    split_sentence = sentence.split()
    if word not in split_sentence:
        raise ValueError("Selected word not found in sentence")
    
    # Identify position of selected word in sentence and set as last position
    index = split_sentence.index(word)
    
    # Identify position of first word to combine with selected word 
    start_index = max(0, index - num)
    
    # Use list comprehension to create a for loop to extract word pairs
    combinations = [(split_sentence[i], word) for i in range(start_index, index)]
    return combinations

# Generate example -------------------------------------------------------------
sentence = "Check the program log and find out whether it ran please"
word = "ran"
n = 8

list_combinations(sentence, word, n)
```

    [('the', 'ran'), ('program', 'ran'), ('log', 'ran'), ('and', 'ran'), ('find', 'ran'), ('out', 'ran'), ('whether', 'ran'), ('it', 'ran')]

We can visualise the relationship between each 2-word pair and the
following word of interest using the flow diagram below. A non-zero
weight indicates an occurrence of the word following the specific 2-word
pair. Larger non-zero weights are represented by thicker lines in the
diagram below.

![](https://e2eml.school/images/transformers/feature_voting.png)

This **second order sequence model with skips** can also be represented
by a transition matrix, except that the individual values in a matrix no
longer represent a probability.

![](https://e2eml.school/images/transformers/transition_matrix_second_order_skips.png)  
This is because each row no longer represents a unique position of the
sequence. A unique position is now represented by multiple rows or
**features**. For example, the sequence point `ran` is described by
multiple 2-word pairs such as `the, ran`, `program, ran` etc.

Each value in the matrix now represents a **vote**. Votes are summed
column-wise and compared to determine the next word prediction.

Most of the features also do not have any predictive power. Only the
features `battery, ran` and `program, ran` determine whether the next
word in the sequence is `down` or `please`.

To use a set of 2 word-pair features for next word prediction, we need
to:

1.  Extract all relevant 2-word pairs or features for a sequence of
    interest using a one-hot vector.  
2.  Matrix multiplication of the one-hot vector and transition matrix is
    equivalent to finding the column sums of all the features.  
3.  Choose the word with the highest vote as the predicted next word.

``` python
# Generate small example of 2 sequence word model with skips -------------------
# We want to predict the word following "is" from the sequence "green leaves 
# mean it is spring".    
command_1 = "yellow leaves mean it is autumn"  
command_2 = "green leaves mean it is spring"    

# Extract all unique 2-word pairs as the rows for the transition matrix  
combo_1 = list_combinations(command_1, "is", 4)
combo_2 = list_combinations(command_2, "is", 4)
combo_all = set(combo_1 + combo_2)

word_pairs = [list(tuple) for tuple in combo_all]
word_pairs_matrix = np.array(sorted(word_pairs))

# Extract all unique words as the columns for the transition matrix  
words_all = (command_1 + " " + command_2).split()
vocab = sorted(set(words_all))

# Manually construct the transition matrix    
# vocab = ['autumn', 'green', 'is', 'it', 'leaves', 'mean', 'spring', 'yellow']
command_t_matrix = np.array(
  [
    [0, 0, 0, 0, 0, 0, 1, 0], # green, is 
    [0.5, 0, 0, 0, 0, 0, 0.5, 0], # it, is
    [0.5, 0, 0, 0, 0, 0, 0.5, 0], # leaves, is
    [0.5, 0, 0, 0, 0, 0, 0.5, 0], # mean, is
    [1, 0, 0, 0, 0, 0, 0, 0], # yellow, is
  ]
)

# Step 1: extract relevant 2-word pairs for the sequence of interest 
# Our sequence of interest is "green leaves mean it is"  
command_2_features = np.array(
  [1, 1, 1, 1, 0]
)

# command_2_features is 1x5 and command_t_matrix is 5x8
command_2_features @ command_t_matrix
```

    array([1.5, 0. , 0. , 0. , 0. , 0. , 2.5, 0. ])

``` python

# The word with the highest vote of 2.5 is "spring"  
# This is our predicted next word  
```

# Masking

There is a major weakness to counting total votes for next word
prediction. Imagine an outcome where the top 2 votes were 10 and 9.
There is only a score difference of 0.1% between the selected versus
ignored word for prediction.

Using total votes can make predictions seem more uncertain than they
are, as votes from uninformative features are also counted. Very small
differences in total votes can then be lost in the statistical noise of
gigantic LLMs.

**Uninformative features** are features that assign equal probability to
all words in the vocabulary. In our example above, 3 out of 4 or 75% of
all features were uninformative!

We can sharpen our prediction by forcing uninformative features to 0, so
they cannot contribute to the overall vote. We do this by creating a
**mask**, which is a one-hot vector of the same length as the features
vector with uninformative features represented by 0s and informative
features represented by 1s.

![](https://e2eml.school/images/transformers/masked_feature_activities.png)

``` python
# Re-examine features for sequence of interest ---------------------------------
command_t_matrix = np.array(
  [
    [0, 0, 0, 0, 0, 0, 1, 0], # green, is 
    [0.5, 0, 0, 0, 0, 0, 0.5, 0], # it, is
    [0.5, 0, 0, 0, 0, 0, 0.5, 0], # leaves, is
    [0.5, 0, 0, 0, 0, 0, 0.5, 0], # mean, is
    [1, 0, 0, 0, 0, 0, 0, 0], # yellow, is
  ]
)

# Our sequence of interest is "green leaves mean it is"  
command_2_features = np.array(
  [1, 1, 1, 1, 0]
)

# Manually create command_2_features mask  
command_2_features_mask = np.array(
  [1, 0, 0, 0, 0]
)

# Multiply one-hot vectors element by element instead of using a dot product
# A mask is useful when the features vector contains values other than 0 and 1
command_2_features * command_2_features_mask   
```

    array([1, 0, 0, 0, 0])

``` python
# Obtain masked next word prediction  
(command_2_features * command_2_features_mask) @ command_t_matrix  
```

    array([0., 0., 0., 0., 0., 0., 1., 0.])

``` python

# Only the word "spring" has a vote of 1, which infinitely improves our 
# next word prediction weight difference (from 1.5 vs 2.5 to 0 vs 1).   
```

This process of selective masking forms the **attention** component of
transformers. Overall, the **selective second order with skips** model
is a useful way to think about what the decoder component of
transformers does.

# Rest Stop and an Off Ramp

When we think about implementing a useful model for sequence prediction,
there are three practical considerations:

- Computers are great at **matrix multiplications**. Expressing
  computation as a matrix multiplication is extremely efficient.  
- Every step needs to be **differentiable**, as all model parameters
  (i.e. transition probabilities and mask values) are learnt using [back
  propagation](https://www.youtube.com/watch?v=Ilg3gGewQ5U). For any
  small change in a parameter, we must be able to calculate the
  corresponding change in the model error.  
- The gradient needs to be **smooth** and **well conditioned** i.e. the
  slope doesn’t change very quickly when you make steps in any direction
  and changes are similar in every direction. This is a tricky condition
  to guarantee.

``` python
# Simple back propagation example ----------------------------------------------
```

# Attention as matrix multiplication

# Key messages

# Other resources
