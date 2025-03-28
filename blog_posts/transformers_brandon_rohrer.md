# Review of Transformers from Scratch
Erika Duan
2025-03-28

- [Original uses of transformers](#original-uses-of-transformers)
- [One-hot encoding](#one-hot-encoding)
- [The dot product](#the-dot-product)
- [Matrix multiplication](#matrix-multiplication)
- [First order sequence models](#first-order-sequence-models)
- [Key messages](#key-messages)
- [Other resources](#other-resources)

``` python
# Import Python libraries ------------------------------------------------------
import numpy as np
```

This is a review of the brilliant [transformers from scratch
tutorial](https://e2eml.school/transformers.html) by Brandon Rohrer.

**Note:** It is useful to think of words as fundamental units in natural
language processing tasks. In practice, algorithms use sub-words or
**tokens** due to increased representation flexibility and processing
efficiency.

# Original uses of transformers

Transformers were originally used for:

- Sequence transduction - converting one sequence of tokens into another
  sequence in another language.  
- Sequence completion - given a starting prompt, output a sequence of
  tokens in the same style.

Sequence transduction requires:  
+ A **vocabulary** - the set of unique tokens that form our language.
The vocabulary is created from the training data set.  
+ A method to convert unique tokens into unique numbers.  
+ A method of encoding token context - ensuring that the word “bank” in
“river bank” has a different numerical representation to the word “bank”
in “bank mortgage”.

# One-hot encoding

The simplest method of representing a vocabulary of words is **one-hot
encoding**.

- Each word is represented by a vector of mostly 0s and a single 1.  
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

  
Dot products are useful for one-hot word representations because they
are mathematically consistent.

- The dot product of any one-hot vector with itself is 1.  
- The dot product of any one-hot vector with a different vector is 0.  
- The dot product of a one-hot vector and a vector of word weights is
  useful for calculating how strongly a word is represented.

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
# Weight = word freq/ total words
weights = np.array([0.25, 0.25, 0.25, 0.25]) 

# Convert find into numpy array and calculate dot product
np.array(find) @ weights
```

    np.float64(0.25)

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

Matrix multiplication can be used as a lookup table. Each row in matrix
A acts as a vector query. Each column in matrix B stores the entire
vector of word weights generated from a unique corpus.

``` python
# Use matrix multiplication as a lookup table ----------------------------------

# Create the vector query matrix 
# Each row represents a unique one-hot vector of interest
simple_query = np.array([
  files,
  my
  ]) 

# Create the entire word weights matrix 
# The first column contains all vector weights from corpus 1  
# The first row contains all different vector weights for a specific vector 
# Vector order MUST be preserved  
weights = np.array([
  [0.1, 0.5],
  [0.3, 0.2],
  [0.9, 0.1],
  [0.2, 0.2], 
])

# Output lookup of specific vector weights 
# The dot product retrieves vector weights for vectors in simple_query  
simple_query @ weights
```

    array([[0.1, 0.5],
           [0.2, 0.2]])

# First order sequence models

Language involves understanding a sequence rather than bag of words. We
can represent small word sequences using a transition model.

# Key messages

# Other resources
