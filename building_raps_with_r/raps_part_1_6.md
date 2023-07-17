# Building RAPs with R - Part 1.6
Erika Duan
2023-07-17

- [Writing good functions](#writing-good-functions)
  - [Good functions do not alter the state of your global
    environment](#good-functions-do-not-alter-the-state-of-your-global-environment)
  - [Good functions are predictable](#good-functions-are-predictable)
  - [Good functions are referentially transparent and
    pure](#good-functions-are-referentially-transparent-and-pure)
- [Using functions in a workflow](#using-functions-in-a-workflow)
  - [Functions are first-class
    objects](#functions-are-first-class-objects)
  - [Function arguments can be
    optional](#function-arguments-can-be-optional)
  - [Write safe functions](#write-safe-functions)
  - [Avoid recursive functions in R as they are
    slow](#avoid-recursive-functions-in-r-as-they-are-slow)
  - [Use anonymous functions when you need to include them in a pipe
    once](#use-anonymous-functions-when-you-need-to-include-them-in-a-pipe-once)
  - [Write small functions are are easier to
    maintain](#write-small-functions-are-are-easier-to-maintain)
- [Using lists with functions](#using-lists-with-functions)

# Writing good functions

We can avoid code repetition by writing bespoke functions. For example,
the original code from script 2 step 2 in [Part
1.3](./raps_part_1_3.qmd) can be simplified by the following function.

``` r
# Single function for plotting commune prices ----------------------------------  
# This function can replace repetitive code in script 2 step 2 from part 1.3  

make_plot <- function(country_level_data,
                      commune_level_data,
                      commune){

  filtered_data <- commune_level_data %>%
    filter(locality == commune)

  data_to_plot <- bind_rows(
    country_level_data,
    filtered_data
  )

  ggplot(data_to_plot) +
    geom_line(aes(y = pl_m2,
                  x = year,
                  group = locality,
                  colour = locality))
}
```

Functional programming extends this concept by exclusively relying on
the evaluation of functions to achieve a required output. In order to
write a good functional program, however, we need to create functions
with specific ‘good’ properties.

## Good functions do not alter the state of your global environment

As described in the [Advanced R
textbook](https://adv-r.hadley.nz/environments.html), the role of an
environment is to bind a set of names to a set of R objects (which
include single values, atomic vectors, lists, data frames and
functions).

Environments have the following properties:  
+ Every name in an environment is unique.  
+ The names in an environment are not ordered.  
+ Every environment has a parent environment, except the empty
environment `emptyenv()`.  
+ Unlike most R objects, environments are not copied when modified (when
you modify an environment, you modify it directly in place).

The global environment is the interactive workspace that we program in
and its parent environment is actually the last package that we attached
using `library("package")`.

We can list the contents of our global environment using `ls()`. When we
start a fresh R session, we start with an empty global environment. When
we create an R object, we are actually assigning a name to an R object
in our environment and the state of our global environment is now
altered.

``` r
# Start fresh R session and list global environment contents -------------------
ls()
#> character(0)     

a <- 3

ls()
#> [1] "a" 

# Our environment now contains an R object bound to the name "a"  
```

When we first define a function (run the R code for a function), we bind
a name to the function and alter the state of our global environment.

``` r
# Create simple function -------------------------------------------------------
# After we define the function add_ten, it appears in our global environment  
add_ten <- function(x) {x + 10} 

ls()
#> [1] "a"       "add_ten"     
```

A good function does not alter the global environment (does not change
the state of our program) when it is run.

``` r
# Run add_ten() ----------------------------------------------------------------
ls()
#> [1] "a"       "add_ten"  

add_ten(x = 3)
#> [1] 13  

ls()
#> [1] "a"       "add_ten"       

# The global environment remains unchanged after we execute add_ten()  
```

Interestingly, when an R object is covertly created inside a function (a
property of a bad function) and that function is run, the global
environment is not altered. This important global environment behaviour
protects us from having intermediate objects inside functions overwrite
the objects in our global environment.

``` r
# Run add_ten_times_two() ------------------------------------------------------
# This is a bad function as it creates a new object and binds it to the name y 
# and this behaviour is hidden inside the function i.e. no function arguments to
# indicate that a second variable y is used.   

add_ten_times_two <- function(x) {
  y <- 2 # Bad practice as the dependency on y is hidden inside the function   
  (x + 10) * y  
}

ls()
#> [1] "a"                 "add_ten"           "add_ten_times_two" 

add_ten_times_two(x = 3)
#> [1] 26

ls()
#> [1] "a"                 "add_ten"           "add_ten_times_two"    

# Run add_ten_times_two_worse() ------------------------------------------------
# The operator <<- is used to save object definitions that are made inside the 
# body of a function. 

add_ten_times_two_worse <- function(x) {
  y <<- 2 # Worse practice as y is saved inside the global environment      
  (x + 10) * y  
}

ls()
#> [1] "a"    "add_ten"    "add_ten_times_two"    "add_ten_times_two_worse"  

add_ten_times_two_worse(x = 3)  
#> [1] 26  

ls() 
#> [1] "a"    "add_ten"    "add_ten_times_two"    "add_ten_times_two_worse"  "y"        

# Running the function add_ten_times_two_worse() now changes our program state
# as our global environment contains a new value y.
```

## Good functions are predictable

The mathematical definition of a function is a relation that always
assigns an input element to the same output element. In programming,
this is a crucial property of a good function. An example of a bad
function is therefore one which does not generate the same output for a
given input.

``` r
# Run print_random_statement() -------------------------------------------------
# This function is bad as it can output different values for the same input  

print_random_statement <- function(day) {
  weather_list <- c("sunny",
                    "cloudy",
                    "raining frogs",
                    "raining cats and dogs")
  paste0(day, " is ", sample(weather_list, 1))
}

print_random_statement("tomorrow")
#> [1] "tomorrow is sunny"   

print_random_statement("tomorrow")
#> [1] "tomorrow is raining frogs"  
```

A simple way of converting the function above into a good function is by
including `set.seed()` in the function body and allowing users to
specify a random seed in the function argument.

``` r
# Run print_good_random_statement() --------------------------------------------
# By specifying a random seed, outputs from random sampling are fixed and the 
# function output is predictable.  

print_good_random_statement <- function(day, seed) { 
  set.seed(seed)
  weather_list <- c("sunny",
                    "cloudy",
                    "raining frogs",
                    "raining cats and dogs")
  paste0(day, " is ", sample(weather_list, 1))
}

print_good_random_statement("tomorrow", seed = 111)
#> [1] "tomorrow is cloudy"

print_good_random_statement("tomorrow", seed = 111)
#> [1] "tomorrow is cloudy"  

# Note that if users forgot to specify a seed, the function outputs the error
# message: Error in set.seed(seed) : argument "seed" is missing, with no default
# This is good function behaviour as it prevents unpredictable functions from 
# executing by accident. 
```

## Good functions are referentially transparent and pure

A function that only uses variable inputs pre-specified by function
arguments is referentially transparent. This is a feature of good
functions.

``` r
# Run opaque_sum() -------------------------------------------------------------
# This function is bad i.e. not referentially transparent as the parameter y is 
# not specified as a function argument. This forces the function to look for an
# object named y in the global environment, which is an unstable dependency.  

opaque_sum <- function(x) {sum(x + y)}  

y <- 4
opaque_sum(x = 3)

#> [1] 7  

y <- 1 
opaque_sum(x = 3) # The same function input results in a different output!  
#> [1] 4 

# Run transparent_sum() --------------------------------------------------------
# This function is referentially transparent and therefore much safer to use

transparent_sum <- function(x, y) {sum(x + y)}
transparent_sum(x = 3, y = 1) # A consistent function output is guaranteed
#> [1] 4 
```

Note that the function `transparent_sum()` is also a pure function, as
it does not write anything to or require anything from the global
environment. Pure functions are referentially transparent by default.

# Using functions in a workflow

## Functions are first-class objects

When people refer to functions as first-class objects (in functional
programming), they are actually referring to the fact that functions are
treated just like other objects and can be manipulated as such. For
example, functions can take other functions as inputs similar to the way
that functions can take individual values as inputs.

Higher order functions are functions which take other functions as an
input.

``` r
# run_function() is an example of a higher order function ---------------------- 
# The input fun is actually another function i.e. to create fun()  

run_function <- function(number, fun) {
  fun(number)
}

run_function(number = 9, fun = sqrt)
#> [1] 3  

# Use fun(...) when you don't know how many arguments the input function has ---  
# The function sort() has a second argument decreasing = FALSE that we might not
# initially be aware of, but later decide to use.  

run_function(number = c(2, 5, 1), fun = sort)
#> [1] 1 2 5  

# As run_function() only has two arguments, we cannot make use of any additional
# arguments that our input function sort() has.  

# run_function(number = c(2, 5, 1), fun = sort, decreasing = TRUE) 
#> Error in run_function(number = c(2, 5, 1), fun = sort, decreasing = TRUE) : 
#>   unused argument (decreasing = TRUE)

# However, if we add the non-specific argument ... in our higher order function,
# we can access additional arguments from our input function.  

run_function_flexible <- function(number, fun, ...) {
  fun(number, ...) # We also need to refer to the location of ... in the body  
}

run_function_flexible(number = c(2, 5, 1), fun = sort, decreasing = TRUE) 
#> [1] 5 2 1  
```

As functions are first-class objects, we can create functions that
return functions (instead of returning an object like an integer or
string or data frame). An example provided by the textbook is a function
that converts warning into errors and stops code execution.

Functions that return other functions are called function factories.

``` r
# strict_sqrt() is a function that returns another function --------------------
# It is desirable for sqrt(-1) to fail if we are not working with complex numbers

sqrt(-1)
#> Warning: NaNs produced [1] NaN

# Create function that outputs a function with tryCatch() ----------------------
# tryCatch() catches warnings generated by fun(...) and outputs a function that
# raises an error via stop().  

strictly <- function(fun){
  function(...){ # Another function is returned inside the function body
    tryCatch({
      fun(...)
    },
    warning = function(warning)stop("Unexpected output. Code has stopped running."))
  }
}

strict_sqrt <- strictly(sqrt) # Returns another function 

class(strict_sqrt)
#> [1] "function"  

# strict_sqrt(-1)
#> Error in value[[3L]](cond) : Unexpected output. Code has stopped running.
```

## Function arguments can be optional

We can make function arguments optional by assigning `arg = NULL` as a
function argument and writing two versions of the function i.e. one for
`is.null(arg)` and one for when the optional argument is specified.

``` r
# Create function with optional argument ---------------------------------------
transparent_sum <- function(x, y = NULL) {
  # When y is not provided
  if(is.null(y)) {
    print("Optional argument y is NULL")  
    x
  } else { # Else refers to !is.null(y)
    x + y
  }
}

transparent_sum(x = 0)
#> [1] "Optional argument y is NULL"
#> [1] 0    

transparent_sum(x = 0, y = 1)
#> [1] 1  
```

## Write safe functions

A safe function is a function that handles silent conversions or
transformations transparently, so that a predictable output is always
returned. Unsafe functions can exist in base R and other packages, for
example `nchar()`.

``` r
# nchar() is unsafe as it outputs unexpected values for large numbers ----------
# nchar() returns a vector containing the size of input strings    
nchar(c("this", "is", "a", "string")) 
#> [1] 4 2 1 6  

# nchar() silently converts integers to strings which is unpredictable behaviour
nchar(100L)
#> [1] 3   

# We expect "100000" to be converted into a string of length 6
# However, 100000 is outputted as 1e+05 and then converted to "1e+05", which is 
# even more unpredictable behaviour.  
nchar(100000)
#> [1] 5
```

Writing explicitly safe functions is also referred to as the practice of
assertive programming.

## Avoid recursive functions in R as they are slow

A function that calls itself in the function body is a recursive
function.

``` r
# Compare iterative and recursive functions ------------------------------------
# n! = 1 * 2 * ... * (n-1) * n  

# The only downside to a recursive function is that it also contains an 
# additional parameter i.  
factorial_iterative <- function(n){
  result = 1
  for(i in 1:n) {
    result = result * i
    i = i + 1 
  }
  result
}

# Recursive functions are much slower in R and best to be avoided  
factorial_recursive <- function(n){
  if(n == 0 || n == 1) {
  result = 1
  } else {
    n * factorial_recursive(n - 1)
  }
}
```

## Use anonymous functions when you need to include them in a pipe once

Anonymous functions are functions which do not have a name assigned to
them. This may feel redundant as the function cannot be reused by the
global environment. However, anonymous functions are useful when you are
using a pipe to perform a series of transformations.

``` r
# Run anonymous function -------------------------------------------------------
(function(x) {x + 1})(3) # The second bracket contains the argument input  
#> [1] 4  

# Anonymous functions can be used with vapply() or purrr::map()  
vapply(
  list(1:5),
  (function(x) {x + 1}),
  FUN.VALUE = numeric(5)
)
#>      [,1]
#> [1,]    2
#> [2,]    3
#> [3,]    4
#> [4,]    5
#> [5,]    6

purrr::map(list(1:5), function(x) {x + 1})
#> [[1]]
#> [1] 2 3 4 5 6

# Anonymous functions can be used with native R pipes  
c(1:5) |>
  (function(x) {x + 1})()
#> [1] 2 3 4 5 6
```

## Write small functions are are easier to maintain

Small functions that only perform one task are easier to maintain, test,
document and debug. For example, instead of writing `big_function(a)`,
it is better to chain a series of smaller functions together
i.e. `a |> function_1() |> function_2() |> function_3()`.

# Using lists with functions

We want to write functions that handle lists, because lists are a
universal interface in R and therefore when we chain a series of
functions, we can be sure that they work in sequence as they are
manipulating the same object type.

In R, data frames, fitted models and ggplot objects are actually all
lists.

``` r
# Properties of lists ---------------------------------------------------------- 
```
