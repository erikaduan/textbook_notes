# Building RAPs with R - Part 1.6
Erika Duan
2025-06-09

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
  - [Use anonymous functions](#use-anonymous-functions)
  - [Write small functions that are easy to
    maintain](#write-small-functions-that-are-easy-to-maintain)
- [Use lists with functions](#use-lists-with-functions)
  - [Map functions across data
    frames](#map-functions-across-data-frames)
- [Useful R functions for functional
  programming](#useful-r-functions-for-functional-programming)

# Writing good functions

We can avoid code repetition by writing custom functions. For example,
the original code from script 2 step 2 in [Part 1.3](./raps_part_1_3.md)
can be simplified by the following function.

``` r
# Single function for plotting commune prices ----------------------------------  
# This function replaces repetitive code in script 2 step 2 from part 1.3  

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

- Every name in an environment is unique.  
- The names in an environment are not ordered.  
- Every environment has a parent environment, except the empty
  environment `emptyenv()`.  
- Unlike most R objects, environments are not copied when modified (when
  you modify an environment, you modify it directly in place).

The global environment is the interactive workspace that we program in
and its parent environment is actually the last package that we attached
using `library("package")`.

We can list the contents of our global environment using `ls()`. When we
start a fresh R session, we start with an empty global environment. When
we create an R object, we are actually assigning a name to an R object
in our environment. The state of our global environment is now altered.

``` r
# Start fresh R session and list global environment contents -------------------
# Manually restart R session to clean the global environment
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

# The global environment remains unchanged after we run add_ten()  
```

When an R object is covertly created inside a function (a property of a
bad function) and that function is run, the global environment is not
altered. This important global environment behaviour protects us from
having intermediate objects inside functions overwrite the objects in
our global environment.

``` r
# Run add_ten_times_two() ------------------------------------------------------
# This is a bad function as it creates a new object and binds it to the name y 
# covertly inside the function. There are no function arguments to indicate that
# a second variable y is used.

add_ten_then_double <- function(x) {
  y <- 2 # Bad practice as an object is covertly created inside the function   
  (x + 10) * y  
}

ls()
#> [1] "a"                 "add_ten"           "add_ten_then_double" 

add_ten_then_double(x = 3)
#> [1] 26

# Luckily, objects created inside functions cannot alter the global environment
# and we do not see a new object y inside our global environment.   
ls()
#> [1] "a"                 "add_ten"           "add_ten_then_double"    

# Run add_ten_then_double_worse() ----------------------------------------------
# The operator <<- is used to save object definitions that are made inside the 
# body of a function. 

add_ten_then_double_worse <- function(x) {
  y <<- 2 # Worst practice as y is saved inside the global environment      
  (x + 10) * y  
}

ls()
#> [1] "a"    "add_ten"    "add_ten_then_double"    "add_ten_then_double_worse"  

add_ten_then_double_worse(x = 3)  
#> [1] 26  

ls() 
#> [1] "a"    "add_ten"    "add_ten_then_double"    "add_ten_then_double_worse" 
#> [5] "y"        

# Running the function add_ten_then_double_worse() now changes our program state
# as our global environment contains a new value y.  
```

## Good functions are predictable

The mathematical definition of a function is a relation that always
assigns the same input element to the same output element. In
programming, this is a crucial property of a good function. An example
of a bad function is therefore one which does not generate the same
output for a given input.

``` r
# Run print_random_statement() -------------------------------------------------
# This function is bad as it outputs different values for the same input  
print_random_statement <- function(day_string) {
  weather_list <- c("sunny",
                    "cloudy",
                    "raining frogs",
                    "raining cats and dogs")
  
  paste0(day_string, " is ", sample(weather_list, 1))
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

print_good_random_statement <- function(day_string, seed) { 
  set.seed(seed)
  weather_list <- c("sunny",
                    "cloudy",
                    "raining frogs",
                    "raining cats and dogs")
  return(paste0(day_string, " is ", sample(weather_list, 1)))
  
  # Unset the seed otherwise the seed will stay set for the RSession
  set.seed(NULL) 
}

print_good_random_statement("tomorrow", seed = 111)
#> [1] "tomorrow is cloudy"

print_good_random_statement("tomorrow", seed = 111)
#> [1] "tomorrow is cloudy"  

# Note that if users forget to specify a seed, the function outputs the error
# message: Error in set.seed(seed) : argument "seed" is missing, with no default.
# This is good error handling as it prevents users from forgetting to set seed 
```

## Good functions are referentially transparent and pure

A function that only uses variable inputs pre-specified by function
arguments is referentially transparent. This is a feature of good
functions.

``` r
# Run opaque_sum() -------------------------------------------------------------
# This function is bad as the parameter y is not specified as a function 
# argument. This forces the function to look for an object named y in the global
# environment, which is unexpected behaviour and introduces an unpredictable 
# dependency.  
opaque_sum <- function(x) {sum(x + y)}  

y <- 4
opaque_sum(x = 3)
#> [1] 7  

y <- 1 
opaque_sum(x = 3) # The same function input now results in a different output!  
#> [1] 4 

# Run transparent_sum() --------------------------------------------------------
# This function is referentially transparent and much safer to use
transparent_sum <- function(x, y) {sum(x + y)}
transparent_sum(x = 3, y = 1) # A consistent output is guaranteed
#> [1] 4 
```

**Note:** `transparent_sum()` is also a pure function, as it does not
write anything to or require anything from the global environment. Pure
functions are referentially transparent by default.

The `withr` package can also be used to purify bad functions without
rewriting the function body.

``` r
# Alternatively use withr::with_seed() and print_random_statement() ------------
# Unpredictable i.e. bad function that outputs random responses
print_random_statement("tomorrow")
#> [1] "tomorrow is raining cats and dogs"

print_random_statement("tomorrow")
#> [1] "tomorrow is raining frogs"

# Predictable i.e. good version of the function
# This function is also pure as it ends with set.seed(NULL) so no new changes
# are introduced in the global environment.  
print_good_random_statement("tomorrow", seed = 111)
#> [1] "tomorrow is cloudy"

# withr::with_seed purifies bad functions without rewriting the function body 
withr::with_seed(seed = 111,
                 print_random_statement("tomorrow"))
#> [1] "tomorrow is cloudy"
```

# Using functions in a workflow

## Functions are first-class objects

When people refer to functions as first-class objects (in functional
programming), they are actually referring to the fact that functions are
treated just like other objects and can be manipulated as such. For
example, functions can take other functions as inputs similar to the way
that functions can take individual values as inputs.

**Higher order** functions are functions which take other functions as
an input.

``` r
# run_function() is an example of a higher order function ---------------------- 
# The input fun is actually another function i.e. to create fun()  
# The input function is then returned i.e. as fun(number)   

run_function <- function(number, fun) {
  fun(number)
}

run_function(number = 9, fun = sqrt)
#> [1] 3  

# Use fun(...) when you don't know how many arguments the input function has   
# The function sort() has a second argument decreasing = FALSE that we might not
# initially be aware of, but later decide to use.  

run_function(number = c(2, 5, 1), fun = sort)
#> [1] 1 2 5  

# As run_function() only has two arguments, we cannot make use of any additional
# arguments that our input function sort() has.  

# run_function(number = c(2, 5, 1), fun = sort, decreasing = TRUE) 
#> Error in run_function(number = c(2, 5, 1), fun = sort, decreasing = TRUE) : 
#>   unused argument (decreasing = TRUE)

# However, if we add the non-specific argument ... in our function, we can 
# access additional arguments from our input function.  
run_function_flexible <- function(number, fun, ...) {
  fun(number, ...) # We also need to refer to the location of ... in the body  
}

run_function_flexible(number = c(2, 5, 1), fun = sort, decreasing = TRUE) 
#> [1] 5 2 1  
```

As functions are first-class objects, we can also create functions that
return functions (instead of returning a data object like an integer or
string or data frame). An example provided by the textbook is a function
that converts warning into errors and stops code execution.

Functions that return other functions are called **function factories**
or **higher order** functions.

``` r
# Convert warnings into errors that stop code execution ------------------------
# We might want zero tolerance for warnings when the input is a negative number  
sqrt(-1)
#> Warning: NaNs produced [1] NaN

# Create higher order function that converts warnings into errors --------------
# tryCatch() catches warnings generated by fun(...) and instead stops the code 
strictly <- function(fun){
  function(...){ # Another function is returned inside the function body
    tryCatch({
      fun(...)
    },
    warning = function(warning)stop("Unexpected output. Code has stopped running."))
  }
}

strict_sqrt <- strictly(sqrt) # Creates a strict version of the sqrt function

class(strict_sqrt)
#> [1] "function"  

# strict_sqrt(-1)
#> Error in value[[3L]](cond) : Unexpected output. Code has stopped running. 
```

**Note:** Functions like
[`purrr::insistently()`](https://purrr.tidyverse.org/reference/insistently.html),
[`purrr::quietly()`](https://purrr.tidyverse.org/reference/quietly.html),
[`purrr::safely()`](https://purrr.tidyverse.org/reference/safely.html)
and
[`purrr::possibly()`](https://purrr.tidyverse.org/reference/possibly.html)
are also function factories that behave like the `strictly()` function
above.

## Function arguments can be optional

We can make function arguments optional by assigning `arg = NULL` as a
function argument and write two versions of the function i.e. one for
`is.null(arg)` and one for when the optional argument is specified.

``` r
# Create function with optional argument ---------------------------------------
transparent_sum <- function(x, y = NULL) {
  # When y is not provided
  if(is.null(y)) {
    print("Optional argument y is NULL")  
    x
  } else { 
    x + y
  }
}

transparent_sum(x = 0) # Only returns x as the function output
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
assertive programming. The use of assertive programming is subjective as
it also decreases the readability of the function body. An alternative
suggestion is to create specific functions to assert object properties
and run those assertions earlier in the script.

## Avoid recursive functions in R as they are slow

A function that calls itself in the function body is a recursive
function.

``` r
# Compare iterative and recursive functions ------------------------------------
# Create function to compute factorials  
# n! = 1 * 2 * ... * (n-1) * n  

# The downside to an iterative function is that it relies on constantly changing 
# the program state and requires an additional parameter i.  
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
    n * factorial_recursive(n - 1) # factorial_recursive() is called in the body
  }
}
```

``` r
# Perform microbenchmark on iterative versus recursive code --------------------
mark(
  factorial_iterative(10),
  factorial_recursive(10)
) 
```

## Use anonymous functions

Anonymous functions are functions which do not have a name assigned to
them. This may feel redundant as the function cannot be reused by the
global environment. However, anonymous functions are useful when you
need to map a function across multiple objects or are using a pipe to
perform a series of transformations.

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

## Write small functions that are easy to maintain

Small functions that only perform one task each are much easier to
maintain, test, document and debug. For example, instead of writing
`big_function()`, it is better to chain a series of smaller functions
together i.e. `a |> function_1() |> function_2() |> function_3()`.

# Use lists with functions

We want to write functions that handle lists, because lists are a
universal interface in R. This means that when we chain a series of
functions together, we are sure that they work in sequence as they are
manipulating the same object type. This explains why higher order
functions that apply a function across multiple objects, such as
`lapply()` or `purr::map()`, operate on lists.

In R, data frames, fitted models and ggplot objects are actually all
lists.

``` r
# Properties of lists ---------------------------------------------------------- 
typeof(datasets::mtcars)
#> [1] "list"

plot <- ggplot(datasets::mtcars, aes(x = mpg, y = hp)) +
  geom_line()

typeof(plot)
#> 1] "list"

# Extract objects from lists ---------------------------------------------------     
list_1 <- list(
  author = c("Erika", "Duan"),
  dataset = datasets::mtcars,
  plot = plot
)

# List objects can be subsetted into lists via [] and elements via [[]]

list_1[1]
#> $author
#> [1] "Erika" "Duan" 

list_1[[1]] 
#> [1] "Erika" "Duan" 

list_1[[1]][[1]] 
#> [1] "Erika" 

list_1$author # Produces same output as list_1[[1]]
#> [1] "Erika" "Duan" 

# Check list output type

typeof(list_1[1])
#> [1] "list"

typeof(list_1[[1]])
#> [1] "character"

typeof(list_1[[1]][[1]])
#> [1] "character"

typeof(list_1$author)
#> [1] "character"
```

Functions like `reduce()`, `lapply()` and `purrr:map()` work based on
the principle that we can use lists to create function factories (in the
function body) that remove the requirement for writing for loops.
Writing for loops is considered to be less reliable as it relies on
making iterative changes to the program state.

``` r
# looping() is equivalent to reduce() in behaviour ----------------------------- 
looping <- function(a_list, a_func, init = NULL, ...){

  # If the user does not provide an `init` value,
  # set the head of the list as the initial value
  if(is.null(init)){
    init <- a_list[[1]]
    a_list <- tail(a_list, -1)
  }

  # Separate the head from the tail of the list
  # and apply the function to the initial value and the head of the list
  head_list = a_list[[1]]
  tail_list = tail(a_list, -1)
  init = a_func(init, head_list, ...)

  # Check if we are done: if there is still some tail, then
  # rerun the whole thing until there is no tail left.
  if(length(tail_list) != 0){
    looping(tail_list, a_func, init, ...)
  }
  else {
    init
  }
}

# Run looping() instead of iterating across the list ---------------------------
# as.list(seq(1:10)) outputs a single value per list element
looping(as.list(seq(1:10)), sum)
#> [1] 55  
```

``` r
# applying() is equivalent to lapply() in behaviour ----------------------------
# Equivalent to applying a function to each element of a list
applying <- function(a_list, a_func, ...){
  
  # Separate the head from the rest i.e. tail of the list
  head_list = a_list[[1]]
  tail_list = tail(a_list, -1)
  result = a_func(head_list, ...)
  
  # Check if we are done: if there is still some tail, rerun the whole thing 
  # until there is no tail left.  
  if(length(tail_list) != 0){
    append(result, applying(tail_list, a_func, ...))
  }
  else {
    result
  }
}

# Run applying() instead of using a for loop on each list element --------------
applying(as.list(seq(1:10)), log10)
#> [1] 0.0000000 0.3010300 0.4771213 0.6020600 0.6989700 0.7781513 0.8450980 0.9030900 0.9542425 1.0000000
```

## Map functions across data frames

A data frame is a special type of list, where each column is a list of
atomic vectors of the same length. This means that we can easily map
functions across the columns of a data frame using `lapply()`,
`vapply()` or `purrr::map()`. The `purrr` package is extremely versatile
for functional programming and contains variations of
[`purrr::map()`](https://purrr.tidyverse.org/reference/map.html) such as
`purrr::map_chr()` and `purrr::walk()`.

**Note:** In contrast to `lapply()`, `purrr::map()` also accepts an
atomic vector of elements as its input.

``` r
# Apply a function across each column of a data frame --------------------------
vapply(mtcars[1:5], mean, FUN.VALUE = numeric(1))
#>        mpg        cyl       disp         hp       drat 
#>  20.090625   6.187500 230.721875 146.687500   3.596563 
```

By extension, we can also split a data frame into a series of smaller
data frames (or nested data frames) and apply the same function across
them using `mutate(object = purrr::map(data, fun))`. The list object
produced by each function is associated with its corresponding nested
data frame subset.

``` r
# Append information about data frame subsets using group_nest() ---------------
iris |>
  group_by(Species) |>
  filter(row_number() < 6)|> # Extract first 5 rows of data per species
  group_nest() |>
  mutate(col_names = purrr::map(data, colnames),
         col_sum = purrr::map(data, colMeans)) %>%
  knitr::kable() # Visualise outputs in tidy table format
```

| Species | data | col_names | col_sum |
|:---|:---|:---|:---|
| setosa | 5.1, 4.9, 4.7, 4.6, 5.0, 3.5, 3.0, 3.2, 3.1, 3.6, 1.4, 1.4, 1.3, 1.5, 1.4, 0.2, 0.2, 0.2, 0.2, 0.2 | Sepal.Length, Sepal.Width , Petal.Length, Petal.Width | 4.86, 3.28, 1.40, 0.20 |
| versicolor | 7.0, 6.4, 6.9, 5.5, 6.5, 3.2, 3.2, 3.1, 2.3, 2.8, 4.7, 4.5, 4.9, 4.0, 4.6, 1.4, 1.5, 1.5, 1.3, 1.5 | Sepal.Length, Sepal.Width , Petal.Length, Petal.Width | 6.46, 2.92, 4.54, 1.44 |
| virginica | 6.3, 5.8, 7.1, 6.3, 6.5, 3.3, 2.7, 3.0, 2.9, 3.0, 6.0, 5.1, 5.9, 5.6, 5.8, 2.5, 1.9, 2.1, 1.8, 2.2 | Sepal.Length, Sepal.Width , Petal.Length, Petal.Width | 6.40, 2.98, 5.68, 2.10 |

``` r
# Plot of Petal.Width versus Petal.Length per species using group_nest() -------
plot_scatterplot <- function(df, group) {
  ggplot(data = df, aes(x = Petal.Width, y = Petal.Length)) +
    geom_point() +
    lims(x = c(0, 3),
         y = c(0, 9)) + 
    labs(title = paste("Petal width versus petal length for", group)) +
    theme_minimal()
}

# Using group_nest instead of fill = group or facet_wrap(vars(group)) allows you
# to create 3 separate plots and store them together in a single list.    
nested_iris <- iris |> 
  group_nest(Species) |> 
  mutate(plots = purrr::map2(
    .x = data,
    .y = Species,
    .f = plot_scatterplot
  ))

nested_iris$plots
```

    [[1]]

<img src="raps_part_1_6_files/figure-commonmark/unnamed-chunk-23-1.png"
style="width:65.0%" />


    [[2]]

<img src="raps_part_1_6_files/figure-commonmark/unnamed-chunk-23-2.png"
style="width:65.0%" />


    [[3]]

<img src="raps_part_1_6_files/figure-commonmark/unnamed-chunk-23-3.png"
style="width:65.0%" />

This property of R is powerful and allows us to produce grouped
statistical models, as described further
[here](https://yuzar-blog.netlify.app/posts/2022-09-12-manymodels/).

``` r
# Model relationship between Petal.Width versus Petal.Length per species -------
# This can be a more intuitive way of presenting models than using 
# lm(Petal.Width ~ Petal.Length * Species, data = iris), which outputs the same
# model coefficients.  

nested_iris <- iris |> 
  group_nest(Species) |>
  mutate(lr_model = purrr::map(
    data,
    ~ lm(Petal.Width ~ Petal.Length, data = .x)
  ))

nested_iris$lr_model
```

    [[1]]

    Call:
    lm(formula = Petal.Width ~ Petal.Length, data = .x)

    Coefficients:
     (Intercept)  Petal.Length  
        -0.04822       0.20125  


    [[2]]

    Call:
    lm(formula = Petal.Width ~ Petal.Length, data = .x)

    Coefficients:
     (Intercept)  Petal.Length  
        -0.08429       0.33105  


    [[3]]

    Call:
    lm(formula = Petal.Width ~ Petal.Length, data = .x)

    Coefficients:
     (Intercept)  Petal.Length  
          1.1360        0.1603  

``` r
# Compare output with lm(Petal.Width ~ Petal.Length * Species, data = iris) ----
# You would need to manually calculate the slope and intercept for each Species
lm(Petal.Width ~ Petal.Length * Species, data = iris)
```

# Useful R functions for functional programming

Similar to the concept of `dplyr::filter()`, the base R function
`Filter()` works on lists and can be used with `Negate()` to return the
opposite condition.

``` r
# Apply Filter() to extract relevant list elements -----------------------------
list_2 <- list(
  a = 1:10,
  b = letters[1:10],
  c = TRUE
)

Filter(is.logical, list_2)
#> $c
#> [1] TRUE

Filter(Negate(is.logical), list_2)
#> $a
#>  [1]  1  2  3  4  5  6  7  8  9 10

#> $b
#>  [1] "a" "b" "c" "d" "e" "f" "g" "h" "i" "j"
```

The function `local()` allows you to run code in a temporary environment
that gets discarded in the end. This is useful when you need to run code
with potential side effects but want to avoid any interactions with the
global environment.

``` r
# Code run inside local() does not appear in the global environment ------------
var_1 <- 1

local({
  var_2 <- 1
})

exists("var_1")
#> [1] TRUE  

exists("var_2")
#> [1] FALSE
```
