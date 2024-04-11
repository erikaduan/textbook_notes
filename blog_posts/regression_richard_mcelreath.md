Review of Regression, Fire and Dangerous Things
================
Erika Duan
2024-04-11

-   <a href="#part-1---the-causal-salad"
    id="toc-part-1---the-causal-salad">Part 1 - the causal salad</a>

This is a review of the following blog posts:

-   Regression, Fire and Dangerous Things [Part
    1](https://elevanth.org/blog/2021/06/15/regression-fire-and-dangerous-things-1-3/)  
-   Regression, Fire and Dangerous Things [Part
    2](https://elevanth.org/blog/2021/06/21/regression-fire-and-dangerous-things-2-3/)  
-   Regression, Fire and Dangerous Things [Part
    3](https://elevanth.org/blog/2021/06/29/regression-fire-and-dangerous-things-3-3/)

# Part 1 - the causal salad

There are a few reasons for performing regression modelling, as listed
in [Regression and Other
Stories](https://avehtari.github.io/ROS-Examples/) (ROS) by Gelman et
al. 

-   Predicting or forecasting outcomes using existing data (does not aim
    to understand causality).  
-   Exploring associations between variables of interest and an
    outcome.  
-   Adjusting outcomes from a sample to infer something about a
    population of interest.  
-   Estimating treatment effects by comparing outcomes between a
    treatment and control (defined in ROS as causal inference).

McElreath emphasises that what scientific practitioners are often
looking for is a method to **separate spurious associations from true
causal relationships**. McElreath is critical of those who carelessly
use regression modelling to identify causal relationships.

**Scenario:**

We are interested in whether the family size of the mother has any
impact on the family size of the daughter.

-   We have data on the family sizes of mother and daughter pairs.  
-   We expect unmeasured confounds (shared environmental exposures) for
    mother and daughter pairs.  
-   Previous research indicates that birth order is associated with
    fertility, which impacts family size.

``` r
# Create a synthetic data model ------------------------------------------------
set.seed(111)

N <- 200 # Number of mother-daughter pairs
U <- rnorm(N) # Simulate confounds

U[1:5]
#> [1] -0.3832159 -0.6019343  0.8216938 -0.4526242  0.3254342

# B1 represents the mother's birth order where 1 indicates 'is first born'
B1 <- rbinom(N, size = 1, prob = 0.5) # Probability 50% are first born
B1[1:5]
#> [1] 0 1 0 0 0

# M represents the mother's family size
# In our model, the mother's family size is directly influenced by B1 and U
M <- rnorm(N, mean = 2*B1 + U) 
M[1:5]
#> [1]  0.37835160  0.03283694 -0.51075527 -4.15239295  1.66977563

M <- ifelse(M < 0, 0, round(M, digits = 0))
M[1:5]
#> [1] 0 0 0 0 2

# B2 represents the daughter's birth order where 1 indicates 'is first born' 
# B2 occurs independently of B1 
B2 <- rbinom(N, size = 1, prob = 0.5)
B2[1:5]
#> [1] 1 1 0 0 1 

# D represents the daughter's family size 
# We first assume that the mother's family size has no impact on the daughter's 
# family size.  
D <- rnorm(N, 2*B2 + U + 0*M) 
D[1:5]
#> [1]  0.4896820  2.4589650  0.5504082 -2.3242933  2.9083178  

D <- ifelse(D < 0, 0, round(D, digits = 0))
D[1:5]
#> [1] 0 2 1 0 3
```

In this example, the effect size of the confound is as large as the
causal effect of the daughter’s birth order on the daughter’s family
size.

![](regression_richard_mcelreath_files/figure-gfm/unnamed-chunk-3-1.png)

![](regression_richard_mcelreath_files/figure-gfm/unnamed-chunk-3-2.png)

![](regression_richard_mcelreath_files/figure-gfm/unnamed-chunk-3-3.png)

Our synthetic data model specifies that the mother’s family size has no
impact on the daughter’s family size. But what happens when we include
the mother’s family size in a regression model?

``` r
# Build linear regression model D = b0 + b1*M ---------------------------------- 
only_mother_family_size <- lm(D ~ M)

# Output tidy linear regression coefficients and p-values  
tidy(only_mother_family_size)
```

    # A tibble: 2 x 5
      term        estimate std.error statistic  p.value
      <chr>          <dbl>     <dbl>     <dbl>    <dbl>
    1 (Intercept)    0.978    0.118       8.26 2.03e-14
    2 M              0.231    0.0685      3.38 8.82e- 4

``` r
# Output model performance metrics
glance(only_mother_family_size)
```

    # A tibble: 1 x 12
      r.squared adj.r.squared sigma statistic  p.value    df logLik   AIC   BIC
          <dbl>         <dbl> <dbl>     <dbl>    <dbl> <dbl>  <dbl> <dbl> <dbl>
    1    0.0545        0.0497  1.23      11.4 0.000882     1  -324.  654.  663.
    # i 3 more variables: deviance <dbl>, df.residual <int>, nobs <int>

\#TODO \# Explain this result is because U contributes to both D AND M,
and therefore the component of U in M is wrongly attributed to M.
