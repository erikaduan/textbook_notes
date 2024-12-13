Review of Regression, Fire and Dangerous Things
================
Erika Duan
2024-12-13

-   <a href="#part-1-causal-salad" id="toc-part-1-causal-salad">Part 1:
    Causal Salad</a>
    -   <a href="#todo---what-if-the-impact-of-u-was-decreased"
        id="toc-todo---what-if-the-impact-of-u-was-decreased">TODO - what if the
        impact of U was decreased?</a>
-   <a href="#part-2-causal-design" id="toc-part-2-causal-design">Part 2:
    Causal Design</a>
-   <a href="#part-3-bayesian-inference"
    id="toc-part-3-bayesian-inference">Part 3: Bayesian Inference</a>
    -   <a href="#step-1-express-the-model-as-a-joint-probability-distribution"
        id="toc-step-1-express-the-model-as-a-joint-probability-distribution">Step
        1: Express the model as a joint probability distribution</a>
    -   <a href="#step-2-teach-the-distribution-to-a-computer"
        id="toc-step-2-teach-the-distribution-to-a-computer">Step 2: Teach the
        distribution to a computer</a>
    -   <a href="#step-3-simulate-causal-interventions"
        id="toc-step-3-simulate-causal-interventions">Step 3: Simulate causal
        interventions</a>
    -   <a href="#dealing-with-confounds"
        id="toc-dealing-with-confounds">Dealing with confounds</a>
-   <a href="#key-messages" id="toc-key-messages">Key messages</a>

This is a review of the following blog posts:

-   Regression, Fire and Dangerous Things [Part
    1](https://elevanth.org/blog/2021/06/15/regression-fire-and-dangerous-things-1-3/)  
-   Regression, Fire and Dangerous Things [Part
    2](https://elevanth.org/blog/2021/06/21/regression-fire-and-dangerous-things-2-3/)  
-   Regression, Fire and Dangerous Things [Part
    3](https://elevanth.org/blog/2021/06/29/regression-fire-and-dangerous-things-3-3/)

# Part 1: Causal Salad

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

Scientific practitioners are often looking for a method to **separate
spurious associations from true causal relationships**, but regression
modelling is not designed for this. McElreath is critical of those who
carelessly use regression modelling to identify causal relationships.

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

A diagram of the causal relationships in our synthetic data model can be
drawn **where m = 0**.

``` mermaid
flowchart LR  
  A(Mother birth order B1) --> B(Mother family size M) 
  C(Unknown confound U) --> B 
  
  D(Daughter birth order B2) --> E(Daughter family size D) 
  C --> E
  
  style E fill:#Fff9e3,stroke:#f96
```

Note that the effect size of the confounder (U) can be as large as the
effect of the daughter’s birth order (B2) on the daughter’s family size
(D).

![](regression_richard_mcelreath_files/figure-gfm/unnamed-chunk-4-1.png)

![](regression_richard_mcelreath_files/figure-gfm/unnamed-chunk-4-2.png)

![](regression_richard_mcelreath_files/figure-gfm/unnamed-chunk-4-3.png)

In our synthetic model, the mother’s family size (M) has **no impact**
on the daughter’s family size (D). But what happens when we include M in
a regression model to predict D?

``` r
# Build linear regression model D = b0 + b1*M ---------------------------------- 
only_M <- lm(D ~ M)

# Output tidy linear regression coefficients and p-values  
tidy(only_M) 
```

    # A tibble: 2 x 5
      term        estimate std.error statistic  p.value
      <chr>          <dbl>     <dbl>     <dbl>    <dbl>
    1 (Intercept)    0.978    0.118       8.26 2.03e-14
    2 M              0.231    0.0685      3.38 8.82e- 4

``` r
# Output model performance metrics
glance(only_M)
```

    # A tibble: 1 x 12
      r.squared adj.r.squared sigma statistic  p.value    df logLik   AIC   BIC
          <dbl>         <dbl> <dbl>     <dbl>    <dbl> <dbl>  <dbl> <dbl> <dbl>
    1    0.0545        0.0497  1.23      11.4 0.000882     1  -324.  654.  663.
    # i 3 more variables: deviance <dbl>, df.residual <int>, nobs <int>

The linear regression model indicates that M is positively associated
with D
i.e. ![E(D) = 0.98 + 0.23 M](https://latex.codecogs.com/svg.latex?E%28D%29%20%3D%200.98%20%2B%200.23%20M "E(D) = 0.98 + 0.23 M")!

This clearly conflicts with our ground truth that D is independent of M
and **should be incredibly scary to regression modelling
practitioners**. In the real world, it is very plausible that an unknown
confounder exists which is predictive of both predictor AND outcome
variables.

What happens if we add more variables into our linear regression model?
Does the misleading association between M and D disappear?

``` r
# Build linear regression model D = b0 + b1*M + b2*B1 + b3*B2 ------------------
M_B1_B2 <- lm(D ~ M + B1 + B2)

# Output tidy linear regression coefficients and p-values  
tidy(M_B1_B2) 
```

    # A tibble: 4 x 5
      term        estimate std.error statistic  p.value
      <chr>          <dbl>     <dbl>     <dbl>    <dbl>
    1 (Intercept)    0.400    0.132       3.02 2.88e- 3
    2 M              0.373    0.0658      5.67 5.15e- 8
    3 B1            -0.437    0.166      -2.64 9.07e- 3
    4 B2             1.33     0.145       9.16 6.85e-17

``` r
# Output model performance metrics
glance(M_B1_B2)
```

    # A tibble: 1 x 12
      r.squared adj.r.squared sigma statistic  p.value    df logLik   AIC   BIC
          <dbl>         <dbl> <dbl>     <dbl>    <dbl> <dbl>  <dbl> <dbl> <dbl>
    1     0.352         0.343  1.02      35.6 2.15e-18     3  -286.  582.  598.
    # i 3 more variables: deviance <dbl>, df.residual <int>, nobs <int>

Unfortunately, adding the variables B1 and B2 produced a model with an
even larger
![\beta](https://latex.codecogs.com/svg.latex?%5Cbeta "\beta")
coefficient for M!

B1 is also negatively associated with D, despite our synthetic model
stating that M is positively dependent on B1 (so we expect B1 and M to
have ![\beta](https://latex.codecogs.com/svg.latex?%5Cbeta "\beta")
coefficients with the same sign if M had a causal effect on D).

If we examined model performance metrics like AIC and BIC, we would be
misled into concluding that the second model was the better model and
use this one for scientific inference. It is very likely that the second
model is a more predictive model. However, the second model is also even
more misleading if we wanted to infer causal relationships between the
predictor (B1, B2 and M) and response (D) variables.

This example illustrates the dangers of causal salads, where we throw
many variables into a model and hope to identify some statistically
significant ones. The best way to counter this practice is to explicitly
think about the **causal relationships among predictor variables** and
not just the causal relationships between predictor and response
variables.

In this case, our scenario is the result of **bias amplification**:

-   The predictor variable of interest (M) is confounded by another
    variable (U). The response variable (D) is also confounded by U.  
-   Another predictor variable (B1) is further included in the model. B1
    is a strong predictor of M as `M <- rnorm(N, 2*B1 + U)`.  
-   The addition of B1 tends to amplify the effects of U and make
    inference much worse.  
-   In best practice, we should add additional predictor variables
    hypothesised to be strong predictors of D **but** not of M. In
    research however, it is usually very difficult to identify such
    variables confidently, especially when there is limited information
    about existing causal relationships.

``` r
# Build linear regression model D = b0 + b1*M + b2*B2 -------------------------- 
# Only include additional predictor variables known to be associated with D
M_B2 <- lm(D ~ M + B2)

# Output tidy linear regression coefficients and p-values  
tidy(M_B2) 
```

    # A tibble: 3 x 5
      term        estimate std.error statistic  p.value
      <chr>          <dbl>     <dbl>     <dbl>    <dbl>
    1 (Intercept)    0.282    0.126       2.23 2.70e- 2
    2 M              0.288    0.0582      4.94 1.63e- 6
    3 B2             1.33     0.148       8.99 2.06e-16

``` r
# The coefficients of the model D ~ M + B2 are more sensible than those of 
# D ~ M + B1 + B2, although the coefficient of M is still misleading.   
```

## TODO - what if the impact of U was decreased?

# Part 2: Causal Design

Regression has no direction whereas causal models are built from
directional relationships. The simplest model
![X \to Y](https://latex.codecogs.com/svg.latex?X%20%5Cto%20Y "X \to Y")
tells us that:

-   A change in X will cause a change in Y.  
-   A change in Y will not impact X.

We can turn our modelling question into the causal graph below. This
graph represents our hypothesis about what is happening, which is why we
include an arrow from M to D.

``` mermaid
flowchart TD  
  B1 -- b --> M  
  U -- k --> M  
  U -- k --> D   
  B2 -- b --> D
  
  M -- m --> D
  
  style D fill:#Fff9e3,stroke:#333
```

Other graph construction decisions:

-   We assume that the influence of birth order on family size ***b***
    is consistent over time (the same effect size for mothers and
    daughters). This assumption may come from domain knowledge or be a
    common sense simplification.  
-   We also assume that the influence of the unobserved confound ***k***
    is the same for mothers and daughters.

The covariance between two variables can be calculated directly from a
causal graph.

If ***b*** is the causal influence of B1 on M, then the covariance
between B1 and M is just the causal effect ***b*** multiplied by the
variance of B1.

![cov(B_1, M) = b \times var(B_1)](https://latex.codecogs.com/svg.latex?cov%28B_1%2C%20M%29%20%3D%20b%20%5Ctimes%20var%28B_1%29 "cov(B_1, M) = b \times var(B_1)")

![b = \tfrac{cov(B_1, M)}{var(B_1)}](https://latex.codecogs.com/svg.latex?b%20%3D%20%5Ctfrac%7Bcov%28B_1%2C%20M%29%7D%7Bvar%28B_1%29%7D "b = \tfrac{cov(B_1, M)}{var(B_1)}")

``` r
# Check causal influence estimation using a causal graph -----------------------
cov(B1, M) / var(B1)

#> [1] 1.246625  
```

``` r
# Check covariate estimation using linear regression ---------------------------
lm(M ~ B1) |>
  tidy()
```

    # A tibble: 2 x 5
      term        estimate std.error statistic  p.value
      <chr>          <dbl>     <dbl>     <dbl>    <dbl>
    1 (Intercept)    0.545     0.111      4.90 2.01e- 6
    2 B1             1.25      0.157      7.95 1.36e-13

We are interested in the causal influence of M on D and would therefore
like to estimate ***m***. We know the following things:

-   We do not know U or ***k*** as U is unobserved.  
-   We can calculate
    ![cov(B_1, D) = b\times m \times var(B_1)](https://latex.codecogs.com/svg.latex?cov%28B_1%2C%20D%29%20%3D%20b%5Ctimes%20m%20%5Ctimes%20var%28B_1%29 "cov(B_1, D) = b\times m \times var(B_1)")
    as we multiply the causes ***b*** and ***m*** when there are
    multiple arrows on a path.

We can therefore solve for ***m***.

![m = \tfrac{cov(B_1, D)}{b\times var(B_1)}](https://latex.codecogs.com/svg.latex?m%20%3D%20%5Ctfrac%7Bcov%28B_1%2C%20D%29%7D%7Bb%5Ctimes%20var%28B_1%29%7D "m = \tfrac{cov(B_1, D)}{b\times var(B_1)}")
where
![b = \tfrac{cov(B_1, M)}{var(B_1)}](https://latex.codecogs.com/svg.latex?b%20%3D%20%5Ctfrac%7Bcov%28B_1%2C%20M%29%7D%7Bvar%28B_1%29%7D "b = \tfrac{cov(B_1, M)}{var(B_1)}")

![m = \tfrac{cov(B_1, D)}{cov(B_1, M)}](https://latex.codecogs.com/svg.latex?m%20%3D%20%5Ctfrac%7Bcov%28B_1%2C%20D%29%7D%7Bcov%28B_1%2C%20M%29%7D "m = \tfrac{cov(B_1, D)}{cov(B_1, M)}")

``` r
# Solve for m using a causal graph ---------------------------------------------
cov(B1, D) / cov(B1, M)
#> [1] -0.02005616  

# This agrees with our prior knowledge that M has no effect on D  
```

``` r
# Test for different values of m -----------------------------------------------
set.seed(111)

D2 <- rnorm(N, 2*B2 + U + 0.7*M) 
D2 <- ifelse(D2 < 0, 0, round(D2, digits = 0))

cov(B1, D2) / cov(B1, M)
#> [1] 0.3991978

D3 <- rnorm(N, 2*B2 + U + 1.5*M) 
D3 <- ifelse(D3 < 0, 0, round(D3, digits = 0))

cov(B1, D3) / cov(B1, M)
#> [1] 1.19655
```

However, this solution does not provide us information about the
uncertainty of our estimate of ***m***. A computationally intensive way
of doing this is to repeat the data simulation many times and obtain a
bootstrap estimate. However, performing bootstraps is not always
possible.

``` r
# Calculate bootstrap estimate for 1000 simulations ----------------------------
set.seed(111)

N <- 200 
U <- rnorm(N, 0, 1) 
B1 <- rbinom(N, size=1, prob = 0.5) 
M <- rnorm(N, 2*B1 + U)
M <- ifelse(M < 0, 0, round(M, digits = 0))
B2 <- rbinom(N, size = 1, prob = 0.5)
D <- rnorm(N, 2*B2 + U + 0*M)
D <- ifelse(D < 0, 0, round(D, digits = 0))

# Define bootstrap statistic i.e. m = cov(B1, D) / cov(B1, M)  
f <- function(data,indices) 
  with(data[indices,], cov(B1, D) / cov(B1, M))

# Define dataset structure
data_sim <- data.frame(
  M = M,
  D = D,
  B1 = B1,
  B2 = B2
)

# Perform bootstrap
boot(data = data_sim, statistic = f, R = 1000) |>
  tidy()
```

    # A tibble: 1 x 3
      statistic     bias std.error
          <dbl>    <dbl>     <dbl>
    1   -0.0201 0.000772     0.151

Thinking like a graph involves multiple stages:

-   **Step 1:** Specify a generative model of the data and include
    unmeasured confounds. This generative model can be a simple directed
    acyclic graph (DAG) or a detailed system of equations.  
-   **Step 2:** Choose a specific exposure of interest and outcome of
    interest, for example M and D respectively.  
-   **Step 3:** Use the model structure to deduce a procedure for
    calculating the causal influence of the exposure on the outcome.  
-   **Step 4:** Repeat steps 2 and 3 if calculating multiple causal
    effects.

A single causal model implies a variety of statistical models, with
potentially a different statistical model for each causal question.

Even if we do not have any ideas about the exact functions between
different variables, we can use **do-calculus** to query a DAG and
determine if there is a method to estimate a causal effect.

The key ideas behind **do-calculus** are:

-   Use statistical or experimental design choices to remove confounds
    for an association of interest between an exposure and outcome.
    Graphically, this is equivalent to removing all arrows entering the
    exposure of interest. In medical research, we can achieve this
    scenario by conducting randomised controlled trials (RCTs) to
    experimentally set different values for M.  
-   The remaining association is an estimate of the causal effect of the
    exposure on the outcome.

In our scenario, we would be interested in modelling the intervention
below to calculate
![p(D\|do(M))](https://latex.codecogs.com/svg.latex?p%28D%7Cdo%28M%29%29 "p(D|do(M))"),
which is the distribution of D when we intervene on M.

``` mermaid
flowchart TD  
  U --> D   
  B2 --> D
  M --> D
  
  B1
  
  style D fill:#Fff9e3,stroke:#333
```

When RCTs cannot be conducted (for ethical or financial reasons),
do-calculus provides an algorithm for deducing statistical methods to
convert our original hypothesised model into the simplified model above
and to then calculate
![p(D\|do(M))](https://latex.codecogs.com/svg.latex?p%28D%7Cdo%28M%29%29 "p(D|do(M))").

# Part 3: Bayesian Inference

According to McElreath, **full-luxury Bayesian inference** is an
approach which:

-   Uses all variables and expresses all of their relationships as a
    joint probability distribution (only uses one statistical model
    unlike the causal graph approach).  
-   Any available data can be used to constrain the joint probability
    distribution, eliminate possibilities and refine information about
    causal effects.  
-   This process automatically realises and derives the statistical
    implications of the causal model.  
-   This process allows estimations in finite samples, with missing
    data, measurement errors or other common errors present.

## Step 1: Express the model as a joint probability distribution

We can rewrite our original generative code as a joint probability
distribution.

``` r
# Original generative model to be converted ------------------------------------
set.seed(111)

N <- 200 
U <- rnorm(N, 0, 1) 
B1 <- rbinom(N, size=1, prob = 0.5) 
M <- rnorm(N, 2*B1 + U)
M <- ifelse(M < 0, 0, round(M, digits = 0)) # [1]
B2 <- rbinom(N, size = 1, prob = 0.5)
D <- rnorm(N, 2*B2 + U + 0*M)
D <- ifelse(D < 0, 0, round(D, digits = 0)) # [1]

# [1] We will skip these lines to simplify our joint probability distribution
```

Let ![i](https://latex.codecogs.com/svg.latex?i "i") represent an
individual mother-daughter pair. The joint probability distribution is
derived from the following distinct probability distributions:

![M_i \sim Normal(\mu_i, \sigma)](https://latex.codecogs.com/svg.latex?M_i%20%5Csim%20Normal%28%5Cmu_i%2C%20%5Csigma%29 "M_i \sim Normal(\mu_i, \sigma)")
where
![\mu_i = \alpha_1 + bB\_{1,i} + kU_i](https://latex.codecogs.com/svg.latex?%5Cmu_i%20%3D%20%5Calpha_1%20%2B%20bB_%7B1%2Ci%7D%20%2B%20kU_i "\mu_i = \alpha_1 + bB_{1,i} + kU_i")

![D_i \sim Normal(\nu_i, \tau)](https://latex.codecogs.com/svg.latex?D_i%20%5Csim%20Normal%28%5Cnu_i%2C%20%5Ctau%29 "D_i \sim Normal(\nu_i, \tau)")
where
![\nu_i = \alpha_2 + bB\_{2,i} + mM_i+ kU_i](https://latex.codecogs.com/svg.latex?%5Cnu_i%20%3D%20%5Calpha_2%20%2B%20bB_%7B2%2Ci%7D%20%2B%20mM_i%2B%20kU_i "\nu_i = \alpha_2 + bB_{2,i} + mM_i+ kU_i")

![B\_{j,i} \sim Bernoulli(p)](https://latex.codecogs.com/svg.latex?B_%7Bj%2Ci%7D%20%5Csim%20Bernoulli%28p%29 "B_{j,i} \sim Bernoulli(p)")

![U_i \sim Normal(0,1)](https://latex.codecogs.com/svg.latex?U_i%20%5Csim%20Normal%280%2C1%29 "U_i \sim Normal(0,1)")

The values for U have not been observed so we cannot estimate the mean
and variance of U. However, it is fine to assign U a standardised normal
distribution. As U is an unobserved variable, it is a prior.

We also need to specify probability distributions for the latent
variables ***b***, ***m***, ***k*** and so on. These parameters are also
unobserved variables and therefore also priors. Assigning prior
probability distributions to unobserved variables is an art that can be
refined by simulating the observations implied by the prior probability
assignments. For our scenario, we will use weakly regularising
distributions that encode skepticism of large causal effects.

![\alpha_1, \alpha_2, b, m \sim Normal(0, 0.5)](https://latex.codecogs.com/svg.latex?%5Calpha_1%2C%20%5Calpha_2%2C%20b%2C%20m%20%5Csim%20Normal%280%2C%200.5%29 "\alpha_1, \alpha_2, b, m \sim Normal(0, 0.5)")  
![k, \sigma, \tau \sim Exponential(1)](https://latex.codecogs.com/svg.latex?k%2C%20%5Csigma%2C%20%5Ctau%20%5Csim%20Exponential%281%29 "k, \sigma, \tau \sim Exponential(1)")  
![p \sim Beta(2,2)](https://latex.codecogs.com/svg.latex?p%20%5Csim%20Beta%282%2C2%29 "p \sim Beta(2,2)")

The parameter ***k*** has been assigned an exponential distribution to
constrain it to be positive. Although we do not know if the effect of U
on M or D is positive or negative, we need to force it to be one or the
other as the sign of U impacts each M-D pair. This enforced constraint
helps with the next step.

## Step 2: Teach the distribution to a computer

The variables of our final joint probability distribution are:

-   the observed variables (the data)  
-   the latent variables (the unknown parameters)

Every time we acquire new data, we acquire new information about these
variables and we can update the joint distribution to see if it implies
new information about any of the other i.e. latent variables.

In our scenario, the information we have is observations of M, D, B1 and
B2. We want to know if these observations imply anything about ***m***.

``` r
# Load Bayesian modelling packages ---------------------------------------------
library(rethinking)
library(cmdstanr)

# Define Bayesian model --------------------------------------------------------
data <- list(N = N,
             M = M,
             D = D,
             B1 = B1,
             B2 = B2)

set.seed(111)

bayes_model <- ulam(
  alist(
    # Mum model
    M ~ normal(mu, sigma),
    mu <- a1 + b*B1 + k*U[i],
    
    # Daughter model
    D ~ normal(nu, tau),
    nu <- a2 + b*B2 + m*M + k*U[i],
    
    # B1 and B2
    B1 ~ bernoulli(p),
    B2 ~ bernoulli(p),
    
    # Unmeasured confound is also included
    vector[N]:U ~ normal(0, 1),
    
    # Priors or latent variables
    c(a1, a2, b, m) ~ normal(0, 0.5),
    c(k, sigma, tau) ~ exponential(1),
    p ~ beta(2, 2)
  ),
  data = data,
  chains = 4, 
  cores = 4, 
  iter = 2000, 
  cmdstan = TRUE)
```

The `cmdstanr` package uses automatic differentiation to sample from the
approximate joint distribution of the specified model, **conditional on
the observed data**.

``` r
# Summarise the marginal distributions of m, b and k ---------------------------
precis(bayes_model, pars = c("m", "b", "k")) 
```

            mean        sd       5.5%     94.5%     rhat  ess_bulk
    m 0.06425187 0.1190338 -0.1317059 0.2571193 1.019846  223.8204
    b 1.25957636 0.1005843  1.1013234 1.4215938 1.000825 3842.8306
    k 0.59038828 0.1671759  0.2837118 0.8159069 1.023639  188.5850

``` r
# Estimates for m, b and k are much closer to the original values used in our 
# generative model i.e. m = 0, b = 2 and k = 1.   
```

## Step 3: Simulate causal interventions

Imagine that we also want to know the causal effect of intervening on B1
on D
i.e. ![P(D \| do(B_1))](https://latex.codecogs.com/svg.latex?P%28D%20%7C%20do%28B_1%29%29 "P(D | do(B_1))").
This causal effect depends on multiple parameters (the product of
***b*** and ***m*** instead of only ***m***) and it can also be
computed.

``` r
# Directly calculate b*m from original Bayesian model --------------------------
# As the causal model is a linear model, the average effect of intervening on B1
# must be b * m.    

posterior <- extract.samples(bayes_model)
quantile(with(posterior, b*m))
```

             0%         25%         50%         75%        100% 
    -0.40037008 -0.01866835  0.08150841  0.18459810  0.55531662 

The median is 0.082 and there is a wide range of both positive and
negative effects, which indicates that there is no effect on D when we
intervene on B1. This is expected, as ***m*** is close to 0 and
uncertain.

``` r
# Simulate intervention on B1 and its effect on D ------------------------------
# B1 can only be 0 or 1 

# Scenario 1: Let B1 = 0  
# Simulate the distribution of M   
M_B1_0 <- with(posterior, a1 + b*0 + k*0)
# Use simulated values of M to simulate the distribution of D  
D_B1_0 <- with(posterior, a2 + b*0 + m*M_B1_0 + k*0)

# Scenario 2: Let B1 = 1
M_B1_1 <- with(posterior, a1 + b*1 + k*0)
D_B1_1 <- with(posterior, a2 + b*0 + m*M_B1_1 + k*0)

# Obtain distribution of D_B1_1 - D_B1_0 --------------------------------------- 
# This is the causal effect of changing B1 from 0 to 1    
dist_D_B1 <- D_B1_1 - D_B1_0
quantile(dist_D_B1)
```

             0%         25%         50%         75%        100% 
    -0.40037008 -0.01866835  0.08150841  0.18459810  0.55531662 

The quantiles calculated by the two methods are identical, as they are
computed from the same simulated samples from the joint probability
distribution.

## Dealing with confounds

An advantage of Bayesian inference is that it can automatically discover
ways to manage confounds like U.

Imagine we have a more complicated generative model where B1 also
influences D directly (for example, because first-born mothers tend to
gain an inheritance which can support their daughter’s future family).
We also have two new variables V and W, which are caused by U.

``` mermaid
flowchart TD  
  U --v--> V 
  U --w--> W 
  U --k--> D 
  U --k--> M
  
  B2 --b--> D
  M --m--> D
  
  B1 --b--> M
  B1 --d--> D
  
  style D fill:#Fff9e3,stroke:#333
```

B1 therefore becomes another confound that we have to control for when
calculating the causal effect of M on D. However, we can now use V and W
to remove the confounding effect of U.

``` r
# Define more complex Bayesian model -------------------------------------------
set.seed(111)

N <- 200 

# Simulate unobserved confound U and its effects on V and W
U <- rnorm(N, 0, 1) 
V <- rnorm(N, U, 1)
W <- rnorm(N, U, 1)

# Simulate birth order and family size
B1 <- rbinom(N, size = 1, prob = 0.5) 
M <- rnorm(N, 2*B1 + U)

B2 <- rbinom(N, size = 1, prob = 0.5)
D <- rnorm(N, 2*B2 + 0.5*B1 + U + 0*M)

# Define Bayesian model --------------------------------------------------------
# We no longer know the distributions of B1 and B2
complex_data <- list(
  N = N,
  M = M,
  D = D,
  B1 = B1,
  B2 = B2,
  V = V,
  W = W)

complex_bayes_model <- ulam(
    alist(
        M ~ normal(muM, sigmaM),
        muM <- a1 + b*B1 + k*U[i],
        
        D ~ normal(muD, sigmaD),
        muD <- a2 + b*B2 + d*B1 + m*M + k*U[i],
        
        W ~ normal(muW, sigmaW),
        muW <- a3 + w*U[i],
        
        V ~ normal(muV , sigmaV),
        muV <- a4 + v*U[i],
        
        vector[N]:U ~ normal(0, 1),
        
        # Still use weakly regularising distributions for priors
        c(a1, a2, a3, a4, b, d, m) ~ normal(0, 0.5),
        c(k, w, v) ~ exponential(1),
        c(sigmaM, sigmaD, sigmaW, sigmaV) ~ exponential(1)
        
    ), 
    data = complex_data,
    chains = 4, 
    cores = 4 , 
    iter = 2000 , 
    cmdstan = TRUE)
```

``` r
precis(complex_bayes_model)
```

                  mean         sd        5.5%      94.5%      rhat ess_bulk
    m       0.16960054 0.08188694  0.03960823 0.30279672 1.0012207 2147.273
    d       0.11476251 0.21787101 -0.21937491 0.46469234 1.0009631 3376.670
    b       1.95276557 0.11099411  1.77532945 2.12747330 1.0014163 4492.762
    a4      0.06231286 0.09396005 -0.09135014 0.21177960 1.0018730 1730.048
    a3     -0.09258681 0.09619545 -0.24360191 0.06319867 1.0015615 2133.334
    a2      0.02533199 0.14210585 -0.19491012 0.25011709 1.0002476 2927.299
    a1     -0.02933989 0.10772048 -0.20312861 0.14205601 1.0002731 2216.305
    v       1.03576492 0.09353699  0.89094659 1.18875650 1.0015247 1612.433
    w       0.94411181 0.09734003  0.79335103 1.10128210 0.9999713 2242.923
    k       0.97237904 0.09985011  0.81729938 1.13449495 1.0028707 1117.619
    sigmaV  0.85668174 0.07402561  0.73700181 0.97304161 1.0008793 1755.953
    sigmaW  1.01028818 0.06871882  0.90271051 1.12292165 1.0000127 3043.080
    sigmaD  1.12867762 0.07092363  1.02012395 1.24701320 0.9997250 4284.407
    sigmaM  0.98410645 0.07165032  0.87547982 1.10058695 1.0030935 2006.551

``` r
# Examine results from confounded multiple regression model --------------------
lm(D ~ M + B1 + B2 + V + W) |>
  tidy()
```

    # A tibble: 6 x 5
      term        estimate std.error statistic  p.value
      <chr>          <dbl>     <dbl>     <dbl>    <dbl>
    1 (Intercept)  -0.0982    0.160     -0.615 5.39e- 1
    2 M             0.380     0.0779     4.88  2.24e- 6
    3 B1           -0.258     0.232     -1.11  2.67e- 1
    4 B2            2.17      0.174     12.5   1.39e-26
    5 V             0.260     0.0836     3.11  2.17e- 3
    6 W             0.297     0.0755     3.93  1.18e- 4

``` r
# Multiple regression fails to account for confounding relationships and 
# provides a positive coefficient for m.  
```

A disadvantage of the Bayesian inference approach is that some
calculations are inefficient or impossible to run. Some algebraic
thinking can help make the inference more efficient and reliable.

For example, in our original model, we can ignore the impact of U by
treating M and D as pairs of values drawn from a common (multivariate
normal) distribution with some correlation induced by U.

The joint probability distribution therefore does not need to include U:

![\Bigl( \begin{matrix} M_i \\\\ D_i \end{matrix} \Bigr) \sim MVNormal \Bigl( \bigl( \begin{matrix} u_i \\\\ v_i \end{matrix} \bigr) , \Sigma \Bigr)](https://latex.codecogs.com/svg.latex?%5CBigl%28%20%5Cbegin%7Bmatrix%7D%20M_i%20%5C%5C%20D_i%20%5Cend%7Bmatrix%7D%20%5CBigr%29%20%5Csim%20MVNormal%20%5CBigl%28%20%5Cbigl%28%20%5Cbegin%7Bmatrix%7D%20u_i%20%5C%5C%20v_i%20%5Cend%7Bmatrix%7D%20%5Cbigr%29%20%2C%20%5CSigma%20%5CBigr%29 "\Bigl( \begin{matrix} M_i \\ D_i \end{matrix} \Bigr) \sim MVNormal \Bigl( \bigl( \begin{matrix} u_i \\ v_i \end{matrix} \bigr) , \Sigma \Bigr)")

![u_i = a_1 + bB\_{1,i}](https://latex.codecogs.com/svg.latex?u_i%20%3D%20a_1%20%2B%20bB_%7B1%2Ci%7D "u_i = a_1 + bB_{1,i}")

![v_i = a_2 + bB\_{2,i} + mM_i](https://latex.codecogs.com/svg.latex?v_i%20%3D%20a_2%20%2B%20bB_%7B2%2Ci%7D%20%2B%20mM_i "v_i = a_2 + bB_{2,i} + mM_i")

![B\_{j, i} \sim Bernoulli(p)](https://latex.codecogs.com/svg.latex?B_%7Bj%2C%20i%7D%20%5Csim%20Bernoulli%28p%29 "B_{j, i} \sim Bernoulli(p)")

``` r
# Define simplified Bayesian model ---------------------------------------------
set.seed(111)

# Sigma is the covariance matrix that attempts to learn the residual association
# between M and D due to U so U can be excluded

mvn_bayes_model <- ulam(
    alist(
        c(M, D) ~ multi_normal(c(mu, nu), Rho, Sigma),
        mu <- a1 + b*B1, 
        nu <- a2 + b*B2 + m*M,
        
        c(a1, a2, b, m) ~ normal(0, 0.5),
        Rho ~ lkj_corr(2),
        Sigma ~ exponential(1)
    ), 
    data = data, 
    chains = 4, 
    cores = 4, 
    cmdstan = TRUE)
```

``` r
precis(mvn_bayes_model, 3)
```

                   mean         sd        5.5%     94.5%     rhat  ess_bulk
    m        0.07435733 0.10559172 -0.09773703 0.2460944 1.005598  685.2339
    b        1.26509529 0.09926416  1.10734340 1.4294633 1.001187 1262.3625
    a2       0.54444654 0.15668036  0.30732971 0.7950777 1.004824  727.3945
    a1       0.51910548 0.08975037  0.37723323 0.6627066 1.005372 1314.2208
    Rho[1,1] 1.00000000 0.00000000  1.00000000 1.0000000       NA        NA
    Rho[2,1] 0.29534553 0.11466591  0.10145410 0.4706128 1.005545  734.8973
    Rho[1,2] 0.29534553 0.11466591  0.10145410 0.4706128 1.005545  734.8973
    Rho[2,2] 1.00000000 0.00000000  1.00000000 1.0000000       NA        NA
    Sigma[1] 1.11346473 0.05691709  1.02656340 1.2081238 1.000072 1937.2072
    Sigma[2] 1.08038981 0.06441891  0.98564164 1.1899950 1.008704  987.7994

Another problem with the Bayesian inference approach is that we do not
necessarily understand how our analysis extracts the final information
(as it is all hidden under complex probability theory). Understanding
why an inference is possible or not helps us anticipate results and
design better research.

The full-luxury Bayesian inference approach and the causal design
approach complement each other because 1) we can use do-calculus to
identify when inference is possible and which variables can be ignored,
and then 2) use Bayesian inference to perform the calculations. Both
models depend on the definition and analysis of a generative model. The
assumptions required to build a generative model must come from prior
research or hypotheses constructed using domain knowledge.

# Key messages

-   The interpretation of statistical results always depends upon causal
    assumptions, assumptions that ultimately cannot be tested with
    available data. This is why we need to interpret statistical
    modelling and machine learning results with caution and be skeptical
    of those making claims about causality. See [Westreich et al
    2013](https://academic.oup.com/aje/article/177/4/292/147738) for a
    more detailed example.  
-   Graphical causal inference (including RCTs) requires a different
    statistical model for each causal query.  
-   Bayesian inference only requires a single statistical model (the
    joint probability distribution) to obtain different causal queries
    through different simulations.  
-   Because we care about estimating uncertainty in a finite sample,
    using a causal design approach without incorporating Bayesian
    inference is incomplete, as we lack a reliable and efficient method
    to perform calculations.
