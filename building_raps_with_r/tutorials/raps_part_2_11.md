# Building RAPs with R - Part 2.11
Erika Duan
2025-06-19

- [Packaging your code](#packaging-your-code)
- [System setup for R package
  development](#system-setup-for-r-package-development)
- [R package states](#r-package-states)
  - [Source package](#source-package)
  - [Bundled package](#bundled-package)
  - [Binary package](#binary-package)
  - [Installed package](#installed-package)
  - [In-memory package](#in-memory-package)
  - [Package libraries](#package-libraries)
- [R Package design considerations](#r-package-design-considerations)
- [R Package creation process](#r-package-creation-process)

``` r
# Load required R packages -----------------------------------------------------
if (!require("pacman")) install.packages("pacman")
pacman::p_load(here,
               devtools,
               roxygen2,
               testthat,
               knitr)
```

# Packaging your code

When your analytical project moves into R package development mode, you
can use packages like `devtools` and `usethis` to improve the quality of
your analytical code. These improvements include:

- Automatically generating documentation for custom functions with
  `roxygen2` and `devtools::document()`  
- Unit testing for custom functions with `usethis::use_test_that()`,
  `usethis::use_test()` and the `testthat` package  
- Automatically defined external R package dependencies

Organising analytical code into a package is beneficial because packages
come with conventions (where to store code, tests and data).
Standardised conventions lead to standardised tools like `devtools`.

**Note:** Loading `devtools` also attaches all functions from `usethis`.

# System setup for R package development

To set up your operating system for optimal R package development:

- Make sure you have the latest version of R or at least R version
  4.5+.  

- Make sure you have a recent version of the RStudio IDE, as new IDE
  features are constantly being added.  

- You will need to attach `devtools` every time you start a new R
  session. This happens multiple times during R package development and
  can be annoying. To automatically attach `devtools` in your
  `.Rprofile` startup file, run `devtools::use_devtools()` and add the
  following code to your `.Rprofile` startup file.

  ``` r
  if (interactive()) {
  suppressMessages(require(devtools))
  }
  ```

- You can also store default package metadata and package development
  preferences in your `.Rprofile` startup file.

  ``` r
  options(
    usethis.description = list(
      "Authors@R" = utils::person(
        "Erika", "Duan",
        email = "erika.duan@example.com",
        role = c("aut", "cre"),
      ),
      License = "MIT + file LICENSE"
    )
  )
  ```

- Windows users also need to install
  [`Rtools`](https://cran.r-project.org/bin/windows/Rtools/). This is a
  toolchain bundle to build R packages that require C, C++ or Fortran
  code compilation from source. The easiest way to download or update
  Rtools is to run an R package development system setup check using
  `devtools::dev_sitrep()` within the RStudio IDE.

# R package states

## Source package

When we create or modify a package, we are working on its source code
and editing the source files. A source package is a directory of files
with a specific structure.

## Bundled package

A bundled package is a package that has been modified and reduced to a
single file (a `.tar` file) and then compressed using gzip (a `.gz`
file). Bundled packages generally have the extension `.tar.gz` and are
platform-agnostic and a transportation friendly intermediary between a
source and an installed package.

Every CRAN package is available in bundled form via the package source
field of its landing page. To unpack a bundled package, run the
following code in the terminal.

``` shell
tar xvf package_0.0.0.tar.gz
```

An uncompressed bundle is difference to a source package in the
following ways:

- Vignettes have been built and a vignette index exists in `./build`.  
- Bundles never contain temporary files.  
- Files listed in `.Rbuildignore` are not included in the bundle.

Each line of `.Rbuildignore` is a Perl-compatible regular expression
that matches (independent of case) to the path of a file in the source
package for exclusion from the bundled package.

The easiest way to modify `.Rbuildignore` is to use
`usethis::use_build_ignore()` as it takes care of details like regular
expression anchoring. The files that should be ignored are:

- Files that help to generate content, such as the `README.Rmd` file
  used to generate `README.md` and `.R` scripts to create or update
  package data set(s).  
- Files that aid package development and documentation, such as
  configuration files for CI/CD and RStudio IDE settings files.

## Binary package

The primary maker and distributor of binary packages in CRAN. A binary
file is a single platform-specific file (mostly specific to Windows or
macOS).

To make a binary package specific to an operating system (OS), use
`devtools::buid(binary = TRUE)` and later `R CMD INSTALL --build`.

When you use `install.packages()`, you are downloading binary R packages
for your relevant OS.

The difference between what the source, uncompressed bundled and
uncompressed binary R package contains is listed below.

![](https://r-pkgs.org/diagrams/package-files.png)

## Installed package

An installed package is a binary package that has been decompressed
locally into a package library. Binary R packages are usually installed
using `install.packages()` for CRAN R packages or
`pak::pkg_install("r_package")` for local, CRAN or Github R packages.

## In-memory package

The function `library(r_package)` loads the package namespace if it is
listed in the current libraries and attaches the package environment to
the search path (an ordered list of environments where R looks for
functions and objects). This allows you to call R package functions
directly i.e. without using `package::function()`.

Commonly used functions for converting between different R package
states are illustrated below.

![](https://r-pkgs.org/diagrams/install-load.png)

**Note:** The function `library(r_package)` does not load the package
functions into the global environment.

**Note:** For package development, `devtools::load_all()` helps you to
load a source package directly into memory, bypassing package
installation requirements.

## Package libraries

In R, a library is a directory containing installed packages. Libraries
are associated with a specific version of R. You can use `.libPaths()`
to identify which libraries are currently active.

``` r
# List currently active libraries ----------------------------------------------
.libPaths()
#> [1] "C:/Users/Erika/AppData/Local/R/win-library/4.5"
#> [2] "C:/Program Files/R/R-4.5.0/library"  

# Identify user library path ---------------------------------------------------
Sys.getenv("R_LIBS_USER")
#> [1] "C:\\Users\\Erika\\AppData\\Local/R/win-library/4.5"
```

There are usually 2 active libraries:

- A **user** library. Add-on packages installed from package
  repositories like CRAN or under local development are kept here.  
- A system-level or **global** library. The core set of base and
  recommended packages that ship with R are kept here.

The separation allows users to clean out their add-on packages without
disturbing their base R installation.

**Note:** The function `library()` should never be used inside a package
as packages rely on a different mechanism for declaring package
dependencies.

# R Package design considerations

TODO

# R Package creation process

This tutorial will cover the [R packages textbook](https://r-pkgs.org/)
by Hadley Wickham and Jennifer Bryan instead of the R package `fusen`.

Briefly, the steps for R package development are:

1.  Load the devtools package using `library(devtools)`.

``` r
# Check devtools package version -----------------------------------------------
packageVersion("devtools")  
```

2.  Create your functions, function documentation, unit tests and
    package documentation.

3.  Use `usethis::create_package` to initialise a new package in a local
    directory. This directory should be somewhere in your home
    directory, but **not** nested in another Git repository or R
    project.

``` r
# Create your R package --------------------------------------------------------
usethis::create_package("~/path/to/package")
```

4.  Once your package is created (inside RStudio), a new instance of
    RStudio should appear and open into your new package. You will need
    to run `library(devtools)` again as this is a new R session.

The following new files (including hidden files) will have been created:

- `.Rbuildignore` - stores a list of the files that we need to use but
  should not include when building the R package from source.  
- `.gitignore` - a standard `git ignore` template.  
- `DESCRIPTION` - a file with metadata about your package.  
- `NAMESPACE` - a file which declares the functions your package exports
  for external use and the external functions your package imports from
  other packages. This should not be edited by hand.  
- `R` - the key package directory which will contain `.R` files with
  your function definitions.  
- `new_package.Rproj` - a standard RStudio project file. Can be
  suppressed with `create_package(..., rstudio = FALSE)`.

5.  Convert your R source package and RStudio project into a git
    repository with `usethis::use_git()`. RStudio may request permission
    to launch itself in your project.

6.  Write your custom R package functions. Make a new `.R` file for each
    user-facing function and name the file after the function. When you
    add more functions, you will want to relax this rule and group
    related functions together. Use `usethis::use_r("custom_function")`
    to create or open an R script below the `.R/` directory.

The R script file should only contain your function and not any
top-level code for R package development like `library(devtools)` or
`usethis::use_git()`.

7.  Use `devtools::load_all()` to load all custom functions. Functions
    are now available for experimentation but **do not** exist in the
    global environment. To test this, run
    `exists("custom_function", where = globalenv(), inherits = FALSE)`
    and you should see `FALSE`.

8.  Remember to use `git commit` for every R script of custom functions
    that you create.

9.  Intermittently check that all the components of your package still
    work using `devtools::check()`.

10. Edit the `DESCRIPTION` file. Add yourself as the package author and
    write some descriptive text in the `Title` and `Description` fields.
    The `License` field should not be manually modified.

11. Configure a valid package license using, for example
    `usethis::use_mit_license()`.

12. Help documentation for custom packages are stored in
    `R/man/custom_function.Rd` and written in an R-specific markup
    language. Use `devtools::document()` the `roxygen2` package to
    automate the creation of help documentation. You can insert an
    `roxygen2` template above your custom function inside RStudio via
    *`Code > Insert roxygen skeleton`*. An example of what this template
    looks like is below.

``` r
#' Split a string
#'
#' @param x A character vector with one element.
#' @param split What to split on.
#'
#' @return A character vector.
#' @export
#'
#' @examples
#' x <- "alfa,bravo,charlie,delta"
#' strsplit1(x, split = ",")
strsplit1 <- function(x, split) {
  strsplit(x, split = split)[[1]]
}
```

Use `devtools::document()` to:

- Convert individual roxygen comments into individual
  `R/man/custom_function.Rd` files.  
- Update the `NAMESPACE` file based on the `@export` tags in roxygen
  comments to include `export(custom_function)`. The export directive in
  `NAMESPACE` is what makes custom functions available to users after
  they attach your package via `library(custom_package)`.

13. Check that all the components of your package still work using
    `devtools::check()`.

14. Once you have created a minimal viable package with documented and
    exported functions and complete package metadata, install the
    package into your library via `install()`. An R CMD build will then
    be completed.

To check that package installation is successful, restart your R session
and attach your new library using `library(custom_function)`.  
15. To create unit tests, first declare your intent to write unit tests
using `usethis::use_test_that()`. This creates your unit testing
directory and `testthat.R` script, and suggests the package `testthat`
as a package dependency.

To create or open a test file, make sure your the R script containing
your custom function(s) is open and run `usethis::use_test()`. Add unit
test(s) for each R script into each test file. Functions like
`testthat::expect_equal()` are useful for writing unit tests.

**Note:** The function `testthat::expect_equal()` relies on
waldo::compare() in `testthat` version 3+.

``` r
# Unit test example ------------------------------------------------------------
test_that("strsplit1() splits a string", {
  expect_equal(strsplit1("a,b,c", split = ","), c("a", "b", "c"))
})
```

Run all unit tests using `devtools::test()`. Unit tests also run
whenever you run `devtools::check()`.

Often during development, you may need to rename an R script. You can
use `usethis::rename_files(old_function, new_function)` to rename both
the R script and its associated unit testing script.

16. When you need to use a function from another package in your own
    package, use `usethis::use_package("existing_package")` to declare
    these external package requirements. External packages will then be
    added to the `Imports` field of `DESCRIPTION`.

When calling functions from external packages in your custom functions,
follow the `package::function()` coding style.

17. Every time you update your package R scripts, call
    `devtools::document()` to update `NAMESPACE` and helper R
    documentation. Then call `devtools::load_all()` to make your updated
    custom functions available for experimentation.

18. To document your package, use `usethis::use_readme_rmd()` to create
    a basic `README.Rmd` file. This function also creates a Git
    pre-commit hook to help keep `README.Rmd` and `README.md` in sync.

The `README.Rmd` file has sections to prompt users to:

- Describe the package purpose  
- Provide installation instructions  
- Demonstrate package functions

Once you have edited your `README.Rmd` file, use
`devtools::build_readme()` to carefully render your `README.md` file
with the most current version of your package.

19. Run `devtools::check()` at the end of package development. You can
    iteratively re-build and install your package locally using
    `devtools::install()`.

A flow chart of the R package development process using `devtools` is
found below.

![](https://r-pkgs.org/diagrams/workflow.png)
