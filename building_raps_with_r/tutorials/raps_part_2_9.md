# Building RAPs with R - Part 2.9
Erika Duan
2025-06-10

- [Rewriting our analytical project](#rewriting-our-analytical-project)
  - [Script 1](#script-1)
  - [Script 2](#script-2)
- [Other resources](#other-resources)

# Rewriting our analytical project

## Script 1

We now return back to the example code from [part
1.3](./raps_part_1_3.md). Interestingly, the author Bruno Rodrigues
prefers to develop solutions in Rmd or Qmd notebooks rather than R
scripts (as Rmd/Qmd code is easier to read and share than a script). His
revised notebook solution can be found
[here](https://raw.githubusercontent.com/b-rodrigues/rap4all/master/rmds/save_data.Rmd).

Script 1 features steps for raw data set extraction and data set
cleaning. I have changed some data object names so no data objects are
rewritten during the data cleaning steps.

The first two steps from the original code are below.

``` r
# ==============================================================================
# Script 1: Generate machine-friendly, cleaned and transformed data set   
# ==============================================================================

library(dplyr)
library(readxl)
library(purrr)
library(stringr)
library(janitor)
library(here)

# ------------------------------------------------------------------------------
# Original step 1: Download raw Excel file from GitHub and transform multiple 
# sheets into a single data frame with a new column for year.    

url <- "https://github.com/b-rodrigues/rap4all/raw/master/datasets/vente-maison-2010-2021.xlsx"

temp_raw_data <- tempfile(fileext = ".xlsx")

download.file(url, temp_raw_data, method = "auto", mode = "wb")

sheets <- excel_sheets(temp_raw_data)

# Function to create a single data frame from a named sheet and then add a 
# column to denote year. This can be done because individual sheets are named 
# after individual years.      

read_clean <- function(..., sheet){
  read_excel(..., sheet = sheet) |>
    mutate(year = sheet)
}

# Concatenate all individual spreadsheet tables into a single table
raw_data <- map(
  sheets,
  ~read_clean(temp_raw_data,
              skip = 10,
              sheet = .)
) |>
  bind_rows() |>
  clean_names()

# ------------------------------------------------------------------------------
# Original step 2: Clean and select required data frame column names and check 
# the cleaned output.

renamed_raw_data <- raw_data |>
  rename(
    locality = commune,
    n_offers = nombre_doffres,
    average_price_nominal_euros = prix_moyen_annonce_en_courant,
    average_price_m2_nominal_euros = prix_moyen_annonce_au_m2_en_courant,
    average_price_m2_nominal_euros = prix_moyen_annonce_au_m2_en_courant
  ) |>
  mutate(locality = str_trim(locality)) |>
  select(year, locality, n_offers, starts_with("average"))  

str(renamed_raw_data)
```

The first step of the **revised code** is below.

The changes are:

- The addition of a detailed comment to describe the data set source.  
- Conversion of the original code into a single large function to
  download the source data set and to clean its column names. This
  seemed to go against previous advice to break down functions into
  small functions that each perform a single task and felt extremely
  messy (especially the nested function). It is possible that this final
  structure is useful for packages like `targets`.  
- Removal of the code `str(raw_data)` to check the properties of the
  cleaned raw data set.

``` r
# ==============================================================================
# Script 1: Generate machine-friendly, cleaned and transformed data set   
# ==============================================================================

library(dplyr)
library(readxl)
library(purrr)
library(stringr)
library(janitor)
library(here)

# ------------------------------------------------------------------------------
# Revised step 1: Download raw Excel file from GitHub and transform multiple 
# sheets into a raw data set containing the columns year, locality, n_offers and
# price related measurements.     

# This data is downloaded from the Luxembourguish Open Data
# Portal (https://data.public.lu/fr/datasets/prix-annonces-des-logements-par-commune/). 
# The data set is called 'Série rétrospective des prix annoncés des maisons par 
# commune, de 2010 à 2021', and the original data is from the 'Observatoire de
# l'habitat'. This data contains prices for houses sold since 2010 for each 
# Luxembourguish commune. 

# The function below uses the permanent URL from the Open Data Portal to access 
# the data, but the author has also re-hosted the data and used his link to 
# download the data for archival purposes. 

get_raw_data <- function(url) {
  temp_raw_data <- tempfile(fileext = ".xlsx")
  
  download.file(url,
                temp_raw_data,
                mode = "wb")
  
  sheets <- excel_sheets(temp_raw_data)
  
  # Function read_clean() is nested within get_raw_data() 
  read_clean <- function(..., sheet){
    read_excel(..., sheet = sheet) %>%
      mutate(year = sheet)
  }
  
  # read_clean() is then mapped across all Excel spreadsheets 
  raw_data <- map_dfr(
    sheets,
    ~read_clean(temp_raw_data,
                skip = 10,
                sheet = .)) %>%
    clean_names()
  
  # Data cleaning steps included within the single function
  renamed_raw_data <- raw_data %>%
    rename(
      locality = commune,
      n_offers = nombre_doffres,
      average_price_nominal_euros = prix_moyen_annonce_en_courant,
      average_price_m2_nominal_euros = prix_moyen_annonce_au_m2_en_courant,
      average_price_m2_nominal_euros = prix_moyen_annonce_au_m2_en_courant
    ) %>%
    mutate(locality = str_trim(locality)) %>%
    select(year, locality, n_offers, starts_with("average"))
}

# Intermediate data objects like temp_raw_data are not stored in the global
# environment when we run the single function get_raw_data() 
raw_data <- get_raw_data(url = "https://github.com/b-rodrigues/rap4all/raw/master/datasets/vente-maison-2010-2021.xlsx")
```

The next step of the original code is below.

``` r
# ------------------------------------------------------------------------------
# Original step 3: Check string values in the locality column and normalise 
# the spelling.  

# As well as multiple spelling version, we also have NAs and long string values
# in the locality column 
renamed_raw_data |>
  dplyr::filter(grepl("Luxembourg", locality)) |>
  dplyr::count(locality)

# Both Pétange and Petange exist and are coded as "P?tange" instead   
renamed_raw_data |>
  dplyr::filter(grepl("P.tange", locality)) |>
  dplyr::count(locality)

no_typo_renamed_raw_data <- renamed_raw_data |>
  mutate(locality = ifelse(grepl("Luxembourg-Ville", locality),
                           "Luxembourg",
                           locality),
         locality = ifelse(grepl("P.tange", locality),
                           "P?tange",
                           locality)) |>
  mutate(across(starts_with("average"), as.numeric)) |> 
  filter(!grepl("Source", locality)) 
```

The next step of the **revised code** is below.

The changes are:

- The addition of a detailed comment to describe the inconsistencies in
  the raw dataset that need to be addressed.  
- Conversion of the original code into a single function to clean the
  raw data set.  
- Removal of exploratory code to check specific locality spelling
  examples.

**Note:** Each object produced by each large function has a different
value attached to it, so that objects are not re-written in this
workflow.

``` r
# ------------------------------------------------------------------------------
# Revised step 2: Clean the raw dataset by normalising spelling and removing
# rows of data at the country rather than locality level. 

# We need to clean the data: "Luxembourg" is "Luxembourg-ville" in 2010 and 
# 2011 but then "Luxembourg". "Pétange" is also spelled non-consistently and we 
# also need to convert columns to the right type. We also directly remove rows 
# where the locality contains information on the "Source" as these contain NAs.   

clean_raw_data <- function(raw_data) {
  raw_data %>%
    mutate(locality = ifelse(grepl("Luxembourg-Ville", locality),
                             "Luxembourg",
                             locality),
           locality = ifelse(grepl("P.tange", locality),
                             "Pétange",
                             locality)) %>%
    filter(!grepl("Source", locality)) %>% 
    mutate(across(starts_with("average"), as.numeric))
}

flat_data <- clean_raw_data(raw_data)
```

The next step of the original code to generate separate commune and
country level data sets is below.

``` r
# ------------------------------------------------------------------------------
# Original step 4: Separate commune and country level data for price

# Examine all rows with missing values for average_price_nominal_euros  
no_typo_renamed_raw_data |>
  filter(is.na(average_price_nominal_euros))

# Keep commune level data
commune_level_data <- no_typo_renamed_raw_data |>
  filter(!grepl("nationale|offres", locality),
         !is.na(locality))

# Keep country level data which is recorded as 'Moyenne nationale' under locality
country_level <- no_typo_renamed_raw_data |>
  filter(grepl("nationale", locality)) |>
  select(-n_offers)

# n_offers at the country level is recorded as 'Total d.offres' under locality 
offers_country <- no_typo_renamed_raw_data |>
  filter(grepl("Total d.offres", locality)) |>
  select(year, n_offers)

country_level_data <- full_join(country_level, offers_country) |>
  select(year, locality, n_offers, everything()) |>
  mutate(locality = "Grand-Duchy of Luxembourg")
```

The next steps of the **revised code** are below.

The changes are:

- The exploratory code to examine all rows with missing values in
  `average_price_nominal_euros` has been removed.  
- The tasks of extracting the country versus commune level data sets
  have been refactored into separate functions i.e. separate steps.

``` r
# ------------------------------------------------------------------------------
# Revised step 3: Output the country level data set separately 
make_country_level_data <- function(flat_data) {
  country_level <- flat_data %>%
    filter(grepl("nationale", locality)) %>%
    select(-n_offers)
  
  offers_country <- flat_data %>%
    filter(grepl("Total d.offres", locality)) %>%
    select(year, n_offers)
  
  full_join(country_level, offers_country) %>%
    select(year, locality, n_offers, everything()) %>%
    mutate(locality = "Grand-Duchy of Luxembourg")
}

country_level_data <- make_country_level_data(flat_data)
```

``` r
# ------------------------------------------------------------------------------
# Revised step 4: Output the commune level data set separately 
make_commune_level_data <- function(flat_data) {
  flat_data %>%
    filter(!grepl("nationale|offres", locality),
           !is.na(locality))
}

commune_level_data <- make_commune_level_data(flat_data)
```

The next step of the original code to extract a reference list of
communes from Wikipedia is below.

``` r
# ------------------------------------------------------------------------------
# Original step 5: Validate data set completeness i.e. whether the data set 
# captures all communes by comparing against a reference data set of communes.    
current_communes <- "https://is.gd/lux_communes" |>
  rvest::read_html(url) |>
  rvest::html_table() |>
  purrr::pluck(2) |>
  janitor::clean_names() |>
  filter(name_2 != "Name") |>
  rename(commune = name_2) |>
  mutate(commune = str_remove(commune, " .$"))

# Test if all communes from the reference data set exist in our prices data set
setdiff(unique(commune_level_data$locality), current_communes$commune)

# Extract a table of former communes from the reference data set
former_communes <- "https://is.gd/lux_former_communes" |>  
  rvest::read_html() |>
  rvest::html_table() |>
  purrr::pluck(3) |>
  janitor::clean_names() |>
  dplyr::filter(year_dissolved > 2009)

former_communes

# Create a vector containing all communes in out prices data set and all former
# communes from the reference dataset - this set should form the set of all 
# communes.  
communes <- unique(c(former_communes$name, current_communes$commune))

# Rename communes to ensure that our prices data set and reference data set use
# the same spelling for individual communes  
# Note that which() does not recognise regex symbols like . and ?  
communes[which(communes == "Clemency")] <- "Clémency"
communes[which(communes == "Redange")] <- "Redange-sur-Attert"
communes[which(communes == "Erpeldange-sur-Sûre")] <- "Erpeldange"
communes[which(communes == "Luxembourg City")] <- "Luxembourg"
communes[which(communes == "Käerjeng")] <- "Kaerjeng"

# We expect the output to be empty (it is not as Clémency is not in communes) 
setdiff(unique(commune_level_data$locality), communes)
```

The next step of the **revised code** is below.

The changes are:

- The task of extracting current and former communes from Wikipedia has
  been refactored as two different functions.  
- The task of comparing our data set’s communes to the Wikipedia
  reference list of all communes and cleaning commune names has been
  refactored into a single large function.  
- Add a comment to describe the expected output from
  `setdiff(flat_data$locality, communes)`.

``` r
# ------------------------------------------------------------------------------
# Revised step 5: Validate data set completeness i.e. whether the data set 
# captures all communes by comparing against a reference data set of communes.    

# We now need to make sure that we have all the communes/localities in the 
# cleaned data set as there were mergers in 2011, 2015 and 2018. We therefore 
# need to account for these localities.   

# We need to scrape data of all former Luxembourguish communes from Wikipedia    
get_former_communes <- function(
    url = "https://is.gd/lux_former_communes",
    min_year = 2009,
    table_position = 3 # Store hard-coded variable as default function argument
) {
  rvest::read_html(url) %>%
    rvest::html_table() %>%
    purrr::pluck(table_position) %>%
    janitor::clean_names() %>%
    filter(year_dissolved > min_year)
}

# We need to scrape data of all current communes from Wikipedia
get_current_communes <- function(
    url = "https://is.gd/lux_communes",
    table_position = 2
) {
  rvest::read_html(url) |>
    rvest::html_table() |>
    purrr::pluck(table_position) |>
    janitor::clean_names() |>
    filter(name_2 != "Name") |>
    rename(commune = name_2) |>
    mutate(commune = str_remove(commune, " .$"))
}

# We need to test if all communes from the reference data set exist in our data set
get_test_communes <- function(former_communes, current_communes){
  communes <- unique(c(former_communes$name, current_communes$commune))
  
  # We need to manually rename some communes as they have a different spelling 
  # between Wikipedia and our data set.  
  communes[which(communes == "Clemency")] <- "Clémency"
  communes[which(communes == "Redange")] <- "Redange-sur-Attert"
  communes[which(communes == "Erpeldange-sur-Sûre")] <- "Erpeldange"
  communes[which(communes == "Luxembourg City")] <- "Luxembourg"
  communes[which(communes == "Käerjeng")] <- "Kaerjeng"
  
  communes
}

former_communes <- get_former_communes()
current_communes <- get_current_communes()

communes <- get_test_communes(former_communes, current_communes)  

# Test whether all communes from our data set are represented  
# If the above code does not show any communes, then this means that we are
# accounting for every commune.   
setdiff(commune_level_data$locality, communes)
```

The last step of the original code remains unchanged.

``` r
# ------------------------------------------------------------------------------
# Step 6: Save the clean data sets 
write.csv(commune_level_data, here("data", "commune_level_data.csv"), row.names = TRUE)
write.csv(country_level_data, here("data", "country_level_data.csv"), row.names = TRUE)

rm(list = ls())  
```

## Script 2

Script 2 features data transformation steps for price index calculations
and plot generation.

The first step of the original code is below.

``` r
# ==============================================================================
# Script 2: Generate price index and plots 
# ==============================================================================

library(dplyr)
library(ggplot2)
library(purrr)
library(tidyr)

commune_level_data <- read.csv(here("data", "commune_level_data.csv"))
country_level_data <- read.csv(here("data", "country_level_data.csv"))  

# ------------------------------------------------------------------------------
# Original step 1: Calculate the Laspeyeres price index for each commune and the
# country. This index allows us to compare prices between different years and is 
# the price at year t divided by the prices in 2010.   

commune_level_data <- commune_level_data %>%
  group_by(locality) %>%
  mutate(p0 = ifelse(year == "2010", average_price_nominal_euros, NA)) %>%
  fill(p0, .direction = "down") %>%
  mutate(p0_m2 = ifelse(year == "2010", average_price_m2_nominal_euros, NA)) %>%
  fill(p0_m2, .direction = "down") %>%
  ungroup() %>%
  mutate(pl = average_price_nominal_euros/p0 * 100,
         pl_m2 = average_price_m2_nominal_euros/p0_m2 * 100)

country_level_data <- country_level_data %>%
  mutate(p0 = ifelse(year == "2010", average_price_nominal_euros, NA)) %>%
  fill(p0, .direction = "down") %>%
  mutate(p0_m2 = ifelse(year == "2010", average_price_m2_nominal_euros, NA)) %>%
  fill(p0_m2, .direction = "down") %>%
  mutate(pl = average_price_nominal_euros/p0 * 100,
         pl_m2 = average_price_m2_nominal_euros/p0_m2 * 100)
```

The first step of the **revised code** is below.

The changes are:

- Reduce code duplication by creating a single function which outputs
  commune or country level Laspeyeres price indexes. Note that this
  solution was disliked by some, as it relies on complex
  meta-programming tricks like `deparse(substitute())` and also mixes
  base R and `tidyr` meta-programming functions i.e. `quo()` and `!!`.

``` r
# ==============================================================================
# Script 2: Generate price index and plots    
# ==============================================================================

library(dplyr)
library(ggplot2)
library(purrr)
library(tidyr)

commune_level_data <- read.csv(here("data", "commune_level_data.csv"))
country_level_data <- read.csv(here("data", "country_level_data.csv"))  

# ------------------------------------------------------------------------------
# Revised step 1: Create a function that calculates the Laspeyeres price index 
# for each commune and the country. This index allows us to compare prices 
# between different years and is the price at year t divided by the 2010 price.   

get_laspeyeres_index <- function(dataset, start_year = "2010") {
  
  # Takes the input dataset and extracts its assigned name as a string
  # We expect two dataset inputs: 1) commune_level_data or country_level_data
  which_dataset <- deparse(substitute(dataset))
  
  # If the input dataset is commune_level_data, we further group by locality,
  # which is a variable in commune_level_data, not the function body or global
  # environment. 
  group_var <- if(grepl("commune", which_dataset)){
    quo(locality)
  } else {
    NULL
  }
  
  dataset %>%
    group_by(!!group_var) %>%
    mutate(p0 = ifelse(year == start_year,
                       average_price_nominal_euros,
                       NA)) %>%
    fill(p0, .direction = "down") %>%
    mutate(p0_m2 = ifelse(year == start_year,
                          average_price_m2_nominal_euros,
                          NA)) %>%
    fill(p0_m2, .direction = "down") %>%
    ungroup() %>%
    mutate(
      pl = average_price_nominal_euros/p0*100,
      pl_m2 = average_price_m2_nominal_euros/p0_m2*100)
  
}

commune_level_clean_data <- get_laspeyeres_index(commune_level_data)
country_level_clean_data <- get_laspeyeres_index(country_level_data)
```

The next step of the original code is below.

``` r
# ------------------------------------------------------------------------------
# Step 2: Create a plot for 5 communes and compare their prices to the national
# price.  

# List the 5 communes of interest  
communes <- c("Luxembourg",
              "Esch-sur-Alzette",
              "Mamer",
              "Schengen",
              "Wincrange")

# Plot for Luxembourg
filtered_data <- commune_level_data %>%
  filter(locality == communes[1])

data_to_plot <- bind_rows(
  country_level_data,
  filtered_data
)

lux_plot <- ggplot(data_to_plot) +
  geom_line(aes(y = pl_m2,
                x = year,
                group = locality,
                colour = locality))

# Plot for Esch sur Alzette
filtered_data <- commune_level_data %>%
  filter(locality == communes[2])

data_to_plot <- bind_rows(
  country_level_data,
  filtered_data
)

esch_plot <- ggplot(data_to_plot) +
  geom_line(aes(y = pl_m2,
                x = year,
                group = locality,
                colour = locality))

# Plot for Mamer  
filtered_data <- commune_level_data %>%
  filter(locality == communes[3])

data_to_plot <- bind_rows(
  country_level_data,
  filtered_data
)

mamer_plot <- ggplot(data_to_plot) +
  geom_line(aes(y = pl_m2,
                x = year,
                group = locality,
                colour = locality))

# plot for Schengen
filtered_data <- commune_level_data %>%
  filter(locality == communes[4])

data_to_plot <- bind_rows(
  country_level_data,
  filtered_data
)

schengen_plot <- ggplot(data_to_plot) +
  geom_line(aes(y = pl_m2,
                x = year,
                group = locality,
                colour = locality))

# Plot for Wincrange
filtered_data <- commune_level_data %>%
  filter(locality == communes[5])

data_to_plot <- bind_rows(
  country_level_data,
  filtered_data
)

wincrange_plot <- ggplot(data_to_plot) +
  geom_line(aes(y = pl_m2,
                x = year,
                group = locality,
                colour = locality))
```

The next step of the **revised code** is below.

The changes are:

- Reduce code duplication by creating a function that outputs the price
  for a commune compared to the national price.  
- Wrap `knitr::knit_child` inside `lapply()` to generate a subheading
  and plot outputs for all communes in the same notebook (this only
  works if script 2 is refactored into a notebook). This also negates
  the need to manually save each plot as a separate object.

``` r
# ------------------------------------------------------------------------------
# Step 2: Create a plot for 5 communes and compare their prices to the national
# price.  
communes <- c("Luxembourg",
              "Esch-sur-Alzette",
              "Mamer",
              "Schengen",
              "Wincrange")

make_plot <- function(commune){
  commune_clean_data <- commune_level_clean_data %>%
    filter(locality == commune)

  data_to_plot <- bind_rows(
    country_level_clean_data,
    commune_clean_data
  )

  ggplot(data_to_plot) +
    geom_line(aes(y = pl_m2,
                  x = year,
                  group = locality,
                  colour = locality))
}

# The author then uses knitr::knit_child() to output report sections for each 
# commune. I prefer to create a child notebook for this purpose. The plots can
# be created using purrr::map(communes, make_plot) although this code does not
# print them into a report. 
purrr::map(communes, make_plot)
```

# Other resources

- The `dplyr`
  [vignette](https://dplyr.tidyverse.org/articles/programming.html) that
  introduces the concept of tidy evaluation.
