# Building RAPs with R - Part 1.3
Erika Duan
2023-08-14

- [Project goal](#project-goal)
- [Project code](#project-code)
- [Discussion questions](#discussion-questions)

# Project goal

This textbook provides an example of a typical R data analysis work flow
and takes us through how it can be made into a progressively more
reproducible standalone project.

The project example provides is on housing in Luxembourg. Its components
involve:

- Converting human-readable Excel spreadsheets into a tidy data frame.  
- Confirming that house prices are correctly extracted for all communes,
  cantons and at a national level.  
- Converting nominal (yearly reported prices) to real prices to allow
  for direct comparisons.  
- Generating some tables and plots based on the cleaned data set.

# Project code

The following project code is provided to us in two parts: 1) the script
to generate the cleaned and transformed data set and 2) the script to
analyse and plot the dataset.

**Note:** I have modified the original location used to read and write
files and to save plots to as `./data/...` and `./output/...`
respectively and added more code comments.

Observations about script 1:

- I found it useful to break up the single script into logical chunks.
  I.e. step 1: extract the data and convert from Excel to csv.  
- I found it useful to clearly document functions and to flag them as
  separate code chunks i.e. for the read_clean() function.  
- The code contains a mixture of code that performs an action versus
  code that reviews a data output. Is there a neater method of handling
  this mixture of code types?  
- Data analytical code relies on hard-coded assumptions, for example to
  skip the first 10 rows when reading each Excel spreadsheet and to
  standardise the spelling of locations (this was the most finnicky to
  decode/debug). Is there a more optimal method of handling hard-coded
  assumptions?  
- Data analytical code relies on assumptions that the format of source
  raw data sets from the internet remains consistent.  
- The `raw_data` object is overwritten but all other intermediate data
  outputs are given unique names and not overwritten.  
- The script uses the format `package::function()` for functions from
  packages that are not commonly used to improve traceability.

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
# Step 1: Download raw Excel file from GitHub and transform multiple sheets into
# a single data frame with a new column for year.    

url <- "https://github.com/b-rodrigues/rap4all/raw/master/datasets/vente-maison-2010-2021.xlsx"

raw_data <- tempfile(fileext = ".xlsx")

download.file(url, raw_data, method = "auto", mode = "wb")

sheets <- excel_sheets(raw_data)

# Function to create a single data frame from a named sheet and then add a 
# column to denote year. This can be done because individual sheets are named 
# after individual years.      

read_clean <- function(..., sheet) {
  read_excel(..., sheet = sheet) |>
    mutate(year = sheet)
}

# Concatenate all individual spreadsheet tables into a single table
raw_data <- map(
  sheets,
  ~read_clean(raw_data,
              skip = 10,
              sheet = .)
                   ) |>
  bind_rows() |>
  clean_names()
# ------------------------------------------------------------------------------
# Step 2: Clean and select required data frame column names and check the 
# cleaned output

raw_data <- raw_data |>
  rename(
    locality = commune,
    n_offers = nombre_doffres,
    average_price_nominal_euros = prix_moyen_annonce_en_courant,
    average_price_m2_nominal_euros = prix_moyen_annonce_au_m2_en_courant,
    average_price_m2_nominal_euros = prix_moyen_annonce_au_m2_en_courant
  ) |>
  mutate(locality = str_trim(locality)) |>
  select(year, locality, n_offers, starts_with("average"))


str(raw_data)
# ------------------------------------------------------------------------------
# Step 3: Check string values in the locality column and normalise the spelling

# As well as multiple spelling version, we also have NAs and long string values
# in the locality column 

raw_data |>
  dplyr::filter(grepl("Luxembourg", locality)) |>
  dplyr::count(locality)

# Both Pétange and Petange exist and are coded as "P?tange" instead   
raw_data |>
  dplyr::filter(grepl("P.tange", locality)) |>
  dplyr::count(locality)

raw_data <- raw_data |>
  mutate(locality = ifelse(grepl("Luxembourg-Ville", locality),
                           "Luxembourg",
                           locality),
         locality = ifelse(grepl("P.tange", locality),
                           "P?tange",
                           locality)
         ) |>
  mutate(across(starts_with("average"), as.numeric))
# ------------------------------------------------------------------------------
# Step 4: Treat NA values by separating commune and country level data for price

# Examine all rows with missing values for average_price_nominal_euros  
raw_data |>
  filter(is.na(average_price_nominal_euros))

# Remove rows citing the source
raw_data <- raw_data |>
  filter(!grepl("Source", locality))

# Keep commune level data
commune_level_data <- raw_data |>
    filter(!grepl("nationale|offres", locality),
           !is.na(locality))

# Keep country level data which is recorded as 'Moyenne nationale' under locality
country_level <- raw_data |>
  filter(grepl("nationale", locality)) |>
  select(-n_offers)

# n_offers at the country level is recorded as 'Total d.offres' under locality 
offers_country <- raw_data |>
  filter(grepl("Total d.offres", locality)) |>
  select(year, n_offers)

country_level_data <- full_join(country_level, offers_country) |>
  select(year, locality, n_offers, everything()) |>
  mutate(locality = "Grand-Duchy of Luxembourg")

# ------------------------------------------------------------------------------
# Step 5: Validate data set completeness i.e. whether the data set captures all
# communes by comparing against a reference data set of communes  

current_communes <- "https://en.wikipedia.org/wiki/List_of_communes_of_Luxembourg" |>
  rvest::read_html() |>
  rvest::html_table() |>
  purrr::pluck(1) |>
  janitor::clean_names()

# Test if all communes from the reference data set exist in our prices data set
setdiff(unique(commune_level_data$locality), current_communes$commune)

# Extract a table of former communes from the reference data set
former_communes <- "https://en.wikipedia.org/wiki/Communes_of_Luxembourg#Former_communes" |>  
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
communes[which(communes == "Redange")] <- "Redange-sur-Attert"
communes[which(communes == "Erpeldange-sur-Sûre")] <- "Erpeldange"
communes[which(communes == "Luxembourg-City")] <- "Luxembourg"
communes[which(communes == "Käerjeng")] <- "Kaerjeng"
communes[which(communes == "Pétange")] <- "P?tange"

# We expect the output to be empty (it is not as Clémency is not in communes) 
setdiff(unique(commune_level_data$locality), communes)

# ------------------------------------------------------------------------------
# Step 6: Save the clean data sets 
write.csv(commune_level_data, here("data", "commune_level_data.csv"), row.names = TRUE)
write.csv(country_level_data, here("data", "country_level_data.csv"), row.names = TRUE)

rm(list = ls())  
```

Observations about script 2:

- Code repetition exists for the generation of commune and country level
  price indexes and for graph plotting. This code can be refactored.
- Intermediate data outputs like `filtered_data` and `data_to_plot` are
  repeatedly overwritten.

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
# Step 1: Calculate the Laspeyeres price index for each commune and the country 
# This index allows us to compare prices between different years and is the 
# price at year t divided by the prices in 2010.   

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

# ------------------------------------------------------------------------------
# Step 3: Save the graphs in the output folder
ggsave(here("output", "lux_plot.pdf"), lux_plot)
ggsave(here("output", "esch_plot.pdf"), esch_plot)
ggsave(here("output", "mamer_plot.pdf"), mamer_plot)
ggsave(here("output", "schengen_plot.pdf"), schengen_plot)
ggsave(here("output", "wincrange_plot.pdf"), wincrange_plot)
```

# Discussion questions

This code example is typical for data analytical projects as it involves
obtaining raw data from external sources (which are assumed to be
stable), some component of hardcoding to clean values and an iterative
reporting component.

Other reproducibility considerations include:

- Clear documentation of package dependencies, package versions and how
  to install them would be useful.  
- Addition of unit tests to ensure that future updates to the raw data
  sets are seamlessly included (i.e. check whether historical data has
  been revised and needs integration, check whether new localities
  appear in the future dataset and their corresponding index is
  appropriately generated).  
- For true reproducibility, creating a snapshot of the original
  computational environment and making this available to other users is
  important.
