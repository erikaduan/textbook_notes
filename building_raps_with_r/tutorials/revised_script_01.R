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

# ------------------------------------------------------------------------------
# Revised step 4: Output the commune level data set separately 
make_commune_level_data <- function(flat_data) {
  flat_data %>%
    filter(!grepl("nationale|offres", locality),
           !is.na(locality))
}

commune_level_data <- make_commune_level_data(flat_data)

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

# ------------------------------------------------------------------------------
# Step 6: Save the clean data sets 
write.csv(commune_level_data, here("data", "commune_level_data.csv"), row.names = TRUE)
write.csv(country_level_data, here("data", "country_level_data.csv"), row.names = TRUE)

# Commented out to run unit tests for tutorial part 2.12
# rm(list = ls())  