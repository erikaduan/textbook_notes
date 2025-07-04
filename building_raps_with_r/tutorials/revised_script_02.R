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
      pl = average_price_nominal_euros/p0 * 100,
      pl_m2 = average_price_m2_nominal_euros/p0_m2 * 100)
  
}

commune_level_clean_data <- get_laspeyeres_index(commune_level_data)
country_level_clean_data <- get_laspeyeres_index(country_level_data)

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