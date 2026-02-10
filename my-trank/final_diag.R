library(tidyverse)
library(janitor)

REPO_ROOT <- getwd()
HOOP_DIR <- file.path(REPO_ROOT, "data", "cleaned_csvs")

hoop_files <- list.files(HOOP_DIR, pattern = "players_2025_HML\\.csv", full.names = TRUE)
hoop_raw <- hoop_files %>%
    set_names() %>%
    map_dfr(~ read_csv(.x, show_col_types = FALSE) %>% clean_names())

names_found <- names(hoop_raw)
print("Names in hoop_raw (first 50):")
print(head(names_found, 50))

print("Is 'off_adj_rapm_value' in names?")
print("off_adj_rapm_value" %in% names_found)

print("Is 'off_team_poss_value' in names?")
print("off_team_poss_value" %in% names_found)
