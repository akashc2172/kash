library(tidyverse)
library(janitor)

# Load helper functions if needed, or just simulate the state
REPO_ROOT <- getwd()
HOOP_DIR         <- file.path(REPO_ROOT, "data", "cleaned_csvs")
hoop_files <- list.files(HOOP_DIR, pattern = "players_2025_HML\\.csv", full.names = TRUE)
hoop_raw <- hoop_files %>%
  set_names() %>%
  map_dfr(~ read_csv(.x, show_col_types = FALSE) %>% clean_names())

# Simulate part of the script logic
# (Assuming join doesn't drop columns we care about)
important_merged <- hoop_raw 

df_clean <- janitor::make_clean_names(names(important_merged))
rename_wanted <- c(
  "off_adj_rapm_value" = "offensive rapm",
  "def_adj_rapm_value" = "defensive rapm"
)
want_old <- names(rename_wanted)
want_old_clean <- janitor::make_clean_names(want_old)

print("Columns matching 'rapm':")
print(names(important_merged)[grep("rapm", names(important_merged))])

rename_actual <- character(0)
for (i in seq_along(want_old)) {
  hit <- which(df_clean == want_old_clean[i])
  if (length(hit) == 1) {
    rename_actual[names(important_merged)[hit]] <- rename_wanted[[i]]
  } else {
    print(f"FAILED to find unique hit for {want_old[i]}. Hits found: {length(hit)}")
  }
}

print("rename_actual:")
print(rename_actual)

names(important_merged)[match(names(rename_actual), names(important_merged))] <- unname(rename_actual)

print("Names now matching 'rapm':")
print(names(important_merged)[grep("rapm", names(important_merged))])
