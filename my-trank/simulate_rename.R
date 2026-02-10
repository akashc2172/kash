library(tidyverse)
library(janitor)

REPO_ROOT <- getwd()
HOOP_DIR <- file.path(REPO_ROOT, "data", "cleaned_csvs")

hoop_files <- list.files(HOOP_DIR, pattern = "players_2025_HML\\.csv", full.names = TRUE)
hoop_raw <- hoop_files %>%
    set_names() %>%
    map_dfr(~ read_csv(.x, show_col_types = FALSE) %>% clean_names())

hoop3 <- hoop_raw %>%
    mutate(
        year = as.integer(year),
        team_key = janitor::make_clean_names(team),
        player_key = janitor::make_clean_names(key)
    )

# Simulate Torvik keep
# (Assuming it doesn't have any overlapping columns that cause .x/.y)
torvik_keep <- tibble(year = 2025, team_key = "duke", player_key = "boozer cameron", mpg = 30)

join_keys <- c("year", "team_key", "player_key")
important_merged <- hoop3 %>% left_join(torvik_keep, by = join_keys)

print("Columns in important_merged matching 'rapm':")
print(names(important_merged)[grep("rapm", names(important_merged))])

rename_wanted <- c(
    "off_adj_rapm_value" = "offensive rapm",
    "def_adj_rapm_value" = "defensive rapm"
)
df_clean <- janitor::make_clean_names(names(important_merged))
want_old <- names(rename_wanted)
want_old_clean <- janitor::make_clean_names(want_old)

rename_actual <- character(0)
for (i in seq_along(want_old)) {
    hit <- which(df_clean == want_old_clean[i])
    if (length(hit) == 1) {
        rename_actual[names(important_merged)[hit]] <- rename_wanted[[i]]
    }
}

print("rename_actual:")
print(rename_actual)

names(important_merged)[match(names(rename_actual), names(important_merged))] <- unname(rename_actual)

print("Names now matching 'rapm':")
print(names(important_merged)[grep("rapm", names(important_merged))])

# Test mutate
important_merged <- important_merged %>%
    mutate(
        `total RAPM` = suppressWarnings(as.numeric(`offensive rapm`)) -
            suppressWarnings(as.numeric(`defensive rapm`))
    )

print("SUCCESS: mutate worked!")
