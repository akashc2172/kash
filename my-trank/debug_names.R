library(tidyverse)
library(janitor)

REPO_ROOT <- getwd()
HOOP_DIR <- file.path(REPO_ROOT, "data", "cleaned_csvs")
BART_2026_FILE <- file.path(REPO_ROOT, "data", "torvik_raw", "torvik_advstats_2026.csv")

hoop_files <- list.files(HOOP_DIR, pattern = "players_2025_HML\\.csv", full.names = TRUE)
hoop_raw <- hoop_files %>%
    set_names() %>%
    map_dfr(~ read_csv(.x, show_col_types = FALSE) %>% clean_names())

print("Names in hoop_raw after clean_names():")
print(names(hoop_raw)[grep("rapm", names(hoop_raw))])

rename_wanted <- c(
    "off_adj_rapm_value" = "offensive rapm",
    "def_adj_rapm_value" = "defensive rapm"
)

df_clean <- janitor::make_clean_names(names(hoop_raw))
want_old <- names(rename_wanted)
want_old_clean <- janitor::make_clean_names(want_old)

print("df_clean highlights:")
print(df_clean[grep("rapm", df_clean)])

print("want_old_clean:")
print(want_old_clean)

rename_actual <- character(0)
for (i in seq_along(want_old)) {
    hit <- which(df_clean == want_old_clean[i])
    if (length(hit) == 1) {
        rename_actual[names(hoop_raw)[hit]] <- rename_wanted[[i]]
    }
}

print("rename_actual:")
print(rename_actual)
