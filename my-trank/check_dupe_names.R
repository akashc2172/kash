library(tidyverse)
library(janitor)

# Load helper functions (summarized)
clean_player <- function(x) {
    x %>%
        str_to_lower() %>%
        str_replace_all("[^a-z\\s]", " ") %>%
        str_squish()
}
clean_name <- function(x) {
    x %>%
        as.character() %>%
        str_to_lower() %>%
        str_replace_all("[^a-z\\s]", " ") %>%
        str_squish()
}
lastfirst_to_firstlast <- function(x) {
    x <- clean_name(x)
    ifelse(str_detect(x, " "), str_replace(x, "^(\\S+)\\s+(.*)$", "\\2 \\1"), x)
}

REPO_ROOT <- getwd()
HOOP_DIR <- file.path(REPO_ROOT, "data", "cleaned_csvs")
TORVIK_BASE_FILE <- file.path(REPO_ROOT, "data", "torvik", "torvik_base_2019_2025.csv")
TEAM_DICT_FILE <- file.path(REPO_ROOT, "data", "team_dict.rds")
team_dict <- readRDS(TEAM_DICT_FILE)

# LOAD HOOP EXPLORER DATA
hoop_files <- list.files(HOOP_DIR, pattern = "^players_\\d{4}_HML\\.csv$", full.names = TRUE)
hoop_raw <- hoop_files %>%
    set_names() %>%
    map_dfr(~ read_csv(.x, show_col_types = FALSE) %>% clean_names())

hoop2 <- hoop_raw %>%
    mutate(
        year = as.integer(year),
        team_raw = team,
        team = unname(team_dict[team_raw]) %>% coalesce(team_raw),
        player_raw = key,
        player = clean_player(player_raw)
    )

torvik2 <- read_csv(TORVIK_BASE_FILE)

hoop3 <- hoop2 %>%
    mutate(
        year = as.integer(year),
        team_key = clean_name(team),
        player_key = lastfirst_to_firstlast(player)
    )

torvik3 <- torvik2 %>%
    mutate(
        torvik_year = as.integer(year),
        year = torvik_year - 1,
        team_key = clean_name(team),
        player_key = clean_name(player)
    )

join_keys <- c("year", "team_key", "player_key")
overlap <- intersect(names(hoop3), names(torvik3))
drop_from_torvik <- setdiff(overlap, join_keys)
torvik_keep <- torvik3 %>% select(-any_of(drop_from_torvik))

important_merged <- hoop3 %>% left_join(torvik_keep, by = join_keys)

df_clean <- janitor::make_clean_names(names(important_merged))
counts <- table(df_clean)
dupes <- counts[counts > 1]
print("Duplicate Clean Names:")
print(dupes)

rename_wanted <- c(
    "off_adj_rapm_value" = "offensive rapm",
    "def_adj_rapm_value" = "defensive rapm"
)
want_old_clean <- janitor::make_clean_names(names(rename_wanted))
print("Are our wanted columns in dupes?")
print(want_old_clean %in% names(dupes))
