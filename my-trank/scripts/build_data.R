## ----cars-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
library(tidyverse)
library(janitor)
library(stringr)
library(dplyr)

library(readr)


## ----ab , echo=FALSE------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1. PATH CONFIGURATION
# If running in RStudio/RMarkdown, set your root directory manually here
# to match your repo location.
# Example: setwd("/Users/akashc/my-trank")
# Or rely on the script logic below:

get_repo_root <- function() {
  # If running interactively, assume we are at the root or need to set it
  return(getwd())
}

REPO_ROOT <- get_repo_root()

# Define Paths based on your structure
HOOP_DIR <- file.path(REPO_ROOT, "data", "cleaned_csvs")
TORVIK_BASE_FILE <- file.path(REPO_ROOT, "data", "torvik", "torvik_base_2019_2025.csv")
BART_2026_CANDIDATES <- c(
  file.path(REPO_ROOT, "data", "torvik_raw", "torvik_advstats_2026.csv"),
  # Legacy/alt path kept for backwards compatibility
  file.path(REPO_ROOT, "data", "bart", "2026trank_data.csv")
)

pick_latest_file <- function(paths) {
  existing <- paths[file.exists(paths)]
  if (length(existing) == 0) return(NA_character_)
  info <- file.info(existing)
  existing[which.max(info$mtime)]
}

BART_2026_FILE <- pick_latest_file(BART_2026_CANDIDATES)
TEAM_DICT_FILE <- file.path(REPO_ROOT, "data", "team_dict.rds")
team_dict <- readRDS(TEAM_DICT_FILE)

OUT_SEASON <- file.path(REPO_ROOT, "public", "data", "season.csv")
OUT_CAREER <- file.path(REPO_ROOT, "public", "data", "career.csv")

# Create output dir if missing
dir.create(dirname(OUT_SEASON), recursive = TRUE, showWarnings = FALSE)

clean_player <- function(x) {
  x %>%
    str_to_lower() %>%
    str_replace_all("[^a-z\\s]", " ") %>%
    str_squish()
}

cat("[config] REPO_ROOT:      ", REPO_ROOT, "\n")
cat("[config] HOOP_DIR:       ", HOOP_DIR, "\n")
cat("[config] TORVIK_BASE:    ", TORVIK_BASE_FILE, "\n")
cat("[config] TORVIK_2026:    ", ifelse(is.na(BART_2026_FILE), "(missing)", BART_2026_FILE), "\n")

# 2. HELPER FUNCTIONS
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

feet_in_to_inches <- function(x) {
  x <- as.character(x)
  ft <- suppressWarnings(as.integer(str_extract(x, "^\\d+")))
  inch <- suppressWarnings(as.integer(str_extract(x, "(?<=-)\\d+$")))
  ifelse(is.na(ft) | is.na(inch), NA_integer_, ft * 12 + inch)
}

pick_first <- function(df, candidates, default = NA) {
  hit <- candidates[candidates %in% names(df)][1]
  if (is.na(hit)) {
    return(rep(default, nrow(df)))
  }
  df[[hit]]
}

wavg <- function(x, w) {
  x <- suppressWarnings(as.numeric(x))
  w <- suppressWarnings(as.numeric(w))
  ok <- !is.na(x) & !is.na(w) & w > 0
  if (!any(ok)) {
    return(NA_real_)
  }
  sum(x[ok] * w[ok], na.rm = TRUE) / sum(w[ok], na.rm = TRUE)
}

# 3. LOAD HOOP EXPLORER DATA
hoop_files <- list.files(HOOP_DIR, pattern = "^players_\\d{4}_HML\\.csv$", full.names = TRUE)
if (length(hoop_files) == 0) stop(paste("No Hoop files found in", HOOP_DIR))

cat("[load] Found", length(hoop_files), "Hoop Explorer files.\n")

hoop_raw <- hoop_files %>%
  set_names() %>%
  map_dfr(~ read_csv(.x, show_col_types = FALSE) %>% clean_names())

hoop2 <- hoop_raw %>%
  mutate(
    year = as.integer(year),
    team_raw = team,
    team = unname(team_dict[team_raw]) %>% coalesce(team_raw),
    player_raw = key, # Hoop Explorer usually uses key as player name; change if yours differs
    player = clean_player(player_raw)
  )
cat("[success] HOOP2 created with", nrow(hoop2), "rows.\n")


## ----pressure, echo=FALSE-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
torvik2 <- read_csv(TORVIK_BASE_FILE)
# ---------------------------------------------------------
# FUNCTION: Convert Raw Bart CSV -> Torvik2 Format
# ---------------------------------------------------------
process_bart_2026 <- function(filepath) {
  # 1. Define Headers Manually
  bart_headers <- c(
    "player_name", "team", "conf", "GP", "Min_per", "ORtg", "usg", "eFG", "TS_per",
    "ORB_per", "DRB_per", "AST_per", "TO_per", "FTM", "FTA", "FT_per", "twoPM", "twoPA", "twoP_per",
    "TPM", "TPA", "TP_per", "blk_per", "stl_per", "ftr", "yr", "ht", "num",
    "porpag", "adjoe", "pfr", "year", "pid", "hometown", "Rec Rank", "ast/tov",
    "rimmade", "rimmade+rimmiss", "midmade", "midmade+midmiss", "rimmade/(rimmade+rimmiss)",
    "midmade/(midmade+midmiss)", "dunksmade", "dunksmiss+dunksmade", "dunksmade/(dunksmade+dunksmiss)",
    "pick", "drtg", "adrtg", "dporpag", "stops", "bpm", "obpm", "dbpm", "gbpm", "mp", "ogbpm", "dgbpm",
    "oreb", "dreb", "treb", "ast", "stl", "blk", "pts", "role", "3p/100?", "birthdate"
  )

  # 2. Read file
  raw <- read_csv(filepath, col_names = bart_headers, show_col_types = FALSE)

  # 3. STRICT Mapping
  df_final <- raw %>%
    transmute(
      # --- Identifiers ---
      player = player_name,
      pos = role,
      exp = yr,

      # *** FIX: Force num to numeric ***
      num = suppressWarnings(as.numeric(num)),
      hgt = ht,
      team = team,
      conf = conf,

      # --- Minutes & Games ---
      g = GP,
      min = Min_per,
      mpg = mp,


      # --- Basic Stats ---
      ppg = pts,
      oreb = oreb,
      dreb = dreb,
      rpg = treb,
      apg = ast,

      # Derived TOV
      tov = ifelse(`ast/tov` > 0, ast / `ast/tov`, 0),
      ast_to = `ast/tov`,
      spg = stl,
      bpg = blk,

      # --- Advanced Rates & Metrics ---
      usg = usg,
      ortg = ORtg,
      efg = eFG,
      ts = TS_per,
      year = year,
      torvik_id = pid,

      # --- Shooting Totals ---
      fgm = twoPM + TPM,
      ftm = FTM,
      fta = FTA,
      ft_pct = FT_per,
      two_m = twoPM,
      two_a = twoPA,
      two_pct = twoP_per,
      three_m = TPM,
      three_a = TPA,
      three_pct = TP_per,
      dunk_m = dunksmade,
      dunk_a = `dunksmiss+dunksmade`,
      dunk_pct = `dunksmade/(dunksmade+dunksmiss)`,
      rim_m = rimmade,
      rim_a = `rimmade+rimmiss`,
      rim_pct = `rimmade/(rimmade+rimmiss)`,
      mid_m = midmade,
      mid_a = `midmade+midmiss`,
      mid_pct = `midmade/(midmade+midmiss)`,

      # --- Advanced Metrics II ---
      porpag = porpag,
      dporpag = dporpag,
      adj_oe = adjoe,
      drtg = drtg,
      adj_de = adrtg,
      stops = stops,
      obpm = obpm,
      dbpm = dbpm,
      bpm = bpm,
      gbpm = gbpm,

      # --- Advanced Rates II ---
      oreb_rate = ORB_per,
      dreb_rate = DRB_per,
      ast = AST_per,
      to = TO_per,
      blk = blk_per,
      stl = stl_per,
      ftr = ftr,
      pfr = pfr,
      rec = `Rec Rank`,
      pick = pick,

      # --- Calculated Fields ---
      fga = twoPA + TPA,
      fg_pct = ifelse((twoPA + TPA) > 0, (twoPM + TPM) / (twoPA + TPA), 0),

      # --- Raw Dupes ---
      team_raw = team,
      player_raw = player_name
    )

  return(df_final)
}

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------

# 1. Define File Path
custom_path <- BART_2026_FILE

# 2. Process and Merge
if (!is.na(custom_path) && file.exists(custom_path)) {
  message(paste("Processing custom file:", custom_path))

  # Process the new file
  torvik_2026 <- process_bart_2026(custom_path)

  # =========================================================
  # NEW: MANUAL TEAM NAME FIXES (2026)
  # =========================================================
  # Define mappings: "Bad Name in 2026 CSV" = "Standard Name in torvik2"
  team_fix_2026 <- c(
    "Charleston" = "College of Charleston",
    "Purdue Fort Wayne" = "Fort Wayne",
    "LIU" = "LIU Brooklyn",
    "Detroit Mercy" = "Detroit",
    "Louisiana" = "Louisiana Lafayette",
    "Saint Francis" = "St. Francis PA"
  )

  # Apply fixes to the 2026 data BEFORE merging
  torvik_2026 <- torvik_2026 %>%
    mutate(team = dplyr::recode(team, !!!team_fix_2026))

  message("Applied 2026 team name fixes.")
  # =========================================================

  # Merge with existing 'torvik2' (from previous chunks) if it exists
  if (exists("torvik2")) {
    torvik2 <- bind_rows(torvik2, torvik_2026)
    message("Success: Merged 2026 data with historical data.")
  } else {
    torvik2 <- torvik_2026
    message("Success: Loaded 2026 data (No historical data found).")
  }
}

## ----save-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# -----------------------------
# Helpers
# -----------------------------
clean_name <- function(x) {
  x %>%
    as.character() %>%
    str_to_lower() %>%
    str_replace_all("[^a-z\\s]", " ") %>%
    str_squish()
}

lastfirst_to_firstlast <- function(x) {
  x <- clean_name(x)
  ifelse(
    str_detect(x, " "),
    str_replace(x, "^(\\S+)\\s+(.*)$", "\\2 \\1"),
    x
  )
}

feet_in_to_inches <- function(x) {
  x <- as.character(x)
  ft <- suppressWarnings(as.integer(str_extract(x, "^\\d+")))
  inch <- suppressWarnings(as.integer(str_extract(x, "(?<=-)\\d+$")))
  ifelse(is.na(ft) | is.na(inch), NA_integer_, ft * 12 + inch)
}


# -----------------------------
# Team-name mapping (Hoop -> Bart)
# -----------------------------
# Hoop team_key -> Bart team_key
team_fix <- c(
  "South Fla." = "South Florida",
  "A&M-Corpus Christi" = "Texas A&M Corpus Chris",
  "Alcorn" = "Alcorn St.",
  "Ark.-Pine Bluff" = "Arkansas Pine Bluff",
  "Boston U." = "Boston University",
  "Dixie St." = "Utah Tech",
  "Central Ark." = "Central Arkansas",
  "Central Conn. St." = "Central Connecticut",
  "Central Mich." = "Central Michigan",
  "Col. of Charleston" = "College of Charleston",
  "CSU Bakersfield" = "Cal St. Bakersfield",
  "Charleston So." = "Charleston Southern",
  "Eastern Ill." = "Eastern Illinois",
  "Eastern Ky." = "Eastern Kentucky",
  "Eastern Mich." = "Eastern Michigan",
  "Eastern Wash." = "Eastern Washington",
  "Fla. Atlantic" = "Florida Atlantic",
  "Ga. Southern" = "Georgia Southern",
  "Houston Baptist" = "Houston Christian",
  "Lamar University" = "Lamar",
  "LMU (CA)" = "Loyola Marymount",
  "Middle Tenn." = "Middle Tennessee",
  "Mississippi Val." = "Mississippi Valley St.",
  "N.C. A&T" = "North Carolina A&T",
  "N.C. Central" = "North Carolina Central",
  "Northern Ariz." = "Northern Arizona",
  "Northern Colo." = "Northern Colorado",
  "North Ala." = "North Alabama",
  "North Carolina St." = "N.C. State",
  "Northern Ill." = "Northern Illinois",
  "Southern Ind." = "Southern Indiana",
  "Northern Ky." = "Northern Kentucky",
  "South Fla." = "South Florida",
  "Southeast Mo. St." = "Southeast Missouri St.",
  "Southeastern La." = "Southeastern Louisiana",
  "Southern Ill." = "Southern Illinois",
  "Southern U." = "Southern",
  "St. Thomas (MN)" = "St. Thomas",
  "Texas A&M Corpus Chr" = "Texas A&M Corpus Chris",
  "Tex. A&M-Commerce" = "Texas A&M Commerce",
  "East Texas A&M" = "Texas A&M Commerce",
  "Western Caro." = "Western Carolina",
  "Western Ill." = "Western Illinois",
  "Western Ky." = "Western Kentucky",
  "Western Mich." = "Western Michigan",
  "Ark.-Pine Bluff" = "Arkansas Pine Bluff",
  "West Ga." = "West Georgia"
)

# expects columns literally named: "hoop name" and "bart name"


map_team_key <- function(x) {
  k <- clean_name(x) # -> cleaned team key
  dplyr::coalesce(unname(team_map[k]), k)
}

# -----------------------------
# 0) Standardize keys + FIX YEAR OFFSET
# -----------------------------
# Hoop uses season-start year already (e.g., 2018 = 2018–19)

hoop3 <- hoop2 %>%
  mutate(
    year = as.integer(year),
    team = dplyr::recode(team, !!!team_fix, .default = team),
    team_key = clean_name(team),
    player_key = lastfirst_to_firstlast(player),
    roster_number = suppressWarnings(as.integer(roster_number)),
    hoop_hgt_in = feet_in_to_inches(roster_height)
  )

torvik3 <- torvik2 %>%
  mutate(
    torvik_year = as.integer(year),
    year = torvik_year - 1,
    team = dplyr::recode(team, !!!team_fix, .default = team),
    team_key = clean_name(team),
    player_key = clean_name(player),
    num = suppressWarnings(as.integer(num)),
    torvik_hgt_in = feet_in_to_inches(hgt)
  )

# -----------------------------
# 1) Remove overlapping Torvik names BEFORE join (except keys)
# -----------------------------
join_keys <- c("year", "team_key", "player_key")

overlap <- intersect(names(hoop3), names(torvik3))
drop_from_torvik <- setdiff(overlap, join_keys)

torvik_keep <- torvik3 %>% select(-any_of(drop_from_torvik))

# -----------------------------
# 2) STRICT JOIN
# -----------------------------
merged1 <- hoop3 %>% left_join(torvik_keep, by = join_keys)

cat("Matched rows after strict join (non-NA mpg): ",
  sum(!is.na(merged1$mpg)), " / ", nrow(merged1), "\n",
  sep = ""
)

# -----------------------------
# 3) Cross-validation flags
# -----------------------------
merged1 <- merged1 %>%
  mutate(
    num_match = !is.na(roster_number) & !is.na(num) & roster_number == num,
    hgt_match = !is.na(hoop_hgt_in) & !is.na(torvik_hgt_in) &
      abs(hoop_hgt_in - torvik_hgt_in) <= 1
  )

# -----------------------------
# 4) NA RESCUE (year/team candidates + REQUIRE jersey+height)
# -----------------------------
need_fix <- merged1 %>% filter(is.na(mpg))

if (nrow(need_fix) > 0) {
  x <- need_fix %>%
    transmute(
      id,
      year, team_key,
      hoop_name = player_key,
      hoop_num = roster_number,
      hoop_hgt = hoop_hgt_in
    )

  y <- torvik_keep %>%
    transmute(
      year, team_key,
      torvik_name = player_key,
      torvik_num = num,
      torvik_hgt = torvik_hgt_in,
      across(everything())
    )

  candidates <- x %>%
    left_join(
      y,
      by = c("year", "team_key"),
      relationship = "many-to-many"
    ) %>%
    mutate(
      num_ok  = !is.na(hoop_num) & !is.na(torvik_num) & hoop_num == torvik_num,
      hgt_ok  = !is.na(hoop_hgt) & !is.na(torvik_hgt) & abs(hoop_hgt - torvik_hgt) <= 1,
      name_ok = hoop_name == torvik_name,
      score   = 4 * as.integer(num_ok) + 2 * as.integer(hgt_ok) + 1 * as.integer(name_ok)
    ) %>%
    filter(num_ok, hgt_ok) %>% # require both (safe)
    group_by(id) %>%
    slice_max(score, n = 1, with_ties = FALSE) %>%
    ungroup()

  torvik_cols <- setdiff(names(torvik_keep), join_keys)

  merged2 <- merged1 %>%
    left_join(
      candidates %>% select(id, all_of(torvik_cols)),
      by = "id",
      suffix = c("", "_rescued")
    )

  for (c in torvik_cols) {
    rc <- paste0(c, "_rescued")
    if (rc %in% names(merged2)) {
      merged2[[c]] <- ifelse(is.na(merged2[[c]]), merged2[[rc]], merged2[[c]])
    }
  }

  merged2 <- merged2 %>% select(-ends_with("_rescued"))
} else {
  merged2 <- merged1
}

cat("Matched rows after rescue (non-NA mpg): ",
  sum(!is.na(merged2$mpg)), " / ", nrow(merged2), "\n",
  sep = ""
)

# -----------------------------
# 5) Audit summary
# -----------------------------
audit <- merged2 %>%
  summarise(
    n = n(),
    matched = sum(!is.na(mpg)),
    pct_matched = mean(!is.na(mpg)),
    pct_num_match_among_matched = mean(num_match[!is.na(mpg)], na.rm = TRUE),
    pct_hgt_match_among_matched = mean(hgt_match[!is.na(mpg)], na.rm = TRUE)
  )

print(audit)

merged <- merged2

out_path <- "torvik3.csv"
write_csv(torvik3, out_path)

# 3) open it in your default app (Excel usually)


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------
# 5) LAST-NAME UNIQUE RESCUE (team + year + last name, must be exactly 1)
#     Robust to name order by trying BOTH ends of Hoop name
# -----------------------------


clean_letters <- function(x) {
  x %>%
    as.character() %>%
    str_to_lower() %>%
    str_replace_all("[^a-z\\s]", " ") %>%
    str_squish()
}

is_roman <- function(tok) {
  roman_re <- "^(i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx)$"
  str_detect(tok, roman_re)
}

drop_suffix_anywhere <- function(tokens) {
  tokens <- tokens[!(tokens %in% c("jr", "sr"))]
  tokens <- tokens[!vapply(tokens, is_roman, logical(1))]
  tokens
}

re_escape <- function(x) {
  str_replace_all(x, "([\\^\\$\\.|\\?\\*\\+\\(\\)\\[\\]\\{\\}\\\\])", "\\\\\\1")
}

# Returns up to TWO candidate last-name tokens from Hoop name:
# - after cleaning letters
# - after removing jr/sr/roman numerals ANYWHERE
# - candidates are: first token AND last token (order-agnostic)
extract_last_candidates <- function(name_vec) {
  raw_vec <- as.character(name_vec)

  out <- lapply(raw_vec, function(raw) {
    if (is.na(raw)) {
      return(character(0))
    }
    raw <- str_trim(raw)
    if (raw == "") {
      return(character(0))
    }

    x <- clean_letters(raw)
    toks <- str_split(x, "\\s+")[[1]]
    toks <- drop_suffix_anywhere(toks)
    if (length(toks) == 0) {
      return(character(0))
    }

    unique(c(toks[1], toks[length(toks)]))
  })

  out
}

# --- rows that still need Torvik ---
need_lastname_rescue <- merged2 %>%
  mutate(.needs_torvik = if ("torvik_year" %in% names(.)) is.na(torvik_year) else is.na(mpg)) %>%
  filter(.needs_torvik) %>%
  select(-.needs_torvik)

if (nrow(need_lastname_rescue) == 0) {
  message("No rows need last-name rescue (all already matched).")
  merged3 <- merged2
} else {
  # Use Hoop's *current* name key (player_key) since that's what you're working with
  hoop_search <- need_lastname_rescue %>%
    transmute(
      id,
      year,
      team_key,
      hoop_name_for_search = player_key
    )

  cand_list <- extract_last_candidates(hoop_search$hoop_name_for_search)

  hoop_search_long <- hoop_search %>%
    mutate(search_last = cand_list) %>%
    tidyr::unnest(search_last) %>%
    filter(!is.na(search_last), search_last != "")

  torvik_lookup <- torvik_keep %>%
    transmute(
      year,
      team_key,
      torvik_player_key = player_key,
      across(everything())
    )

  cand <- hoop_search_long %>%
    left_join(
      torvik_lookup,
      by = c("year", "team_key"),
      relationship = "many-to-many"
    ) %>%
    mutate(
      last_pat = paste0("\\b", re_escape(search_last), "\\b"),
      last_match = str_detect(torvik_player_key, last_pat)
    ) %>%
    filter(last_match)

  # accept ONLY if exactly 1 torvik_player_key matched across either candidate token
  unique_hits <- cand %>%
    group_by(id) %>%
    summarise(n_candidates = n_distinct(torvik_player_key), .groups = "drop") %>%
    filter(n_candidates == 1) %>%
    select(id)

  picked <- cand %>%
    semi_join(unique_hits, by = "id") %>%
    group_by(id) %>%
    slice(1) %>% # unique anyway
    ungroup()

  join_keys <- c("year", "team_key", "player_key")
  torvik_cols <- setdiff(names(torvik_keep), join_keys)

  merged3 <- merged2 %>%
    left_join(
      picked %>% select(id, all_of(torvik_cols), torvik_player_key),
      by = "id",
      suffix = c("", "_lname")
    )

  for (c in torvik_cols) {
    cname <- paste0(c, "_lname")
    if (cname %in% names(merged3)) {
      merged3[[c]] <- ifelse(is.na(merged3[[c]]), merged3[[cname]], merged3[[c]])
    }
  }

  merged3 <- merged3 %>%
    mutate(
      player_key_std = ifelse(is.na(player_key) | player_key == "", torvik_player_key, player_key)
    ) %>%
    select(-ends_with("_lname"))

  message("Last-name rescue merged rows: ", n_distinct(picked$id))
}

merged <- merged3


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------
# POST STEP) TOKEN RESCUE (after everything else)
# Try LAST token first, then FIRST token, within (year + team_key).
# Must be EXACTLY 1 Torvik candidate to merge.
# -----------------------------

library(tidyr)

# --- helpers ---
is_roman <- function(tok) {
  str_detect(tok, "^(i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx)$")
}

drop_jr_sr_roman_anywhere <- function(toks) {
  toks <- toks[!(toks %in% c("jr", "sr"))]
  toks <- toks[!vapply(toks, is_roman, logical(1))]
  toks
}

re_escape <- function(x) {
  str_replace_all(x, "([\\^\\$\\.|\\?\\*\\+\\(\\)\\[\\]\\{\\}\\\\])", "\\\\\\1")
}

# from a cleaned name like "jr terrell brown" -> c(last="brown", first="terrell")
# returns 0, 1, or 2 tokens
make_two_tokens <- function(name_vec) {
  x <- as.character(name_vec)
  x[is.na(x)] <- ""

  lapply(x, function(s) {
    s <- str_trim(s)
    if (s == "") {
      return(character(0))
    }

    toks <- str_split(s, "\\s+")[[1]]
    toks <- toks[toks != ""]
    toks <- drop_jr_sr_roman_anywhere(toks)

    if (length(toks) == 0) {
      return(character(0))
    }

    last_tok <- toks[length(toks)]
    first_tok <- toks[1]

    # order matters: try last token first, then first token
    unique(c(last_tok, first_tok))
  })
}

# choose rows that still need torvik (use mpg NA as the indicator)
need_tok <- merged2 %>% filter(is.na(mpg))

if (nrow(need_tok) == 0) {
  message("POST token rescue: nothing to do (all already matched).")
  merged3 <- merged2
} else {
  hoop_tokens <- need_tok %>%
    transmute(
      id,
      year,
      team_key,
      name_for_tokens = player_key
    ) %>%
    mutate(tok_list = make_two_tokens(name_for_tokens)) %>%
    unnest(tok_list) %>%
    group_by(id) %>%
    mutate(priority = row_number()) %>% # 1 = last token, 2 = first token
    ungroup() %>%
    rename(token = tok_list) %>%
    filter(!is.na(token), token != "")

  torvik_lookup <- torvik_keep %>%
    mutate(.torvik_row = row_number()) %>%
    transmute(
      year,
      team_key,
      torvik_player_key = player_key,
      .torvik_row,
      across(everything())
    )

  # join by year+team_key then filter by token match in torvik_player_key
  cand <- hoop_tokens %>%
    left_join(torvik_lookup, by = c("year", "team_key"), relationship = "many-to-many") %>%
    mutate(
      pat = paste0("\\b", re_escape(token), "\\b"),
      tok_match = str_detect(torvik_player_key, pat)
    ) %>%
    filter(tok_match)

  # helper to pick unique matches at a given priority
  pick_unique_at_priority <- function(cand_df, pri) {
    cand_df %>%
      filter(priority == pri) %>%
      group_by(id) %>%
      filter(n_distinct(.torvik_row) == 1) %>%
      slice(1) %>%
      ungroup()
  }

  picked1 <- pick_unique_at_priority(cand, 1) # try LAST token first
  remaining_ids <- setdiff(need_tok$id, picked1$id)

  picked2 <- cand %>%
    filter(id %in% remaining_ids) %>%
    pick_unique_at_priority(., 2) # then try FIRST token

  picked <- bind_rows(picked1, picked2)

  join_keys <- c("year", "team_key", "player_key")
  torvik_cols <- setdiff(names(torvik_keep), join_keys)

  merged3 <- merged2 %>%
    left_join(
      picked %>% select(id, all_of(torvik_cols)),
      by = "id",
      suffix = c("", "_tok")
    )

  # fill ONLY where torvik values are still NA
  for (c in torvik_cols) {
    cc <- paste0(c, "_tok")
    if (cc %in% names(merged3)) {
      merged3[[c]] <- ifelse(is.na(merged3[[c]]), merged3[[cc]], merged3[[c]])
    }
  }

  merged3 <- merged3 %>% select(-ends_with("_tok"))

  message("POST token rescue merged rows: ", n_distinct(picked$id))
}

# keep your final name consistent
merged <- merged3


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


important_merged <- merged

# -----------------------------
# Helpers: Excel column letters -> numeric index
# -----------------------------
# -----------------------------
# 1) Drop columns (Explicitly listed with names)
# -----------------------------
drop_cols <- c(
  # --- Singles ---
  "id", "year", "style", "pos_freqs", "doc_count", "pos_class", "code", "pos_confidences",
  "team_raw", "gender", "player", "team_key", "player_key", "hgt",
  "torvik_hgt_in", "num_match", "hgt_match",
  "roster_origin", "roster_pos", "roster_year_class", "roster_height",
  "roster_number", "def_adj_prod_override", "def_rtg_override",
  "off_2p_ast_value", "off_ft_value", "off_ftr_value",
  "off_assist_value", "off_to_value", "off_orb_value", "off_usage_value",
  "def_orb_value", "def_ftr_value", "def_to_value", "def_2prim_value",
  "def_team_poss_pct_value", "off_rtg_value",
  "def_adj_rapm_prod_old_value", "def_adj_rapm_prod_value", "off_adj_rapm_prod_value",
  "off_adj_rapm_old_value", "def_adj_rapm_override", "def_adj_rapm_old_value",
  "def_adj_prod_old_value", "def_adj_prod_value", "def_adj_rtg_override",
  "def_adj_rtg_old_value", "def_rtg_old_value", "def_rtg_value", "def_adj_rtg_value",
  "off_adj_prod_value", "off_adj_rtg_value", "off_team_poss_pct_value",
  "off_efg_value", "off_trans_efg_value", "off_scramble_efg_value",
  "def_adj_rapm_prod_override", "player_raw",
  "off_trans_2prim_ast_value", "off_trans_2p_ast_value", "off_trans_3p_ast_value",
  "off_trans_2prim_value", "off_trans_2pmid_ast_value", "off_trans_ft_value",
  "off_trans_ftr_value", "off_trans_2primr_value", "off_trans_2pmidr_value",
  "off_trans_3pr_value", "off_trans_assist_value",
  "tier",
  "three_pct", "two_pct",
  "adj_rtg_margin_rank", "def_adj_rtg_rank", "off_adj_rtg_rank",
  "adj_prod_margin_rank", "def_adj_prod_rank", "off_adj_prod_rank",
  "adj_rapm_margin_rank", "def_adj_rapm_rank", "off_adj_rapm_rank",
  "adj_rapm_prod_margin_rank", "def_adj_rapm_prod_rank", "off_adj_rapm_prod_rank",
  "off_adj_opp_value", "off_poss_value", "def_adj_opp_value",
  "off_scramble_2p_value", "off_scramble_2p_ast_value",
  "off_scramble_3p_value", "off_scramble_3p_ast_value",
  "off_scramble_2prim_value", "off_scramble_2prim_ast_value",
  "off_scramble_2pmid_value", "off_scramble_2pmid_ast_value",
  "off_scramble_ft_value", "off_scramble_ftr_value",
  "off_scramble_2primr_value", "off_scramble_2pmidr_value",
  "off_scramble_3pr_value", "off_scramble_assist_value",
  "rim_m", "rim_a", "rim_pct", "mid_m", "mid_a", "mid_pct"
)

# Ensure checking names against "clean" names
drop_cols_clean <- janitor::make_clean_names(drop_cols)

important_merged <- important_merged %>%
  select(-any_of(drop_cols_clean))

# -----------------------------
# 2) Rename columns (only if present)
# -----------------------------
rename_wanted <- c(
  "Total_off_2p_attempts_value" = "2pa",
  "Total_off_3p_attempts_value" = "3pa",
  "Total_off_2prim_attempts_value" = "rim attempts",
  "Total_off_2pmid_attempts_value" = "middy attempts",
  "Off_2p_value" = "2p%",
  "Off_3p_value" = "3p%",
  "off_3p_ast_value" = "assisted 3p %",
  "Off_2prim_value" = "rim%",
  "Off_2prim_ast_value" = "assisted rim fg%",
  "Off_2pmid_value" = "middy fg%",
  "Off_2pmid_ast_value" = "assisted middy fg%",
  "Off_2primr_value" = "rim attempt rate",
  "Off_2pmidr_value" = "middy attempt rate",
  "Off_3pr_value" = "3 pt rate",
  "duration_mins_value" = "total minutes",
  "Off_trans_2p_value" = "transition 2p%",
  "Off_trans_3p_value" = "transition 3p%",
  "Off_trans_2prim_value" = "transition rim 2p%",
  "off_trans_2pmid_value" = "transition midrange fg%",
  "Off_efg_value" = "hoop-e efg%",
  "Off_scramble_efg_value" = "2nd chance pts efg%",
  "Off_trans_efg_value" = "transition efg%",
  "Def_2prim_value" = "opponent 2p rim fg%",
  "Off_trans_assist_value" = "% of assists via transition",
  "Off_ast_rim_value" = "% of assists end w/ rim",
  "Off_ast_mid_value" = "% of assists end w/ middy",

  # you typed "Off_ast_mid_value" again for 3p; I'm assuming you meant Off_ast_3p_value
  "Off_ast_3p_value" = "% of assists end w/ 3p",
  "Rec" = "recruit rank",
  "off_adj_rapm_value" = "offensive rapm",
  "def_adj_rapm_value" = "defensive rapm"
)

# 2. Rename columns
message("Renaming columns in important_merged...")

# Robust rename of ALL wanted columns
# Note: Source columns are "clean_names" (lowercase/underscores) because janitor::clean_names was run on hoop data
important_merged <- important_merged %>%
  rename(
    "2pa"                         = any_of(c("total_off_2p_attempts_value")),
    "3pa"                         = any_of(c("total_off_3p_attempts_value")),
    "rim attempts"                = any_of(c("total_off_2prim_attempts_value")),
    "middy attempts"              = any_of(c("total_off_2pmid_attempts_value")),
    "2p%"                         = any_of(c("off_2p_value")),
    "3p%"                         = any_of(c("off_3p_value")),
    "assisted 3p %"               = any_of(c("off_3p_ast_value")),
    "rim%"                        = any_of(c("off_2prim_value")),
    "assisted rim fg%"            = any_of(c("off_2prim_ast_value")),
    "middy fg%"                   = any_of(c("off_2pmid_value")),
    "assisted middy fg%"          = any_of(c("off_2pmid_ast_value")),
    "rim attempt rate"            = any_of(c("off_2primr_value")),
    "middy attempt rate"          = any_of(c("off_2pmidr_value")),
    "3 pt rate"                   = any_of(c("off_3pr_value")),
    "total minutes"               = any_of(c("duration_mins_value")),
    "transition 2p%"              = any_of(c("off_trans_2p_value")),
    "transition 3p%"              = any_of(c("off_trans_3p_value")),
    "transition rim 2p%"          = any_of(c("off_trans_2prim_value")),
    "transition midrange fg%"     = any_of(c("off_trans_2pmid_value")),
    "hoop-e efg%"                 = any_of(c("off_efg_value")),
    "2nd chance pts efg%"         = any_of(c("off_scramble_efg_value")),
    "transition efg%"             = any_of(c("off_trans_efg_value")),
    "opponent 2p rim fg%"         = any_of(c("def_2prim_value")),
    "% of assists via transition" = any_of(c("off_trans_assist_value")),
    "% of assists end w/ rim"     = any_of(c("off_ast_rim_value")),
    "% of assists end w/ middy"   = any_of(c("off_ast_mid_value")),
    "% of assists end w/ 3p"      = any_of(c("off_ast_3p_value")),
    "recruit rank"                = any_of(c("rec", "Rec")),
    "offensive rapm"              = any_of(c("off_adj_rapm_value", "Off_adj_rapm_value")),
    "defensive rapm"              = any_of(c("def_adj_rapm_value", "Def_adj_rapm_value"))
  )

# Verify required columns
if (!"offensive rapm" %in% names(important_merged)) {
  message("WARNING: 'offensive rapm' column missing! Creating empty column.")
  important_merged$`offensive rapm` <- NA_real_
}
if (!"defensive rapm" %in% names(important_merged)) {
  message("WARNING: 'defensive rapm' column missing! Creating empty column.")
  important_merged$`defensive rapm` <- NA_real_
}

# -----------------------------
# 3) Add Total RAPM
# total RAPM = offensive rapm - defensive rapm
# (defensive rapm is typically negative, so this effectively adds magnitude)
# -----------------------------
important_merged <- important_merged %>%
  mutate(
    `total RAPM` = suppressWarnings(as.numeric(`offensive rapm`)) -
      suppressWarnings(as.numeric(`defensive rapm`))
  )

# -----------------------------
# 4) Add per-100 possession reference stats
# -----------------------------
important_merged <- important_merged %>%
  mutate(
    off_team_poss_value = suppressWarnings(as.numeric(off_team_poss_value)),
    def_team_poss_value = suppressWarnings(as.numeric(def_team_poss_value)),
    g = suppressWarnings(as.numeric(g)),
    spg = suppressWarnings(as.numeric(spg)),
    bpg = suppressWarnings(as.numeric(bpg)),
    stops = suppressWarnings(as.numeric(stops)),
    `3pa` = suppressWarnings(as.numeric(`3pa`)),
    `2pa` = suppressWarnings(as.numeric(`2pa`)),
    dunk_a = suppressWarnings(as.numeric(dunk_a)),
    `rim attempts` = suppressWarnings(as.numeric(`rim attempts`)),
    `middy attempts` = suppressWarnings(as.numeric(`middy attempts`)),
    `stls/100` = ifelse(def_team_poss_value > 0, 100 * spg * g / def_team_poss_value, NA_real_),
    `blks/100` = ifelse(def_team_poss_value > 0, 100 * bpg * g / def_team_poss_value, NA_real_),
    `stops/100` = ifelse(def_team_poss_value > 0, 100 * stops / def_team_poss_value, NA_real_),
    `3pa/100` = ifelse(off_team_poss_value > 0, 100 * `3pa` / off_team_poss_value, NA_real_),
    `midfga/100` = ifelse(off_team_poss_value > 0, 100 * `middy attempts` / off_team_poss_value, NA_real_),
    `dunkfga/100` = ifelse(off_team_poss_value > 0, 100 * dunk_a / off_team_poss_value, NA_real_),
    `rimfga/100` = ifelse(off_team_poss_value > 0, 100 * `rim attempts` / off_team_poss_value, NA_real_),
    `2pa/100` = ifelse(off_team_poss_value > 0, 100 * `2pa` / off_team_poss_value, NA_real_)
  )


## ----savea----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
out_path <- "season.csv"
write_csv(important_merged, out_path)


## ----saveasd--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ============================================
# CAREER CSV from mergednew (group by torvik_id)
# One row per torvik_id
# ============================================


# ---- load (use your object if already in memory) ----
# df <- mergednew            # if you already have it as an object
df <- important_merged

# ---- helpers ----
wavg <- function(x, w) {
  ok <- !is.na(x) & !is.na(w) & w > 0
  if (!any(ok)) {
    return(NA_real_)
  }
  sum(x[ok] * w[ok], na.rm = TRUE) / sum(w[ok], na.rm = TRUE)
}

# pick the "most recent season row" per torvik_id
has_torvik_year <- "torvik_year" %in% names(df)
has_year <- "year" %in% names(df)

if (!has_torvik_year && !has_year) {
  stop("df has neither 'torvik_year' nor 'year' column. Check names(df).")
}

df <- df %>%
  mutate(
    .season_sort = if (has_torvik_year) {
      suppressWarnings(as.integer(.data[["torvik_year"]]))
    } else {
      suppressWarnings(as.integer(.data[["year"]]))
    }
  )


last_row <- df %>%
  filter(!is.na(roster_ncaa_id), roster_ncaa_id != "") %>%
  group_by(roster_ncaa_id) %>%
  slice_max(.season_sort, n = 1, with_ties = FALSE) %>%
  ungroup()

# ---- aggregate career stats per torvik_id ----
career_agg <- df %>%
  filter(!is.na(roster_ncaa_id), roster_ncaa_id != "") %>%
  group_by(roster_ncaa_id) %>%
  summarise(
    # helper
    career_total_assists = sum(apg * g, na.rm = TRUE),

    # SUM columns
    career_2pa = sum(`2pa`, na.rm = TRUE),
    career_3pa = sum(`3pa`, na.rm = TRUE),
    career_rim_attempts = sum(`rim attempts`, na.rm = TRUE),
    career_middy_attempts = sum(`middy attempts`, na.rm = TRUE),
    career_fga = sum(fga, na.rm = TRUE),
    career_total_minutes = sum(`total minutes`, na.rm = TRUE),
    career_stops = sum(stops, na.rm = TRUE),
    career_fta = sum(fta, na.rm = TRUE),
    career_ftm = sum(ftm, na.rm = TRUE),
    career_fgm = sum(fgm, na.rm = TRUE),
    career_dunk_a = sum(dunk_a, na.rm = TRUE),
    career_dunk_m = sum(dunk_m, na.rm = TRUE),
    career_three_a = sum(three_a, na.rm = TRUE),
    career_three_m = sum(three_m, na.rm = TRUE),
    career_two_a = sum(two_a, na.rm = TRUE),
    career_two_m = sum(two_m, na.rm = TRUE),
    career_g = sum(g, na.rm = TRUE),
    # --- POSSESSIONS (career totals) ---
    career_off_poss = sum(off_team_poss_value, na.rm = TRUE),
    career_def_poss = sum(def_team_poss_value, na.rm = TRUE),

    # --- TOTAL STOCKS (convert per-game -> totals, then sum) ---
    career_total_steals = sum(spg * g, na.rm = TRUE),
    career_total_blocks = sum(bpg * g, na.rm = TRUE),

    # --- TOTAL attempts for rate stats (sum raw attempts) ---
    career_total_3pa = sum(`3pa`, na.rm = TRUE),
    career_total_2pa = sum(`2pa`, na.rm = TRUE),
    career_total_mid_att = sum(`middy attempts`, na.rm = TRUE),
    career_total_rim_att = sum(`rim attempts`, na.rm = TRUE),
    career_total_dunk_att = sum(dunk_a, na.rm = TRUE),


    # WEIGHTED % (by attempts)
    career_2p_pct = wavg(`2p%`, `2pa`),
    career_3p_pct = wavg(`3p%`, `3pa`),
    career_rim_pct = wavg(`rim%`, `rim attempts`),
    career_middy_fg_pct = wavg(`middy fg%`, `middy attempts`),

    # WEIGHTED rates (by season fga)
    career_rim_attempt_rate = wavg(`rim attempt rate`, fga),
    career_middy_attempt_rate = wavg(`middy attempt rate`, fga),

    # WEIGHTED fg_pct (by season fga)
    career_fg_pct = wavg(fg_pct, fga),

    # WEIGHTED ftr (by season total minutes)
    career_ftr = wavg(ftr, `total minutes`),

    # assists-end weights (by apg*g)
    career_pct_ast_end_3p = wavg(`% of assists end w/ 3p`, apg * g),
    career_pct_ast_end_middy = wavg(`% of assists end w/ middy`, apg * g),
    career_pct_ast_end_rim = wavg(`% of assists end w/ rim`, apg * g),

    # MINUTES-weighted rate stats
    career_stl = wavg(stl, `total minutes`),
    career_blk = wavg(blk, `total minutes`),
    career_to = wavg(to, `total minutes`),
    career_ast = wavg(ast, `total minutes`),
    career_obpm = wavg(obpm, `total minutes`),
    career_dbpm = wavg(dbpm, `total minutes`),
    career_bpm = wavg(bpm, `total minutes`),
    career_usg = wavg(usg, `total minutes`),
    career_oreb_rate = wavg(oreb_rate, `total minutes`),
    career_dreb_rate = wavg(dreb_rate, `total minutes`),
    career_porpag = wavg(porpag, `total minutes`),
    career_dporpag = wavg(dporpag, `total minutes`),

    # RAPM (simple average across seasons)
    career_offensive_rapm = mean(`offensive rapm`, na.rm = TRUE),
    career_defensive_rapm = mean(`defensive rapm`, na.rm = TRUE),
    career_rapm = mean(`total RAPM`, na.rm = TRUE),


    # new: adjusted apg/tov/ast_to (minutes-weighted)
    career_adjusted_apg = wavg(apg, `total minutes`),
    career_adjusted_tov = wavg(tov, `total minutes`),
    career_adjusted_ast_to = wavg(ast_to, `total minutes`),
    .groups = "drop"
  ) %>%
  mutate(
    # derived % that depend on career totals

    # NOTE: you wrote ft_pct = fta/ftm; that’s backwards in normal life.
    # I’m using the normal definition ft_pct = ftm/fta. Change if you truly want the other way.
    career_ft_pct = ifelse(career_fta > 0, career_ftm / career_fta, NA_real_),

    # NOTE: you wrote 3 pt rate = 3pa/fgm (weird). Doing exactly that:
    career_3_pt_rate = ifelse(career_fgm > 0, career_3pa / career_fga, NA_real_),

    # NOTE: you wrote dunk_pct = dunk_a/dunk_m (will be >1 usually). Normal is dunk_m/dunk_a.
    career_dunk_pct = ifelse(career_dunk_a > 0, career_dunk_m / career_dunk_a, NA_real_),

    # TS%: you described the denominator slightly wrong.
    # Correct TS% is: points / (2*(FGA + 0.44*FTA))
    career_points = 2 * career_two_m + 3 * career_three_m + career_ftm,
    career_ts = ifelse((career_fga + 0.44 * career_fta) > 0,
      career_points / (2 * (career_fga + 0.44 * career_fta)),
      NA_real_
    ),
    `career stls/100` = ifelse(career_def_poss > 0, 100 * career_total_steals / career_def_poss, NA_real_),
    `career blks/100` = ifelse(career_def_poss > 0, 100 * career_total_blocks / career_def_poss, NA_real_),
    `career stops/100` = ifelse(career_def_poss > 0, 100 * career_stops / career_def_poss, NA_real_), # you already sum stops as career_stops

    `career 3pa/100` = ifelse(career_off_poss > 0, 100 * career_total_3pa / career_off_poss, NA_real_),
    `career 2pa/100` = ifelse(career_off_poss > 0, 100 * career_total_2pa / career_off_poss, NA_real_),
    `career midfga/100` = ifelse(career_off_poss > 0, 100 * career_total_mid_att / career_off_poss, NA_real_),
    `career rimfga/100` = ifelse(career_off_poss > 0, 100 * career_total_rim_att / career_off_poss, NA_real_),
    `career dunkfga/100` = ifelse(career_off_poss > 0, 100 * career_total_dunk_att / career_off_poss, NA_real_)
  )


# ---- build final career table: start from most-recent row, overwrite with career values ----
career <- last_row %>%
  left_join(career_agg, by = "roster_ncaa_id") %>%
  mutate(
    # copy-most-recent columns are already in last_row; we only overwrite computed ones:
    `2pa` = career_2pa,
    `3pa` = career_3pa,
    `rim attempts` = career_rim_attempts,
    `middy attempts` = career_middy_attempts,
    fga = career_fga,
    `total minutes` = career_total_minutes,
    stops = career_stops,
    fta = career_fta,
    ftm = career_ftm,
    fgm = career_fgm,
    dunk_a = career_dunk_a,
    dunk_m = career_dunk_m,
    three_a = career_three_a,
    three_m = career_three_m,
    two_a = career_two_a,
    two_m = career_two_m,
    g = career_g,
    `2p%` = career_2p_pct,
    `3p%` = career_3p_pct,
    `rim%` = career_rim_pct,
    `middy fg%` = career_middy_fg_pct,
    `rim attempt rate` = career_rim_attempt_rate,
    `middy attempt rate` = career_middy_attempt_rate,
    fg_pct = career_fg_pct,
    ftr = career_ftr,
    `% of assists end w/ 3p` = career_pct_ast_end_3p,
    `% of assists end w/ middy` = career_pct_ast_end_middy,
    `% of assists end w/ rim` = career_pct_ast_end_rim,
    off_team_poss_value = career_off_poss,
    def_team_poss_value = career_def_poss,
    `stls/100` = `career stls/100`,
    `blks/100` = `career blks/100`,
    `stops/100` = `career stops/100`,
    `3pa/100` = `career 3pa/100`,
    `2pa/100` = `career 2pa/100`,
    `midfga/100` = `career midfga/100`,
    `rimfga/100` = `career rimfga/100`,
    `dunkfga/100` = `career dunkfga/100`,
    stl = career_stl,
    blk = career_blk,
    to = career_to,
    ast = career_ast,
    obpm = career_obpm,
    dbpm = career_dbpm,
    bpm = career_bpm,
    usg = career_usg,
    oreb_rate = career_oreb_rate,
    dreb_rate = career_dreb_rate,
    porpag = career_porpag,
    dporpag = career_dporpag,
    `offensive rapm` = career_offensive_rapm,
    `defensive rapm` = career_defensive_rapm,
    `total RAPM` = career_rapm,
    ft_pct = career_ft_pct,
    ts = career_ts,
    `3 pt rate` = career_3_pt_rate,
    dunk_pct = career_dunk_pct,

    # NEW columns
    adjusted_apg = career_adjusted_apg,
    adjusted_tov = career_adjusted_tov,
    adjusted_ast_to = career_adjusted_ast_to
  ) %>%
  # blank-out columns you explicitly want blank in career
  mutate(
    `assisted 3p %` = NA_real_,
    `assisted rim fg%` = NA_real_,
    `assisted middy fg%` = NA_real_,
    torvik_year = NA_integer_,
    pfr = NA_real_,
    adj_oe = NA_real_,
    drtg = NA_real_,
    adj_de = NA_real_,
    efg = NA_real_,
    ortg = NA_real_,
    bpg = NA_real_,
    spg = NA_real_,
    apg = NA_real_,
    tov = NA_real_,
    ast_to = NA_real_,
    rpg = NA_real_,
    dreb = NA_real_,
    oreb = NA_real_,
    ppg = NA_real_,
    mpg = NA_real_,
    min = NA_real_,
    exp = NA_real_,
    `transition 2p%` = NA_real_,
    `transition 3p%` = NA_real_,
    `transition midrange fg%` = NA_real_
  ) %>%
  select(-starts_with("career_"), -.season_sort)

# ---- write ----
write_csv(career, "career.csv")


## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------
# FINAL CHUNK: Save to Public Data
# -----------------------------

# 1. Define the output paths
# (REPO_ROOT was defined in Chunk 1)
out_season_path <- file.path(REPO_ROOT, "public", "data", "season.csv")
out_career_path <- file.path(REPO_ROOT, "public", "data", "career.csv")

# 2. Save the files
cat("[save] Writing Season file to:", out_season_path, "\n")
readr::write_csv(important_merged, out_season_path)

cat("[save] Writing Career file to:", out_career_path, "\n")
readr::write_csv(career, out_career_path)

# 3. Validation
if (file.exists(out_season_path) && file.exists(out_career_path)) {
  cat("\nSUCCESS! Files are now in public/data/.\n")
  cat("You can now commit and push these changes to GitHub.\n")
} else {
  cat("\nERROR: Files were not saved. Check permissions or paths.\n")
}
