# =============================================================================
# 
#
# Processes raw per-individual output from SLiM population genetics
# simulations into per-population summary statistics (quantile
# distributions) and matched individual-level tables, split by timepoint.
#
# For each simulation run, this script:
#   1. Reads per-individual data ("*_IndData.txt") and extracts two
#      pre-bottleneck timepoints (Year 0 and Year 90).
#   2. Reads population size data ("*_N.txt") and extracts two
#      post-bottleneck target timepoints (Year 210 and Year 310).
#   3. Computes decile summary statistics (0th-100th percentile) across
#      individuals for each variable, at each timepoint.
#   4. Parses simulation parameters (mutation rate, ancestral/bottleneck/
#      recovered population sizes, litter size) from the input filename.
#   5. Combines results across all runs into training tables for
#      downstream modelling (e.g. DNN training data).
#
# Input file naming convention (as produced by the SLiM simulation runs):
#   <prefix>_Mu<mu>_Na<Na>_Nb<Nb>_Nr<Nr>_LitterSize<litter>_Seed<seed>_IndData.txt
#   <prefix>_Mu<mu>_Na<Na>_Nb<Nb>_Nr<Nr>_LitterSize<litter>_Seed<seed>_N.txt
#
# Outputs (written to OUTPUT_DIR):
#   disty0.csv, disty90.csv                 - population-level quantile
#                                              distributions at Year 0 / 90
#   dist_target_210.csv, dist_target_310.csv - target Year values at Year 210 / 310
#   indivy0.csv, indivy90.csv                - individual-level data at Year 0 / 90
#   indiv_target_210.csv, indiv_target_310.csv - individual-level target values
# =============================================================================

library(tidyr)

# -----------------------------------------------------------------------------
# User-configurable paths
# -----------------------------------------------------------------------------
INPUT_DIR  <- "data/raw"        # directory containing *_IndData.txt and *_N.txt
OUTPUT_DIR <- "data/processed"  # directory to write summary CSVs to

# -----------------------------------------------------------------------------
# preprocess_slim()
#
# Reads a single simulation run's individual-level data file and matching
# population-size file, and produces summary and individual-level tables
# at the timepoints of interest.
#
# Args:
#   ind_data_fp : path to the "*_IndData.txt" file for one simulation run
#   n_data_fp   : path to the matching "*_N.txt" file for the same run
#
# Returns:
#   A named list of eight data frames:
#     dist_y0, dist_y90            - per-variable quantile distributions
#                                     (Year 0 and Year 90), one row per
#                                     variable, with run parameters attached
#     target_210, target_310       - target Year values at Year 210 / 310
#     indiv_y0, indiv_y90          - raw individual-level rows at Year 0 / 90,
#                                     with run parameters attached
#     indiv_target_210,
#     indiv_target_310             - target Year values repeated to match
#                                     the individual-level row count (n = 10),
#                                     for use as training labels
# -----------------------------------------------------------------------------
preprocess_slim <- function(ind_data_fp, n_data_fp) {

  n_individuals_per_run <- 10  # number of individuals sampled per timepoint

  ind_df <- read.table(ind_data_fp, header = TRUE)
  df_y0  <- ind_df %>% filter(Year == 0)
  df_y90 <- ind_df %>% filter(Year == 90)  # Year 0 and Year 90 = 100 and 10
                                            # years before bottleneck start

  # --- Parse run parameters from the filename ------------------------------
  run_params <- data.frame(
    mu         = as.numeric(str_extract(ind_data_fp, "(?<=Mu).*(?=_Na)")),
    na         = as.numeric(str_extract(ind_data_fp, "(?<=Na).*(?=_Nb)")),
    nb         = as.numeric(str_extract(ind_data_fp, "(?<=Nb).*(?=_Nr)")),
    nr         = as.numeric(str_extract(ind_data_fp, "(?<=Nr).*(?=_Litter)")),
    littersize = as.numeric(str_extract(ind_data_fp, "(?<=LitterSize).*(?=_Seed)"))
  )

  # --- Helper: compute decile summary stats for one timepoint's data -------
  # ------ Note: only 0, 20, 40, 60, 80, 100 were used for downstream inputs
  summarise_deciles <- function(df_timepoint) {
    df_timepoint %>%
      pivot_longer(cols = -Year, names_to = "variable", values_to = "value") %>%
      group_by(variable) %>%
      summarise(
        q0   = quantile(value, 0.0), q10 = quantile(value, 0.1),
        q20  = quantile(value, 0.2), q30 = quantile(value, 0.3),
        q40  = quantile(value, 0.4), q50 = quantile(value, 0.5),
        q60  = quantile(value, 0.6), q70 = quantile(value, 0.7),
        q80  = quantile(value, 0.8), q90 = quantile(value, 0.9),
        q100 = quantile(value, 1.0),
        .groups = "drop"
      ) %>%
      pivot_wider(names_from = variable, values_from = q0:q100) %>%
      bind_cols(run_params) %>%
      mutate(match_col = ind_data_fp)
  }

  dist_y0  <- summarise_deciles(df_y0)
  dist_y90 <- summarise_deciles(df_y90)

  indiv_y0  <- df_y0  %>% bind_cols(run_params) %>% mutate(match_col = ind_data_fp)
  indiv_y90 <- df_y90 %>% bind_cols(run_params) %>% mutate(match_col = ind_data_fp)

  # --- Format target (population size) data ---------------------------------
  n_df <- read.table(n_data_fp, header = TRUE)

  extract_target <- function(target_year, repeat_rows = 1) {
    n_df %>%
      filter(Year == target_year) %>%
      select(Year) %>%
      mutate(match_col = ind_data_fp) %>%
      slice(rep(1:n(), each = repeat_rows))
  }

  target_210 <- extract_target(210)
  target_310 <- extract_target(310)
  indiv_target_210 <- extract_target(210, repeat_rows = n_individuals_per_run)
  indiv_target_310 <- extract_target(310, repeat_rows = n_individuals_per_run)

  list(
    dist_y0 = dist_y0, dist_y90 = dist_y90,
    target_210 = target_210, target_310 = target_310,
    indiv_y0 = indiv_y0, indiv_y90 = indiv_y90,
    indiv_target_210 = indiv_target_210, indiv_target_310 = indiv_target_310
  )
}

# -----------------------------------------------------------------------------
# Batch processing: run preprocess_slim() over every simulation run in
# INPUT_DIR and combine results into single tables.
# -----------------------------------------------------------------------------
run_batch_processing <- function(input_dir, output_dir) {

  n_data_files <- list.files(input_dir, pattern = "_N.txt", full.names = TRUE)

  results <- list(
    dist_y0 = list(), dist_y90 = list(),
    target_210 = list(), target_310 = list(),
    indiv_y0 = list(), indiv_y90 = list(),
    indiv_target_210 = list(), indiv_target_310 = list()
  )

  for (n_fp in n_data_files) {
    ind_data_fp <- str_replace(n_fp, "_N.txt", "_IndData.txt")
    pps <- preprocess_slim(ind_data_fp, n_fp)
    for (name in names(results)) {
      results[[name]] <- c(results[[name]], list(pps[[name]]))
    }
  }

  combined <- lapply(results, bind_rows)

  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  write.csv(combined$dist_y0,          file.path(output_dir, "disty0.csv"))
  write.csv(combined$dist_y90,         file.path(output_dir, "disty90.csv"))
  write.csv(combined$target_210,       file.path(output_dir, "dist_target_210.csv"))
  write.csv(combined$target_310,       file.path(output_dir, "dist_target_310.csv"))
  write.csv(combined$indiv_y0,         file.path(output_dir, "indivy0.csv"))
  write.csv(combined$indiv_y90,        file.path(output_dir, "indivy90.csv"))
  write.csv(combined$indiv_target_210, file.path(output_dir, "indiv_target_210.csv"))
  write.csv(combined$indiv_target_310, file.path(output_dir, "indiv_target_310.csv"))

  invisible(combined)
}

# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
# Process a single simulation run:
#
#   result <- preprocess_slim(
#     ind_data_fp = "data/raw/DNNTrainingData1_Bottleneck_Mu2.0e-08_Na500_Nb20_Nr500_LitterSize8_Seed101_IndData.txt",
#     n_data_fp   = "data/raw/DNNTrainingData1_Bottleneck_Mu2.0e-08_Na500_Nb20_Nr500_LitterSize8_Seed101_N.txt"
#   )
#
# Process all simulation runs in INPUT_DIR and write combined CSVs to OUTPUT_DIR:
#
#   run_batch_processing(INPUT_DIR, OUTPUT_DIR)
