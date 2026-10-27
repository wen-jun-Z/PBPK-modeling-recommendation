suppressPackageStartupMessages(library(ospsuite))
suppressPackageStartupMessages(library(jsonlite))

script_argument <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_argument) != 1) {
  stop("Could not resolve caffeine v3 R script directory.")
}
script_directory <- dirname(normalizePath(
  sub("^--file=", "", script_argument[[1]]),
  mustWork = TRUE
))
source(file.path(script_directory, "caffeine_v3_common.R"), local = TRUE)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2 || length(args) > 3) {
  stop(
    paste(
      "Usage: caffeine_v3_identify.R <identification-config.json>",
      "<preflight|identify> [phase-output-directory]"
    )
  )
}
config_path <- normalizePath(args[[1]], mustWork = TRUE)
phase <- args[[2]]
if (!(phase %in% c("preflight", "identify"))) {
  stop(paste("Unknown caffeine v3 identification phase:", phase))
}
config <- fromJSON(config_path, simplifyVector = FALSE)
if (!isTRUE(config$accepted)) {
  stop("Identification config is not accepted.")
}
if (config$protocol_version != "caffeine-fit-validation-protocol-v3-1.0.0") {
  stop("Caffeine v3 protocol version drifted.")
}
if (as.character(packageVersion("ospsuite")) != config$ospsuite_r_version) {
  stop("OSPSuite-R version mismatch.")
}

config_sha256 <- v3_sha256_file(config_path)
if (phase == "preflight") {
  if (length(args) != 3) {
    stop("Preflight requires a separate immutable output directory.")
  }
  output_directory <- v3_ensure_new_directory(args[[3]])
} else {
  if (length(args) != 2) {
    stop("Identification does not accept an output override.")
  }
  root_output <- config$output_directory
  if (dir.exists(root_output)) {
    stop(paste("Immutable identification run already exists:", root_output))
  }
  dir.create(root_output, recursive = TRUE, showWarnings = FALSE)
  output_directory <- v3_ensure_new_directory(
    file.path(root_output, "01_identification")
  )
}

target_output <- config$target_output
penalty_objective <- 1e12
calibration_names <- c("CAL_BLANCHARD_IV", "CAL_BLANCHARD_PO")
scenarios <- config$calibration_scenarios
names(scenarios) <- vapply(scenarios, function(x) x$name, character(1))
if (!identical(names(scenarios), calibration_names)) {
  stop("Identification config contains a non-calibration scenario.")
}
if (any(vapply(scenarios, function(x) x$split, character(1)) != "calibration")) {
  stop("Identification split isolation failed.")
}

parameters <- config$parameters
names(parameters) <- vapply(parameters, function(x) x$parameter_id, character(1))
point_ids <- names(parameters)[vapply(
  parameters,
  function(x) x$identification_mode == "point_estimated",
  logical(1)
)]
set_ids <- names(parameters)[vapply(
  parameters,
  function(x) x$identification_mode == "set_identified",
  logical(1)
)]
clearance_id <- "total_cyp1a2_specific_clearance"
permeability_id <- "specific_intestinal_permeability_transcellular"
if (!identical(point_ids, clearance_id)) {
  stop("Clearance must be the only point-estimated parameter.")
}
if (!identical(set_ids, permeability_id)) {
  stop("Permeability must be the only set-identified parameter.")
}

mapping_rows <- v3_mapping_rows(parameters)
mapping_paths <- vapply(mapping_rows, function(x) x$runtime_path, character(1))
if (length(mapping_rows) != 10 || anyDuplicated(mapping_paths)) {
  stop("Runtime mapping count or uniqueness drifted.")
}
mapping_by_path <- mapping_rows
names(mapping_by_path) <- mapping_paths

variable_ids <- c(clearance_id, permeability_id)
variable_paths <- names(v3_runtime_values_for(
  c(
    total_cyp1a2_specific_clearance = parameters[[clearance_id]]$value,
    specific_intestinal_permeability_transcellular =
      parameters[[permeability_id]]$value
  ),
  parameters,
  variable_ids
))
if (length(variable_paths) != 4 || anyDuplicated(variable_paths)) {
  stop("Caffeine v3 variable runtime path set drifted.")
}

simulations <- list()
preflight_rows <- list()
for (scenario_name in calibration_names) {
  scenario <- scenarios[[scenario_name]]
  pkml_path <- file.path(config$runtime_root, scenario$pkml$relative_path)
  if (v3_sha256_file(pkml_path) != scenario$pkml$sha256) {
    stop(paste("PKML checksum mismatch:", scenario_name))
  }
  simulation <- loadSimulation(
    pkml_path,
    loadFromCache = FALSE,
    addToCache = FALSE,
    resetIds = FALSE
  )
  setOutputs(target_output, simulation, stopIfNotFound = TRUE)
  for (mapping in mapping_rows) {
    logical_spec <- parameters[[mapping$parameter_id]]
    runtime_parameter <- getParameter(mapping$runtime_path, simulation)
    mapped_value <- as.numeric(logical_spec$value) * mapping$multiplier
    expected_base <- v3_to_base_value(
      runtime_parameter,
      mapped_value,
      mapping$unit
    )
    actual_base <- as.numeric(runtime_parameter$value)
    valid <- v3_nearly_equal(actual_base, expected_base)
    preflight_rows[[length(preflight_rows) + 1]] <- data.frame(
      scenario_name = scenario_name,
      parameter_id = mapping$parameter_id,
      runtime_path = mapping$runtime_path,
      mapping_rule = mapping$mapping_rule,
      multiplier = mapping$multiplier,
      expected_logical_value = as.numeric(logical_spec$value),
      expected_mapped_value = mapped_value,
      expected_unit = mapping$unit,
      expected_base_value = expected_base,
      actual_base_value = actual_base,
      actual_base_unit = ifelse(
        is.null(runtime_parameter$unit),
        "",
        runtime_parameter$unit
      ),
      valid = valid,
      stringsAsFactors = FALSE
    )
    if (!valid) {
      stop(paste("Runtime parameter value drifted:", mapping$runtime_path))
    }
  }
  pka_count <- getParameter("Caffeine|pKa_Bases_Count", simulation)
  if (!v3_nearly_equal(as.numeric(pka_count$value), 1)) {
    stop(paste("Caffeine base pKa count is not one:", scenario_name))
  }
  simulations[[scenario_name]] <- simulation
}
preflight_table <- do.call(rbind, preflight_rows)
v3_write_csv(
  preflight_table,
  file.path(output_directory, "runtime_parameter_preflight.csv")
)

runtime_parameters_by_path <- lapply(
  mapping_paths,
  function(path) getParameter(path, simulations[[calibration_names[[1]]]])
)
names(runtime_parameters_by_path) <- mapping_paths

batch_value_vector <- function(batch, values_by_path) {
  paths <- batch$getVariableParameters()
  if (!is.character(paths) || anyDuplicated(paths)) {
    stop("SimulationBatch variable parameter API contract drifted.")
  }
  if (!setequal(paths, names(values_by_path))) {
    stop("SimulationBatch variable parameter set drifted.")
  }
  vapply(seq_along(paths), function(index) {
    path <- paths[[index]]
    mapping <- mapping_by_path[[path]]
    runtime_parameter <- runtime_parameters_by_path[[path]]
    v3_to_base_value(
      runtime_parameter,
      values_by_path[[path]],
      mapping$unit
    )
  }, numeric(1))
}

make_batches <- function(simulation_names = calibration_names) {
  output <- lapply(simulation_names, function(name) {
    createSimulationBatch(
      simulation = simulations[[name]],
      parametersOrPaths = variable_paths
    )
  })
  names(output) <- simulation_names
  output
}

run_parameter_set <- function(batches, logical_values) {
  values_by_path <- v3_runtime_values_for(
    logical_values,
    parameters,
    variable_ids
  )
  result_ids <- list()
  for (name in names(batches)) {
    values <- batch_value_vector(batches[[name]], values_by_path)
    result_ids[[name]] <- batches[[name]]$addRunValues(
      parameterValues = values
    )
  }
  results <- runSimulationBatches(
    unname(batches),
    silentMode = TRUE,
    stopIfFails = TRUE
  )
  flattened <- v3_flatten_batch_results(results)
  output <- lapply(names(batches), function(name) {
    v3_curve_from_result(
      v3_result_for_id(flattened, result_ids[[name]]),
      target_output
    )
  })
  names(output) <- names(batches)
  output
}

initial_logical <- c(
  total_cyp1a2_specific_clearance = as.numeric(
    parameters[[clearance_id]]$value
  ),
  specific_intestinal_permeability_transcellular = as.numeric(
    parameters[[permeability_id]]$value
  )
)
contract_curves <- run_parameter_set(
  make_batches(calibration_names[[1]]),
  initial_logical
)
contract_curve <- contract_curves[[calibration_names[[1]]]]
batch_contract <- list(
  schema_version = "1.0.0",
  protocol_version = config$protocol_version,
  generated_at = v3_utc_now(),
  scenario_name = calibration_names[[1]],
  logical_parameter_ids = as.list(variable_ids),
  variable_parameter_paths = as.list(variable_paths),
  output_path = target_output,
  output_unit = "µmol/l",
  row_count = nrow(contract_curve),
  time_min = min(contract_curve$time_min),
  time_max = max(contract_curve$time_min),
  accepted = nrow(contract_curve) > 1 &&
    all(is.finite(contract_curve$concentration_umol_per_l)) &&
    all(contract_curve$concentration_umol_per_l >= 0)
)
v3_write_json(
  batch_contract,
  file.path(output_directory, "batch_api_contract.json")
)
if (!batch_contract$accepted) {
  stop("SimulationBatch API contract run failed.")
}

if (phase == "preflight") {
  manifest <- list(
    schema_version = "1.0.0",
    protocol_version = config$protocol_version,
    generated_at = v3_utc_now(),
    config_sha256 = config_sha256,
    phase = phase,
    scenario_count = length(simulations),
    parameter_mapping_check_count = nrow(preflight_table),
    validation_package_loaded = FALSE,
    validation_scenarios_loaded = FALSE,
    accepted = all(preflight_table$valid) && batch_contract$accepted,
    artifacts = v3_artifact_inventory(
      output_directory,
      excluded = "preflight_manifest.json"
    )
  )
  v3_write_json(manifest, file.path(output_directory, "preflight_manifest.json"))
  quit(status = ifelse(manifest$accepted, 0, 2))
}

observations <- lapply(scenarios, v3_observation_frame)
if (nrow(observations[["CAL_BLANCHARD_IV"]]) != 17) {
  stop("Blanchard IV observation count drifted.")
}
if (nrow(observations[["CAL_BLANCHARD_PO"]]) != 11) {
  stop("Blanchard PO observation count drifted.")
}
iv_observations <- observations[["CAL_BLANCHARD_IV"]]
iv_objective_rows <- iv_observations[iv_observations$included, ]
if (nrow(iv_objective_rows) != 17 || any(iv_objective_rows$observed_umol_per_l <= 0)) {
  stop("Blanchard IV objective observations drifted.")
}

clearance_spec <- parameters[[clearance_id]]
clearance_lower <- as.numeric(clearance_spec$lower_bound)
clearance_upper <- as.numeric(clearance_spec$upper_bound)
clearance_span <- clearance_upper - clearance_lower
if (
  !is.finite(clearance_span) || clearance_span <= 0 ||
    clearance_lower != 0
) {
  stop("Frozen clearance search domain drifted.")
}
theta_from_z <- function(z) clearance_lower + as.numeric(z) * clearance_span
z_from_theta <- function(theta) (as.numeric(theta) - clearance_lower) / clearance_span

evaluation_path <- file.path(output_directory, "clearance_evaluations.csv")
evaluation_index <- 0L
evaluation_cache <- new.env(hash = TRUE, parent = emptyenv())
evaluate_clearance <- function(
  z,
  start_index,
  evaluation_phase,
  use_cache = TRUE,
  retain_curve = FALSE
) {
  evaluation_index <<- evaluation_index + 1L
  if (!is.finite(z) || z < 0 || z > 1) {
    return(list(
      objective = penalty_objective,
      curve = NULL,
      status = "outside_bounds",
      error = "normalized clearance outside [0,1]"
    ))
  }
  key <- sprintf("%.17g", z)
  clearance <- theta_from_z(z)
  logical_values <- c(
    total_cyp1a2_specific_clearance = clearance,
    specific_intestinal_permeability_transcellular = as.numeric(
      parameters[[permeability_id]]$value
    )
  )
  if (use_cache && exists(key, envir = evaluation_cache, inherits = FALSE)) {
    cached <- get(key, envir = evaluation_cache, inherits = FALSE)
    evaluated <- cached
    status <- "cached"
  } else {
    evaluated <- tryCatch({
      curve <- run_parameter_set(
        make_batches("CAL_BLANCHARD_IV"),
        logical_values
      )[["CAL_BLANCHARD_IV"]]
      predicted <- v3_interpolate_curve(curve, iv_objective_rows$time_min)
      if (any(predicted <= 0)) {
        stop("Clearance prediction is non-positive.")
      }
      objective <- mean(
        (log(predicted) - log(iv_objective_rows$observed_umol_per_l))^2
      )
      list(objective = objective, curve = curve, status = "ok", error = "")
    }, error = function(error) {
      list(
        objective = penalty_objective,
        curve = NULL,
        status = "failed",
        error = conditionMessage(error)
      )
    })
    status <- evaluated$status
    if (evaluated$status == "ok") {
      assign(
        key,
        evaluated[c("objective", "curve", "status", "error")],
        envir = evaluation_cache
      )
    }
  }
  v3_append_csv(
    data.frame(
      evaluation_index = evaluation_index,
      start_index = start_index,
      phase = evaluation_phase,
      timestamp_utc = v3_utc_now(),
      z_total_clearance = z,
      total_cyp1a2_specific_clearance = clearance,
      objective = evaluated$objective,
      status = status,
      error = ifelse(is.null(evaluated$error), "", evaluated$error),
      stringsAsFactors = FALSE
    ),
    evaluation_path
  )
  if (!retain_curve) {
    evaluated$curve <- NULL
  }
  evaluated
}

start_theta <- vapply(
  config$clearance_starts,
  function(x) as.numeric(x$total_cyp1a2_specific_clearance),
  numeric(1)
)
start_z <- vapply(start_theta, z_from_theta, numeric(1))
if (length(start_z) != 9 || any(start_z <= 0) || any(start_z >= 1)) {
  stop("Locked nine-start clearance grid is invalid.")
}

multi_start_rows <- list()
for (start_index in seq_along(start_z)) {
  result <- optim(
    par = start_z[[start_index]],
    fn = function(z) {
      evaluate_clearance(
        z[[1]],
        start_index,
        "optimization",
        use_cache = TRUE
      )$objective
    },
    method = "L-BFGS-B",
    lower = 0,
    upper = 1,
    control = list(
      maxit = 250,
      factr = 1e7,
      pgtol = 1e-8,
      parscale = 1,
      ndeps = 1e-4
    )
  )
  multi_start_rows[[start_index]] <- data.frame(
    start_index = start_index,
    initial_z_total_clearance = start_z[[start_index]],
    initial_total_clearance = start_theta[[start_index]],
    fitted_z_total_clearance = result$par[[1]],
    fitted_total_clearance = theta_from_z(result$par[[1]]),
    objective = result$value,
    convergence_code = result$convergence,
    function_evaluations = unname(result$counts[["function"]]),
    gradient_evaluations = unname(result$counts[["gradient"]]),
    message = ifelse(is.null(result$message), "", result$message),
    stringsAsFactors = FALSE
  )
  v3_write_csv(
    do.call(rbind, multi_start_rows),
    file.path(output_directory, "clearance_multi_start_results.csv")
  )
}
multi_start <- do.call(rbind, multi_start_rows)
evaluation_table <- read.csv(evaluation_path, stringsAsFactors = FALSE)
converged <- multi_start[
  multi_start$convergence_code == 0 &
    is.finite(multi_start$objective) &
    multi_start$objective < penalty_objective,
]
converged <- converged[order(converged$objective, converged$start_index), ]

write_clearance_failure <- function(status) {
  manifest <- list(
    schema_version = "1.0.0",
    protocol_version = config$protocol_version,
    generated_at = v3_utc_now(),
    config_sha256 = config_sha256,
    phase = "identify",
    run_completed = FALSE,
    scenario_count = length(simulations),
    loaded_scenarios = as.list(names(simulations)),
    validation_package_sha256_locked = config$locked_validation_package_sha256,
    validation_package_loaded = FALSE,
    validation_scenarios_loaded = FALSE,
    identification = list(
      clearance = list(
        start_count = length(start_z),
        converged_start_count = nrow(converged),
        successful_evaluation_count = sum(evaluation_table$status == "ok"),
        checks = list(any_successful_converged_start = FALSE),
        accepted = FALSE
      ),
      absorption_set = list(checks = list(not_started = FALSE), accepted = FALSE),
      accepted = FALSE
    ),
    validation = NULL,
    uncertainty = list(successful_sample_count = 0),
    parameter_identifiability = FALSE,
    prediction_identifiability = FALSE,
    scientific_status = status,
    registration_allowed = FALSE,
    artifacts = v3_artifact_inventory(
      output_directory,
      excluded = "identification_manifest.json"
    )
  )
  v3_write_json(
    manifest,
    file.path(output_directory, "identification_manifest.json")
  )
}

if (nrow(converged) == 0) {
  write_clearance_failure("calibration_runtime_failed")
  quit(status = 2)
}

best_row <- converged[1, ]
best_z <- best_row$fitted_z_total_clearance[[1]]
best_clearance <- theta_from_z(best_z)
thresholds <- config$thresholds
top_count <- as.integer(thresholds$top_run_count)
enough_converged <- nrow(converged) >=
  as.integer(thresholds$minimum_converged_starts)
top_objective_reproducible <- FALSE
top_parameter_reproducible <- FALSE
if (nrow(converged) >= top_count) {
  top <- converged[seq_len(top_count), ]
  relative_objective_difference <- max(
    abs(top$objective - top$objective[[1]]) /
      max(abs(top$objective[[1]]), 1e-12)
  )
  top_z_range <- max(top$fitted_z_total_clearance) -
    min(top$fitted_z_total_clearance)
  top_objective_reproducible <- relative_objective_difference <=
    as.numeric(thresholds$top_objective_relative_tolerance)
  top_parameter_reproducible <- top_z_range <=
    as.numeric(thresholds$top_parameter_normalized_range_max)
}
margin <- as.numeric(thresholds$boundary_margin_fraction)
not_near_bound <- best_z > margin && best_z < 1 - margin

hessian_step <- 1e-4
hessian_z <- if (not_near_bound) {
  (
    evaluate_clearance(
      best_z + hessian_step,
      0,
      "hessian",
      TRUE
    )$objective -
      2 * evaluate_clearance(best_z, 0, "hessian", TRUE)$objective +
      evaluate_clearance(
        best_z - hessian_step,
        0,
        "hessian",
        TRUE
      )$objective
  ) / hessian_step^2
} else {
  NA_real_
}
hessian_identifiable <- is.finite(hessian_z) && hessian_z > 0

repeat_first <- evaluate_clearance(
  best_z,
  0,
  "repeatability_1",
  use_cache = FALSE,
  retain_curve = TRUE
)
repeat_second <- evaluate_clearance(
  best_z,
  0,
  "repeatability_2",
  use_cache = FALSE,
  retain_curve = TRUE
)
objective_repeatable <- v3_nearly_equal(
  repeat_first$objective,
  repeat_second$objective,
  as.numeric(thresholds$repeatability_relative_tolerance)
)
prediction_repeatable <- objective_repeatable &&
  repeat_first$status == "ok" && repeat_second$status == "ok" &&
  identical(repeat_first$curve$time_min, repeat_second$curve$time_min) &&
  all(
    abs(
      repeat_first$curve$concentration_umol_per_l -
        repeat_second$curve$concentration_umol_per_l
    ) <= as.numeric(thresholds$repeatability_relative_tolerance) *
      pmax(1, abs(repeat_first$curve$concentration_umol_per_l))
  )

best_curve <- repeat_first$curve
predicted <- v3_interpolate_curve(best_curve, iv_observations$time_min)
ratios <- ifelse(
  iv_observations$observed_umol_per_l > 0,
  predicted / iv_observations$observed_umol_per_l,
  NA_real_
)
signed_logs <- ifelse(
  iv_observations$observed_umol_per_l > 0 & predicted > 0,
  log(ratios),
  NA_real_
)
point_table <- data.frame(
  scenario_name = "CAL_BLANCHARD_IV",
  dataset_id = iv_observations$dataset_id,
  split = iv_observations$split,
  observation_id = iv_observations$observation_id,
  time_min = iv_observations$time_min,
  observed_umol_per_l = iv_observations$observed_umol_per_l,
  predicted_umol_per_l = predicted,
  predicted_to_observed_ratio = ratios,
  signed_log_residual = signed_logs,
  included = iv_observations$included,
  exclusion_reason = iv_observations$exclusion_reason,
  source_artifact_id = iv_observations$source_artifact_id,
  source_location = iv_observations$source_location,
  stringsAsFactors = FALSE
)
positive <- point_table[point_table$included & point_table$observed_umol_per_l > 0, ]
metric_gmfe <- exp(mean(abs(positive$signed_log_residual)))
metric_bias <- exp(mean(positive$signed_log_residual))
metric_two_fold <- mean(
  positive$predicted_to_observed_ratio >= 0.5 &
    positive$predicted_to_observed_ratio <= 2
)
clearance_curve_accepted <-
  all(is.finite(point_table$predicted_umol_per_l)) &&
  all(point_table$predicted_umol_per_l >= 0) &&
  metric_gmfe <= as.numeric(thresholds$gmfe_max) &&
  metric_bias >= as.numeric(thresholds$geometric_bias_min) &&
  metric_bias <= as.numeric(thresholds$geometric_bias_max) &&
  metric_two_fold >= as.numeric(thresholds$minimum_two_fold_fraction)
clearance_metrics <- data.frame(
  scenario_name = "CAL_BLANCHARD_IV",
  dataset_id = iv_observations$dataset_id[[1]],
  split = "calibration",
  included_count = nrow(positive),
  positive_count = nrow(positive),
  gmfe = metric_gmfe,
  geometric_bias = metric_bias,
  log_rmse = sqrt(mean(positive$signed_log_residual^2)),
  two_fold_fraction = metric_two_fold,
  zero_baseline_pass = TRUE,
  finite_nonnegative = TRUE,
  accepted = clearance_curve_accepted,
  stringsAsFactors = FALSE
)
v3_write_csv(best_curve, file.path(output_directory, "clearance_curve.csv"))
v3_write_csv(
  point_table,
  file.path(output_directory, "clearance_point_predictions.csv")
)
v3_write_csv(
  clearance_metrics,
  file.path(output_directory, "clearance_metrics.csv")
)

profile_observation_count <- as.integer(thresholds$disposition_observation_count)
profile_parameter_count <- as.integer(thresholds$disposition_parameter_count)
profile_df <- profile_observation_count - profile_parameter_count
profile_threshold <- repeat_first$objective * (
  1 + qf(
    as.numeric(thresholds$profile_confidence_level),
    df1 = 1,
    df2 = profile_df
  ) / profile_df
)
profile_grid <- seq(
  0,
  1,
  length.out = as.integer(thresholds$profile_grid_points)
)
profile_grid[[which.min(abs(profile_grid - best_z))]] <- best_z
profile_grid <- sort(profile_grid)
profile_table <- do.call(rbind, lapply(seq_along(profile_grid), function(index) {
  z <- profile_grid[[index]]
  result <- evaluate_clearance(z, 0, "profile", TRUE)
  data.frame(
    parameter_id = clearance_id,
    grid_index = index,
    fixed_z = z,
    fixed_parameter_value = theta_from_z(z),
    objective = result$objective,
    threshold = profile_threshold,
    status = result$status,
    stringsAsFactors = FALSE
  )
}))

profile_crossing <- function(rows, side) {
  selected <- if (side == "left") {
    rows[rows$fixed_z <= best_z, ]
  } else {
    rows[rows$fixed_z >= best_z, ]
  }
  selected <- selected[order(selected$fixed_z), ]
  if (side == "left") {
    selected <- selected[nrow(selected):1, ]
  }
  if (nrow(selected) < 2) {
    return(NA_real_)
  }
  delta <- selected$objective - profile_threshold
  for (index in seq_len(nrow(selected) - 1)) {
    if (delta[[index]] == 0) {
      return(selected$fixed_z[[index]])
    }
    if (delta[[index]] * delta[[index + 1]] <= 0) {
      left <- selected[index, ]
      right <- selected[index + 1, ]
      if (right$objective == left$objective) {
        return(mean(c(left$fixed_z, right$fixed_z)))
      }
      return(
        left$fixed_z +
          (profile_threshold - left$objective) *
            (right$fixed_z - left$fixed_z) /
            (right$objective - left$objective)
      )
    }
  }
  NA_real_
}
left_crossing <- profile_crossing(profile_table, "left")
right_crossing <- profile_crossing(profile_table, "right")
profile_accepted <- all(is.finite(profile_table$objective)) &&
  all(profile_table$status %in% c("ok", "cached")) &&
  is.finite(left_crossing) && is.finite(right_crossing) &&
  0 < left_crossing && left_crossing < best_z &&
  best_z < right_crossing && right_crossing < 1
v3_write_csv(
  profile_table,
  file.path(output_directory, "clearance_profile_likelihood.csv")
)
v3_write_json(
  list(
    method = "31-point one-parameter normalized profile likelihood",
    threshold_method = "RSS_min*(1+F_0.95(1,n-p)/(n-p))",
    observation_count = profile_observation_count,
    parameter_count = profile_parameter_count,
    threshold = profile_threshold,
    best_z = best_z,
    best_value = best_clearance,
    left_crossing_z = left_crossing,
    right_crossing_z = right_crossing,
    left_crossing_value = ifelse(
      is.finite(left_crossing),
      theta_from_z(left_crossing),
      NA_real_
    ),
    right_crossing_value = ifelse(
      is.finite(right_crossing),
      theta_from_z(right_crossing),
      NA_real_
    ),
    accepted = profile_accepted
  ),
  file.path(output_directory, "clearance_profile_summary.json")
)

clearance_checks <- list(
  at_least_five_converged = enough_converged,
  top_five_objective_reproducible = top_objective_reproducible,
  top_five_clearance_reproducible = top_parameter_reproducible,
  optimum_not_near_bounds = not_near_bound,
  one_dimensional_hessian_finite_positive = hessian_identifiable,
  bilateral_profile_likelihood_interval_finite = profile_accepted,
  blanchard_iv_meets_curve_standard = clearance_curve_accepted,
  objective_repeatable = objective_repeatable,
  prediction_repeatable = prediction_repeatable,
  validation_package_not_loaded = TRUE,
  validation_scenarios_not_loaded = identical(names(simulations), calibration_names)
)
clearance_accepted <- all(unlist(clearance_checks))

v3_write_csv(
  data.frame(
    parameter_id = clearance_id,
    fitted_value = best_clearance,
    normalized_value = best_z,
    unit = clearance_spec$unit,
    lower_bound = clearance_lower,
    upper_bound = clearance_upper,
    stringsAsFactors = FALSE
  ),
  file.path(output_directory, "best_clearance_parameter.csv")
)
v3_write_csv(
  data.frame(
    parameter_id = clearance_id,
    normalized_hessian_value = hessian_z,
    stringsAsFactors = FALSE
  ),
  file.path(output_directory, "clearance_hessian_normalized.csv")
)
clearance_variance_z <- if (clearance_accepted) {
  (2 * repeat_first$objective / profile_df) / hessian_z
} else {
  NA_real_
}
clearance_variance_original <- clearance_variance_z * clearance_span^2
v3_write_csv(
  data.frame(
    parameter_id = clearance_id,
    variance_normalized = clearance_variance_z,
    variance_original_units = clearance_variance_original,
    standard_error_original_units = sqrt(clearance_variance_original),
    stringsAsFactors = FALSE
  ),
  file.path(output_directory, "clearance_covariance.csv")
)

if (!clearance_accepted) {
  status <- if (!hessian_identifiable || !profile_accepted) {
    "clearance_unidentifiable"
  } else {
    "calibration_failed"
  }
  manifest <- list(
    schema_version = "1.0.0",
    protocol_version = config$protocol_version,
    generated_at = v3_utc_now(),
    config_sha256 = config_sha256,
    phase = "identify",
    run_completed = TRUE,
    scenario_count = length(simulations),
    loaded_scenarios = as.list(names(simulations)),
    validation_package_sha256_locked = config$locked_validation_package_sha256,
    validation_package_loaded = FALSE,
    validation_scenarios_loaded = FALSE,
    identification = list(
      clearance = list(
        start_count = length(start_z),
        converged_start_count = nrow(converged),
        successful_evaluation_count = sum(evaluation_table$status == "ok"),
        best_objective = repeat_first$objective,
        best_total_cyp1a2_specific_clearance = best_clearance,
        normalized_hessian = hessian_z,
        profile_threshold = profile_threshold,
        checks = clearance_checks,
        accepted = FALSE
      ),
      absorption_set = list(checks = list(not_started = FALSE), accepted = FALSE),
      accepted = FALSE
    ),
    validation = NULL,
    uncertainty = list(successful_sample_count = 0),
    parameter_identifiability = FALSE,
    prediction_identifiability = FALSE,
    scientific_status = status,
    registration_allowed = FALSE,
    artifacts = v3_artifact_inventory(
      output_directory,
      excluded = "identification_manifest.json"
    )
  )
  v3_write_json(
    manifest,
    file.path(output_directory, "identification_manifest.json")
  )
  quit(status = 2)
}

candidate_specs <- config$permeability_candidates
candidate_ids <- vapply(candidate_specs, function(x) as.integer(x$candidate_id), integer(1))
candidate_values <- vapply(
  candidate_specs,
  function(x) as.numeric(x$value_dm_per_min),
  numeric(1)
)
candidate_logs <- vapply(
  candidate_specs,
  function(x) as.numeric(x$log10_dm_per_min),
  numeric(1)
)
if (
  !identical(candidate_ids, seq_len(71)) ||
    !v3_nearly_equal(candidate_values[[1]], 1e-9, 1e-12) ||
    !v3_nearly_equal(candidate_values[[71]], 1e-2, 1e-12) ||
    any(abs(diff(candidate_logs) - 0.1) > 1e-12)
) {
  stop("Locked permeability candidate grid drifted.")
}
absorption_protocol <- config$absorption_set_protocol
po_observations <- observations[["CAL_BLANCHARD_PO"]]
curve_root <- file.path(output_directory, "absorption_curves")
dir.create(curve_root, recursive = TRUE, showWarnings = FALSE)

run_absorption_grid <- function(repeat_id) {
  repeat_directory <- file.path(curve_root, paste0("repeat_", repeat_id))
  dir.create(repeat_directory, recursive = TRUE, showWarnings = FALSE)
  rows <- list()
  aggregate_curves <- list()
  curves_by_candidate <- list()
  for (index in seq_along(candidate_ids)) {
    candidate_id <- candidate_ids[[index]]
    permeability <- candidate_values[[index]]
    logical_values <- c(
      total_cyp1a2_specific_clearance = best_clearance,
      specific_intestinal_permeability_transcellular = permeability
    )
    evaluated <- tryCatch({
      curves <- run_parameter_set(make_batches(calibration_names), logical_values)
      auc_iv <- v3_trapezoid_auc_0_24(curves[["CAL_BLANCHARD_IV"]])
      auc_po <- v3_trapezoid_auc_0_24(curves[["CAL_BLANCHARD_PO"]])
      if (!is.finite(auc_iv) || auc_iv <= 0 || !is.finite(auc_po)) {
        stop("Candidate AUC is invalid.")
      }
      tmax_index <- which.max(
        curves[["CAL_BLANCHARD_PO"]]$concentration_umol_per_l
      )
      tmax_po_min <- curves[["CAL_BLANCHARD_PO"]]$time_min[[tmax_index]]
      combined <- do.call(rbind, lapply(calibration_names, function(name) {
        data.frame(
          candidate_id = candidate_id,
          permeability_dm_per_min = permeability,
          scenario_name = name,
          time_min = curves[[name]]$time_min,
          concentration_umol_per_l =
            curves[[name]]$concentration_umol_per_l,
          stringsAsFactors = FALSE
        )
      }))
      curve_path <- file.path(
        repeat_directory,
        sprintf("candidate_%03d.csv", candidate_id)
      )
      v3_write_csv(combined, curve_path)
      aggregate <- combined
      aggregate$repeat_id <- repeat_id
      list(
        status = "ok",
        error = "",
        curves = curves,
        combined = aggregate,
        auc_iv = auc_iv,
        auc_po = auc_po,
        f_model = auc_po / auc_iv,
        tmax_po_min = tmax_po_min,
        curve_path = v3_relative_path(curve_path, output_directory),
        curve_sha256 = v3_sha256_file(curve_path)
      )
    }, error = function(error) {
      list(
        status = "failed",
        error = conditionMessage(error),
        curves = NULL,
        combined = NULL,
        auc_iv = NA_real_,
        auc_po = NA_real_,
        f_model = NA_real_,
        tmax_po_min = NA_real_,
        curve_path = "",
        curve_sha256 = ""
      )
    })
    curves_by_candidate[[as.character(candidate_id)]] <- evaluated$curves
    if (!is.null(evaluated$combined)) {
      aggregate_curves[[length(aggregate_curves) + 1]] <- evaluated$combined
    }
    absolute_accepted <- evaluated$status == "ok" &&
      evaluated$f_model >= as.numeric(absorption_protocol$bioavailability_lower) &&
      evaluated$f_model <= as.numeric(absorption_protocol$bioavailability_upper) &&
      evaluated$tmax_po_min >= as.numeric(absorption_protocol$tmax_lower_min) &&
      evaluated$tmax_po_min <= as.numeric(absorption_protocol$tmax_upper_min)
    rows[[index]] <- data.frame(
      repeat_id = repeat_id,
      candidate_id = candidate_id,
      log10_permeability_dm_per_min = candidate_logs[[index]],
      permeability_dm_per_min = permeability,
      auc_iv_0_24 = evaluated$auc_iv,
      auc_po_0_24 = evaluated$auc_po,
      f_model = evaluated$f_model,
      tmax_po_min = evaluated$tmax_po_min,
      absolute_constraints_accepted = absolute_accepted,
      in_admissible_set = FALSE,
      status = evaluated$status,
      error = evaluated$error,
      curve_path = evaluated$curve_path,
      curve_sha256 = evaluated$curve_sha256,
      stringsAsFactors = FALSE
    )
  }
  list(
    rows = do.call(rbind, rows),
    aggregate_curves = do.call(rbind, aggregate_curves),
    curves = curves_by_candidate
  )
}

build_plateau_comparisons <- function(grid_result, repeat_id) {
  rows <- list()
  for (lower_id in seq_len(61)) {
    upper_id <- lower_id + 10
    lower_row <- grid_result$rows[
      grid_result$rows$candidate_id == lower_id,
    ]
    upper_row <- grid_result$rows[
      grid_result$rows$candidate_id == upper_id,
    ]
    lower_curve <- grid_result$curves[[as.character(lower_id)]][[
      "CAL_BLANCHARD_PO"
    ]]
    upper_curve <- grid_result$curves[[as.character(upper_id)]][[
      "CAL_BLANCHARD_PO"
    ]]
    valid <- nrow(lower_row) == 1 && nrow(upper_row) == 1 &&
      lower_row$status == "ok" && upper_row$status == "ok" &&
      !is.null(lower_curve) && !is.null(upper_curve)
    if (valid) {
      lower_predictions <- v3_interpolate_curve(
        lower_curve,
        po_observations$time_min
      )
      upper_predictions <- v3_interpolate_curve(
        upper_curve,
        po_observations$time_min
      )
      delta_f <- abs(lower_row$f_model - upper_row$f_model)
      delta_tmax_h <- abs(
        lower_row$tmax_po_min - upper_row$tmax_po_min
      ) / 60
      max_curve_delta <- max(v3_symmetric_relative_delta(
        lower_predictions,
        upper_predictions
      ))
    } else {
      delta_f <- NA_real_
      delta_tmax_h <- NA_real_
      max_curve_delta <- NA_real_
    }
    accepted <- valid &&
      delta_f <= as.numeric(
        absorption_protocol$plateau_bioavailability_delta_max
      ) &&
      delta_tmax_h <= as.numeric(
        absorption_protocol$plateau_tmax_delta_h_max
      ) &&
      max_curve_delta <= as.numeric(
        absorption_protocol$plateau_concentration_symmetric_relative_delta_max
      )
    rows[[lower_id]] <- data.frame(
      repeat_id = repeat_id,
      lower_candidate_id = lower_id,
      tenfold_higher_candidate_id = upper_id,
      lower_permeability_dm_per_min = candidate_values[[lower_id]],
      tenfold_higher_permeability_dm_per_min = candidate_values[[upper_id]],
      absolute_delta_f_model = delta_f,
      absolute_delta_tmax_h = delta_tmax_h,
      max_symmetric_relative_concentration_delta = max_curve_delta,
      accepted = accepted,
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}

derive_admissible_set <- function(candidate_rows, comparison_rows) {
  possible <- integer()
  for (start_id in seq_len(61)) {
    absolute_ok <- all(candidate_rows$absolute_constraints_accepted[
      candidate_rows$candidate_id >= start_id
    ])
    plateau_ok <- all(comparison_rows$accepted[
      comparison_rows$lower_candidate_id >= start_id
    ])
    if (absolute_ok && plateau_ok) {
      possible <- c(possible, start_id)
    }
  }
  if (length(possible) == 0) {
    return(integer())
  }
  seq(min(possible), 71)
}

grid_first <- run_absorption_grid(1)
comparisons_first <- build_plateau_comparisons(grid_first, 1)
allowed_first <- derive_admissible_set(grid_first$rows, comparisons_first)
grid_first$rows$in_admissible_set <-
  grid_first$rows$candidate_id %in% allowed_first
grid_second <- run_absorption_grid(2)
comparisons_second <- build_plateau_comparisons(grid_second, 2)
allowed_second <- derive_admissible_set(grid_second$rows, comparisons_second)
grid_second$rows$in_admissible_set <-
  grid_second$rows$candidate_id %in% allowed_second

candidate_table <- rbind(grid_first$rows, grid_second$rows)
comparison_table <- rbind(comparisons_first, comparisons_second)
curve_table <- rbind(grid_first$aggregate_curves, grid_second$aggregate_curves)
v3_write_csv(
  candidate_table,
  file.path(output_directory, "absorption_candidates.csv")
)
v3_write_csv(
  comparison_table,
  file.path(output_directory, "absorption_plateau_comparisons.csv")
)
v3_write_csv(
  curve_table,
  file.path(output_directory, "absorption_candidate_curves.csv")
)

candidate_metrics_repeatable <- identical(allowed_first, allowed_second) &&
  all(vapply(c(
    "auc_iv_0_24",
    "auc_po_0_24",
    "f_model",
    "tmax_po_min"
  ), function(column) {
    all(v3_nearly_equal(
      grid_first$rows[[column]],
      grid_second$rows[[column]],
      as.numeric(thresholds$repeatability_relative_tolerance)
    ))
  }, logical(1)))
candidate_curves_repeatable <- identical(
  grid_first$aggregate_curves[, setdiff(names(grid_first$aggregate_curves), "repeat_id")],
  grid_second$aggregate_curves[, setdiff(names(grid_second$aggregate_curves), "repeat_id")]
)
curve_checksums_repeatable <- identical(
  grid_first$rows$curve_sha256,
  grid_second$rows$curve_sha256
) && identical(
  basename(grid_first$rows$curve_path),
  basename(grid_second$rows$curve_path)
)
comparison_repeatable <- identical(
  comparisons_first[, setdiff(names(comparisons_first), "repeat_id")],
  comparisons_second[, setdiff(names(comparisons_second), "repeat_id")]
)
set_nonempty <- length(allowed_first) > 0
set_contiguous <- set_nonempty && identical(
  allowed_first,
  seq(min(allowed_first), max(allowed_first))
)
set_reaches_upper_guard <- set_nonempty && max(allowed_first) == 71
upper_guard_plateau <- comparisons_first$accepted[
  comparisons_first$lower_candidate_id == 61
]
absorption_checks <- list(
  all_seventy_one_candidates_executed_twice =
    nrow(candidate_table) == 142 && all(candidate_table$status == "ok"),
  all_sixty_one_tenfold_comparisons_executed_twice =
    nrow(comparison_table) == 122,
  admissible_set_nonempty = set_nonempty,
  admissible_set_contiguous = set_contiguous,
  admissible_set_reaches_upper_numerical_guard = set_reaches_upper_guard,
  upper_guard_has_tenfold_plateau_evidence =
    length(upper_guard_plateau) == 1 && isTRUE(upper_guard_plateau),
  admissible_set_repeatable = identical(allowed_first, allowed_second),
  candidate_metrics_repeatable = candidate_metrics_repeatable,
  candidate_curves_repeatable = candidate_curves_repeatable,
  candidate_curve_checksums_repeatable = curve_checksums_repeatable,
  plateau_comparisons_repeatable = comparison_repeatable,
  validation_package_not_loaded = TRUE,
  validation_scenarios_not_loaded = identical(names(simulations), calibration_names)
)
absorption_accepted <- all(unlist(absorption_checks))
reference_candidate_id <- if (set_nonempty) min(allowed_first) else NA_integer_
reference_permeability <- if (set_nonempty) {
  candidate_values[[reference_candidate_id]]
} else {
  NA_real_
}

if (set_nonempty) {
  reference_curves <- grid_first$curves[[as.character(reference_candidate_id)]]
  po_curve <- reference_curves[["CAL_BLANCHARD_PO"]]
  po_predicted <- v3_interpolate_curve(po_curve, po_observations$time_min)
  po_ratio <- po_predicted / po_observations$observed_umol_per_l
  po_diagnostic_points <- data.frame(
    scenario_name = "CAL_BLANCHARD_PO",
    dataset_id = po_observations$dataset_id,
    observation_id = po_observations$observation_id,
    time_min = po_observations$time_min,
    observed_umol_per_l = po_observations$observed_umol_per_l,
    predicted_umol_per_l = po_predicted,
    predicted_to_observed_ratio = po_ratio,
    signed_log_residual = log(po_ratio),
    source_artifact_id = po_observations$source_artifact_id,
    source_location = po_observations$source_location,
    diagnostic_only = TRUE,
    stringsAsFactors = FALSE
  )
  v3_write_csv(
    po_diagnostic_points,
    file.path(output_directory, "po_calibration_diagnostic_points.csv")
  )
  v3_write_csv(
    data.frame(
      dataset_id = po_observations$dataset_id[[1]],
      reference_candidate_id = reference_candidate_id,
      reference_permeability_dm_per_min = reference_permeability,
      gmfe = exp(mean(abs(log(po_ratio)))),
      geometric_bias = exp(mean(log(po_ratio))),
      log_rmse = sqrt(mean(log(po_ratio)^2)),
      two_fold_fraction = mean(po_ratio >= 0.5 & po_ratio <= 2),
      diagnostic_only = TRUE,
      included_in_clearance_objective = FALSE,
      included_in_external_validation = FALSE,
      stringsAsFactors = FALSE
    ),
    file.path(output_directory, "po_calibration_diagnostic_metrics.csv")
  )
}

set_manifest <- list(
  schema_version = "1.0.0",
  protocol_version = config$protocol_version,
  generated_at = v3_utc_now(),
  config_sha256 = config_sha256,
  identification_type = "set_identified_nuisance_parameter",
  parameter_id = permeability_id,
  search_guard_lower_dm_per_min = candidate_values[[1]],
  search_guard_upper_dm_per_min = candidate_values[[71]],
  lower_bound_dm_per_min = reference_permeability,
  upper_bound_interpretation = "right_censored_numerical_guard",
  reference_selection = "minimum_admissible_candidate",
  reference_candidate_id = reference_candidate_id,
  admissible_candidate_ids = as.list(allowed_first),
  admissible_candidate_count = length(allowed_first),
  candidate_table_sha256 = v3_sha256_file(
    file.path(output_directory, "absorption_candidates.csv")
  ),
  comparison_table_sha256 = v3_sha256_file(
    file.path(output_directory, "absorption_plateau_comparisons.csv")
  ),
  aggregate_curve_table_sha256 = v3_sha256_file(
    file.path(output_directory, "absorption_candidate_curves.csv")
  ),
  repeatability = list(
    candidate_metrics_identical = candidate_metrics_repeatable,
    candidate_curves_identical = candidate_curves_repeatable,
    candidate_curve_checksums_identical = curve_checksums_repeatable,
    comparisons_identical = comparison_repeatable,
    admissible_set_identical = identical(allowed_first, allowed_second)
  ),
  checks = absorption_checks,
  accepted = absorption_accepted
)
v3_write_json(
  set_manifest,
  file.path(output_directory, "absorption_set_manifest.json")
)

identification_accepted <- clearance_accepted && absorption_accepted
scientific_status <- if (identification_accepted) {
  "identification_accepted_validation_locked"
} else {
  "absorption_structure_failed"
}
manifest <- list(
  schema_version = "1.0.0",
  protocol_version = config$protocol_version,
  generated_at = v3_utc_now(),
  config_sha256 = config_sha256,
  phase = "identify",
  run_completed = TRUE,
  scenario_count = length(simulations),
  loaded_scenarios = as.list(names(simulations)),
  validation_package_sha256_locked = config$locked_validation_package_sha256,
  validation_package_loaded = FALSE,
  validation_scenarios_loaded = FALSE,
  identification = list(
    clearance = list(
      start_count = length(start_z),
      converged_start_count = nrow(converged),
      successful_evaluation_count = sum(evaluation_table$status == "ok"),
      best_objective = repeat_first$objective,
      best_total_cyp1a2_specific_clearance = best_clearance,
      normalized_hessian = hessian_z,
      clearance_variance_original_units = clearance_variance_original,
      profile_threshold = profile_threshold,
      checks = clearance_checks,
      accepted = clearance_accepted
    ),
    absorption_set = list(
      reference_candidate_id = reference_candidate_id,
      reference_permeability_dm_per_min = reference_permeability,
      admissible_candidate_count = length(allowed_first),
      admissible_candidate_ids = as.list(allowed_first),
      upper_bound_interpretation = "right_censored_numerical_guard",
      checks = absorption_checks,
      accepted = absorption_accepted
    ),
    accepted = identification_accepted
  ),
  validation = NULL,
  uncertainty = list(successful_sample_count = 0),
  parameter_identifiability = FALSE,
  prediction_identifiability = FALSE,
  scientific_status = scientific_status,
  registration_allowed = FALSE,
  artifacts = v3_artifact_inventory(
    output_directory,
    excluded = "identification_manifest.json"
  )
)
v3_write_json(
  manifest,
  file.path(output_directory, "identification_manifest.json")
)
quit(status = ifelse(identification_accepted, 0, 2))
