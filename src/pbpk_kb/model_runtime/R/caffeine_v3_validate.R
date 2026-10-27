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
if (length(args) != 2 || args[[2]] != "validate") {
  stop("Usage: caffeine_v3_validate.R <identification-config.json> validate")
}
config_path <- normalizePath(args[[1]], mustWork = TRUE)
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

root_output <- normalizePath(config$output_directory, mustWork = TRUE)
identification_directory <- file.path(root_output, "01_identification")
identification_manifest_path <- file.path(
  identification_directory,
  "identification_manifest.json"
)
best_clearance_path <- file.path(
  identification_directory,
  "best_clearance_parameter.csv"
)
clearance_covariance_path <- file.path(
  identification_directory,
  "clearance_covariance.csv"
)
set_manifest_path <- file.path(
  identification_directory,
  "absorption_set_manifest.json"
)
candidate_table_path <- file.path(
  identification_directory,
  "absorption_candidates.csv"
)
for (required_path in c(
  identification_manifest_path,
  best_clearance_path,
  clearance_covariance_path,
  set_manifest_path,
  candidate_table_path
)) {
  if (!file.exists(required_path)) {
    stop(paste("Accepted identification artifact is missing:", required_path))
  }
}

identification_manifest <- fromJSON(
  identification_manifest_path,
  simplifyVector = FALSE
)
identification_checks <- c(
  unlist(identification_manifest$identification$clearance$checks),
  unlist(identification_manifest$identification$absorption_set$checks)
)
if (
  identification_manifest$config_sha256 != config_sha256 ||
    identification_manifest$phase != "identify" ||
    !isTRUE(identification_manifest$run_completed) ||
    identification_manifest$scientific_status !=
      "identification_accepted_validation_locked" ||
    !isTRUE(identification_manifest$identification$accepted) ||
    length(identification_checks) == 0 ||
    !all(identification_checks) ||
    isTRUE(identification_manifest$validation_package_loaded) ||
    isTRUE(identification_manifest$validation_scenarios_loaded) ||
    isTRUE(identification_manifest$registration_allowed)
) {
  stop("Identification manifest does not unlock validation.")
}

set_manifest <- fromJSON(set_manifest_path, simplifyVector = FALSE)
set_checks <- unlist(set_manifest$checks)
if (
  !isTRUE(set_manifest$accepted) ||
    set_manifest$identification_type != "set_identified_nuisance_parameter" ||
    set_manifest$upper_bound_interpretation !=
      "right_censored_numerical_guard" ||
    length(set_checks) == 0 ||
    !all(set_checks) ||
    set_manifest$candidate_table_sha256 != v3_sha256_file(candidate_table_path)
) {
  stop("Absorption set manifest failed its locked contract.")
}
admissible_ids <- as.integer(unlist(set_manifest$admissible_candidate_ids))
if (
  length(admissible_ids) == 0 ||
    !identical(admissible_ids, seq(min(admissible_ids), 71)) ||
    as.integer(set_manifest$reference_candidate_id) != min(admissible_ids)
) {
  stop("Admissible permeability set is empty, non-contiguous, or drifted.")
}

parameters <- config$parameters
names(parameters) <- vapply(parameters, function(x) x$parameter_id, character(1))
clearance_id <- "total_cyp1a2_specific_clearance"
permeability_id <- "specific_intestinal_permeability_transcellular"
best_clearance_table <- read.csv(best_clearance_path, stringsAsFactors = FALSE)
if (
  nrow(best_clearance_table) != 1 ||
    best_clearance_table$parameter_id[[1]] != clearance_id
) {
  stop("Best clearance artifact identity drifted.")
}
best_clearance <- as.numeric(best_clearance_table$fitted_value[[1]])
clearance_lower <- as.numeric(parameters[[clearance_id]]$lower_bound)
clearance_upper <- as.numeric(parameters[[clearance_id]]$upper_bound)
if (
  !is.finite(best_clearance) ||
    best_clearance <= clearance_lower || best_clearance >= clearance_upper
) {
  stop("Best clearance is outside strict locked bounds.")
}
clearance_covariance <- read.csv(
  clearance_covariance_path,
  stringsAsFactors = FALSE
)
if (
  nrow(clearance_covariance) != 1 ||
    clearance_covariance$parameter_id[[1]] != clearance_id ||
    !is.finite(clearance_covariance$variance_original_units[[1]]) ||
    clearance_covariance$variance_original_units[[1]] <= 0
) {
  stop("Clearance covariance is not finite and positive.")
}

candidate_specs <- config$permeability_candidates
candidate_ids <- vapply(candidate_specs, function(x) as.integer(x$candidate_id), integer(1))
candidate_values <- vapply(
  candidate_specs,
  function(x) as.numeric(x$value_dm_per_min),
  numeric(1)
)
if (!identical(candidate_ids, seq_len(71))) {
  stop("Permeability grid identity drifted.")
}
admissible_values <- candidate_values[admissible_ids]
if (
  any(!is.finite(admissible_values)) ||
    any(admissible_values <= 0) ||
    !isTRUE(all.equal(
      min(admissible_values),
      as.numeric(set_manifest$lower_bound_dm_per_min),
      tolerance = 1e-14
    ))
) {
  stop("Admissible permeability values drifted.")
}

# The validation package is read only after all identification gates above pass.
validation_package_path <- normalizePath(
  config$locked_validation_package_path,
  mustWork = TRUE
)
if (v3_sha256_file(validation_package_path) !=
  config$locked_validation_package_sha256) {
  stop("Locked validation package SHA-256 drifted.")
}
validation_package <- fromJSON(
  validation_package_path,
  simplifyVector = FALSE
)
package_checks <- unlist(validation_package$checks)
if (
  validation_package$checksum != config$locked_validation_package_checksum ||
    !isTRUE(validation_package$accepted) ||
    length(package_checks) == 0 ||
    !all(package_checks)
) {
  stop("Locked validation package failed its contract.")
}
validation_package_loaded_at <- v3_utc_now()

calibration_names <- c("CAL_BLANCHARD_IV", "CAL_BLANCHARD_PO")
validation_names <- c(
  "HOLD_WAHLLANDER_IV",
  "EXT_EUDRACT_PO",
  "EXT_PMC535_PO",
  "EXT2_PMC572_PO_MALE",
  "EXT2_PMC572_PO_FEMALE"
)
all_expected_names <- c(calibration_names, validation_names)
scenarios <- c(config$calibration_scenarios, validation_package$scenarios)
names(scenarios) <- vapply(scenarios, function(x) x$name, character(1))
if (!identical(names(scenarios), all_expected_names)) {
  stop("Unlocked scenario order or identity drifted.")
}
if (sum(vapply(scenarios, function(x) length(x$observations), integer(1))) != 50) {
  stop("Unlocked observation count drifted.")
}

mapping_rows <- v3_mapping_rows(parameters)
mapping_paths <- vapply(mapping_rows, function(x) x$runtime_path, character(1))
if (length(mapping_rows) != 10 || anyDuplicated(mapping_paths)) {
  stop("Runtime mapping set drifted.")
}
mapping_by_path <- mapping_rows
names(mapping_by_path) <- mapping_paths
variable_ids <- c(clearance_id, permeability_id)
variable_paths <- names(v3_runtime_values_for(
  c(
    total_cyp1a2_specific_clearance = best_clearance,
    specific_intestinal_permeability_transcellular = min(admissible_values)
  ),
  parameters,
  variable_ids
))
if (length(variable_paths) != 4 || anyDuplicated(variable_paths)) {
  stop("Set-validation variable path set drifted.")
}

target_output <- config$target_output
simulations <- list()
preflight_rows <- list()
for (scenario_name in all_expected_names) {
  scenario <- scenarios[[scenario_name]]
  pkml_path <- file.path(config$runtime_root, scenario$pkml$relative_path)
  if (v3_sha256_file(pkml_path) != scenario$pkml$sha256) {
    stop(paste("Unlocked PKML checksum mismatch:", scenario_name))
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
      expected_base_value = expected_base,
      actual_base_value = actual_base,
      valid = valid,
      stringsAsFactors = FALSE
    )
    if (!valid) {
      stop(paste("Unlocked runtime parameter drifted:", scenario_name))
    }
  }
  simulations[[scenario_name]] <- simulation
}

validation_directory <- file.path(root_output, "02_validation_uncertainty")
progress_manifest_path <- file.path(
  validation_directory,
  "validation_execution_progress.json"
)
run_manifest_path <- file.path(root_output, "run_manifest.json")
validation_manifest_path <- file.path(
  validation_directory,
  "validation_manifest.json"
)
resume_validation <- dir.exists(validation_directory)
if (resume_validation) {
  if (file.exists(run_manifest_path) || file.exists(validation_manifest_path)) {
    stop("Completed immutable validation output already exists.")
  }
  if (!file.exists(progress_manifest_path)) {
    stop("Validation directory has no resumable execution manifest.")
  }
  progress_manifest <- fromJSON(progress_manifest_path, simplifyVector = FALSE)
  if (
    !identical(as.character(progress_manifest$config_sha256), config_sha256) ||
      !identical(
        as.character(progress_manifest$identification_manifest_sha256),
        v3_sha256_file(identification_manifest_path)
      ) ||
      !identical(
        as.character(progress_manifest$absorption_set_manifest_sha256),
        v3_sha256_file(set_manifest_path)
      ) ||
      !identical(
        as.character(progress_manifest$validation_package_sha256),
        v3_sha256_file(validation_package_path)
      ) ||
      !identical(
        as.integer(progress_manifest$uncertainty_seed),
        as.integer(config$uncertainty$seed)
      ) ||
      !identical(
        as.integer(progress_manifest$sample_count),
        as.integer(config$uncertainty$sample_count)
      )
  ) {
    stop("Validation checkpoint identity drifted; resume refused.")
  }
} else {
  dir.create(validation_directory, recursive = TRUE, showWarnings = FALSE)
  v3_write_json(
    list(
      status = "deterministic_set_validation",
      config_sha256 = config_sha256,
      identification_manifest_sha256 =
        v3_sha256_file(identification_manifest_path),
      absorption_set_manifest_sha256 = v3_sha256_file(set_manifest_path),
      validation_package_sha256 = v3_sha256_file(validation_package_path),
      uncertainty_seed = as.integer(config$uncertainty$seed),
      sample_count = as.integer(config$uncertainty$sample_count),
      completed_scenarios = list(),
      timestamp_utc = v3_utc_now()
    ),
    progress_manifest_path
  )
}
v3_write_csv(
  do.call(rbind, preflight_rows),
  file.path(validation_directory, "unlocked_runtime_parameter_preflight.csv")
)

runtime_parameters_by_path <- lapply(
  mapping_paths,
  function(path) getParameter(path, simulations[[all_expected_names[[1]]]])
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

make_batches <- function(simulation_names, parameter_paths) {
  output <- lapply(simulation_names, function(name) {
    createSimulationBatch(
      simulation = simulations[[name]],
      parametersOrPaths = parameter_paths
    )
  })
  names(output) <- simulation_names
  output
}

run_parameter_set <- function(batches, logical_values, logical_ids) {
  values_by_path <- v3_runtime_values_for(
    logical_values,
    parameters,
    logical_ids
  )
  result_ids <- list()
  for (name in names(batches)) {
    values <- batch_value_vector(batches[[name]], values_by_path)
    result_ids[[name]] <- batches[[name]]$addRunValues(parameterValues = values)
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

concentration_metrics <- function(rows) {
  thresholds <- config$thresholds
  included <- rows[rows$included, ]
  positive <- included[included$observed_umol_per_l > 0, ]
  zero <- included[included$observed_umol_per_l == 0, ]
  if (nrow(positive) == 0 || any(positive$predicted_umol_per_l <= 0)) {
    stop("Concentration metric set has no valid positive observations.")
  }
  ratios <- positive$predicted_umol_per_l / positive$observed_umol_per_l
  log_errors <- log(ratios)
  zero_pass <- nrow(zero) == 0 || all(
    zero$predicted_umol_per_l <=
      as.numeric(thresholds$zero_baseline_max_umol_per_l)
  )
  finite_nonnegative <- all(is.finite(included$predicted_umol_per_l)) &&
    all(included$predicted_umol_per_l >= 0)
  gmfe <- exp(mean(abs(log_errors)))
  bias <- exp(mean(log_errors))
  log_rmse <- sqrt(mean(log_errors^2))
  two_fold_fraction <- mean(ratios >= 0.5 & ratios <= 2)
  accepted <- finite_nonnegative && zero_pass &&
    gmfe <= as.numeric(thresholds$gmfe_max) &&
    bias >= as.numeric(thresholds$geometric_bias_min) &&
    bias <= as.numeric(thresholds$geometric_bias_max) &&
    two_fold_fraction >= as.numeric(thresholds$minimum_two_fold_fraction)
  data.frame(
    dataset_id = rows$dataset_id[[1]],
    split = rows$split[[1]],
    included_count = nrow(included),
    positive_count = nrow(positive),
    gmfe = gmfe,
    geometric_bias = bias,
    log_rmse = log_rmse,
    two_fold_fraction = two_fold_fraction,
    zero_baseline_pass = zero_pass,
    finite_nonnegative = finite_nonnegative,
    accepted = accepted,
    stringsAsFactors = FALSE
  )
}

weighted_secondary_curve <- function(curves) {
  male <- curves[["EXT2_PMC572_PO_MALE"]]
  female <- curves[["EXT2_PMC572_PO_FEMALE"]]
  if (!identical(male$time_min, female$time_min)) {
    stop("PMC572 male and female simulation grids differ.")
  }
  data.frame(
    time_min = male$time_min,
    concentration_umol_per_l =
      0.5 * male$concentration_umol_per_l +
      0.5 * female$concentration_umol_per_l
  )
}

secondary_endpoints <- function(curve) {
  molecular_weight <- as.numeric(
    config$uncertainty$caffeine_molecular_weight_g_per_mol
  )
  end_h <- as.numeric(config$uncertainty$pmc572_auc_end_h)
  end_min <- end_h * 60
  if (max(curve$time_min) < end_min) {
    stop("PMC572 simulation does not cover 24 h.")
  }
  within <- curve[curve$time_min <= end_min, ]
  if (tail(within$time_min, 1) < end_min) {
    within <- rbind(
      within,
      data.frame(
        time_min = end_min,
        concentration_umol_per_l = v3_interpolate_curve(curve, end_min)
      )
    )
  }
  concentration_ng_per_ml <-
    within$concentration_umol_per_l * molecular_weight
  auc_0_t <- sum(
    diff(within$time_min / 60) *
      (head(concentration_ng_per_ml, -1) +
        tail(concentration_ng_per_ml, -1)) / 2
  )
  maximum_index <- which.max(concentration_ng_per_ml)
  cmax <- concentration_ng_per_ml[[maximum_index]]
  tmax <- within$time_min[[maximum_index]] / 60
  terminal_times_h <- as.numeric(unlist(
    config$uncertainty$pmc572_terminal_times_h
  ))
  terminal_concentrations <- v3_interpolate_curve(
    curve,
    terminal_times_h * 60
  ) * molecular_weight
  if (any(terminal_concentrations <= 0)) {
    stop("PMC572 terminal concentrations are non-positive.")
  }
  terminal_fit <- lm(log(terminal_concentrations) ~ terminal_times_h)
  lambda_z <- -unname(coef(terminal_fit)[[2]])
  if (!is.finite(lambda_z) || lambda_z <= 0) {
    stop("PMC572 terminal lambda_z is not finite and positive.")
  }
  c24 <- v3_interpolate_curve(curve, end_min) * molecular_weight
  c(
    auc_0_t = auc_0_t,
    auc_0_inf = auc_0_t + c24 / lambda_z,
    cmax = cmax,
    tmax = tmax,
    half_life = log(2) / lambda_z
  )
}

curve_endpoints <- function(curve) {
  end_min <- min(24 * 60, max(curve$time_min))
  within <- curve[curve$time_min <= end_min, ]
  if (tail(within$time_min, 1) < end_min) {
    within <- rbind(
      within,
      data.frame(
        time_min = end_min,
        concentration_umol_per_l = v3_interpolate_curve(curve, end_min)
      )
    )
  }
  c(
    auc_0_24_h_umol_h_per_l = sum(
      diff(within$time_min / 60) *
        (head(within$concentration_umol_per_l, -1) +
          tail(within$concentration_umol_per_l, -1)) / 2
    ),
    cmax_umol_per_l = max(within$concentration_umol_per_l),
    tmax_h = within$time_min[[which.max(
      within$concentration_umol_per_l
    )]] / 60
  )
}

secondary_observed <- validation_package$secondary_metrics
names(secondary_observed) <- vapply(
  secondary_observed,
  function(x) x$endpoint,
  character(1)
)
deterministic_curve_root <- file.path(
  validation_directory,
  "set_validation_curves"
)
dir.create(deterministic_curve_root, recursive = TRUE, showWarnings = FALSE)

run_validation_set <- function(repeat_id) {
  repeat_directory <- file.path(
    deterministic_curve_root,
    paste0("repeat_", repeat_id)
  )
  dir.create(repeat_directory, recursive = TRUE, showWarnings = FALSE)
  candidate_status_rows <- list()
  curve_rows <- list()
  point_rows <- list()
  metric_rows <- list()
  secondary_rows <- list()
  endpoint_rows <- list()
  curve_hash_rows <- list()
  for (index in seq_along(admissible_ids)) {
    candidate_id <- admissible_ids[[index]]
    permeability <- candidate_values[[candidate_id]]
    logical_values <- c(
      total_cyp1a2_specific_clearance = best_clearance,
      specific_intestinal_permeability_transcellular = permeability
    )
    evaluated <- tryCatch({
      curves <- run_parameter_set(
        make_batches(validation_names, variable_paths),
        logical_values,
        variable_ids
      )
      list(status = "ok", error = "", curves = curves)
    }, error = function(error) {
      list(status = "failed", error = conditionMessage(error), curves = NULL)
    })
    candidate_status_rows[[index]] <- data.frame(
      repeat_id = repeat_id,
      candidate_id = candidate_id,
      permeability_dm_per_min = permeability,
      status = evaluated$status,
      error = evaluated$error,
      stringsAsFactors = FALSE
    )
    if (evaluated$status != "ok") {
      for (scenario_name in validation_names) {
        metric_rows[[length(metric_rows) + 1]] <- data.frame(
          repeat_id = repeat_id,
          candidate_id = candidate_id,
          permeability_dm_per_min = permeability,
          scenario_name = scenario_name,
          dataset_id = scenarios[[scenario_name]]$dataset_id,
          split = scenarios[[scenario_name]]$split,
          included_count = 0,
          positive_count = 0,
          gmfe = NA_real_,
          geometric_bias = NA_real_,
          log_rmse = NA_real_,
          two_fold_fraction = NA_real_,
          zero_baseline_pass = FALSE,
          finite_nonnegative = FALSE,
          accepted = FALSE,
          stringsAsFactors = FALSE
        )
      }
      for (endpoint in names(secondary_observed)) {
        spec <- secondary_observed[[endpoint]]
        secondary_rows[[length(secondary_rows) + 1]] <- data.frame(
          repeat_id = repeat_id,
          candidate_id = candidate_id,
          permeability_dm_per_min = permeability,
          endpoint = endpoint,
          predicted_value = NA_real_,
          observed_value = as.numeric(spec$observed_value),
          unit = spec$unit,
          fold_error = NA_real_,
          absolute_error = NA_real_,
          accepted = FALSE,
          metric_id = spec$metric_id,
          source_artifact_id = spec$source_artifact_id,
          source_location = spec$source_location,
          stringsAsFactors = FALSE
        )
      }
      next
    }
    curves <- evaluated$curves
    combined <- do.call(rbind, lapply(validation_names, function(name) {
      data.frame(
        candidate_id = candidate_id,
        permeability_dm_per_min = permeability,
        scenario_name = name,
        time_min = curves[[name]]$time_min,
        concentration_umol_per_l = curves[[name]]$concentration_umol_per_l,
        stringsAsFactors = FALSE
      )
    }))
    curve_path <- file.path(
      repeat_directory,
      sprintf("candidate_%03d.csv", candidate_id)
    )
    v3_write_csv(combined, curve_path)
    curve_hash_rows[[length(curve_hash_rows) + 1]] <- data.frame(
      repeat_id = repeat_id,
      candidate_id = candidate_id,
      relative_path = v3_relative_path(curve_path, validation_directory),
      sha256 = v3_sha256_file(curve_path),
      byte_size = unname(file.info(curve_path)$size),
      stringsAsFactors = FALSE
    )
    combined$repeat_id <- repeat_id
    curve_rows[[length(curve_rows) + 1]] <- combined

    for (scenario_name in validation_names) {
      curve <- curves[[scenario_name]]
      scenario <- scenarios[[scenario_name]]
      observations <- v3_observation_frame(scenario)
      predicted <- v3_interpolate_curve(curve, observations$time_min)
      ratio <- ifelse(
        observations$observed_umol_per_l > 0,
        predicted / observations$observed_umol_per_l,
        NA_real_
      )
      points <- data.frame(
        repeat_id = repeat_id,
        candidate_id = candidate_id,
        permeability_dm_per_min = permeability,
        scenario_name = scenario_name,
        dataset_id = observations$dataset_id,
        split = observations$split,
        observation_id = observations$observation_id,
        time_min = observations$time_min,
        observed_umol_per_l = observations$observed_umol_per_l,
        predicted_umol_per_l = predicted,
        predicted_to_observed_ratio = ratio,
        signed_log_residual = ifelse(
          observations$observed_umol_per_l > 0 & predicted > 0,
          log(ratio),
          NA_real_
        ),
        included = observations$included,
        exclusion_reason = observations$exclusion_reason,
        source_artifact_id = observations$source_artifact_id,
        source_location = observations$source_location,
        stringsAsFactors = FALSE
      )
      point_rows[[length(point_rows) + 1]] <- points
      metrics <- concentration_metrics(points)
      metric_rows[[length(metric_rows) + 1]] <- cbind(
        data.frame(
          repeat_id = repeat_id,
          candidate_id = candidate_id,
          permeability_dm_per_min = permeability,
          scenario_name = scenario_name,
          stringsAsFactors = FALSE
        ),
        metrics
      )
      endpoints <- curve_endpoints(curve)
      endpoint_rows[[length(endpoint_rows) + 1]] <- data.frame(
        repeat_id = repeat_id,
        candidate_id = candidate_id,
        permeability_dm_per_min = permeability,
        scenario_name = scenario_name,
        endpoint = names(endpoints),
        predicted_value = as.numeric(endpoints),
        unit = c("umol*h/l", "umol/l", "h"),
        stringsAsFactors = FALSE
      )
    }
    secondary_predicted <- secondary_endpoints(
      weighted_secondary_curve(curves)
    )
    for (endpoint in names(secondary_predicted)) {
      spec <- secondary_observed[[endpoint]]
      predicted_value <- unname(secondary_predicted[[endpoint]])
      observed_value <- as.numeric(spec$observed_value)
      fold_error <- predicted_value / observed_value
      absolute_error <- abs(predicted_value - observed_value)
      accepted <- if (endpoint == "tmax") {
        absolute_error <= as.numeric(
          config$thresholds$tmax_absolute_error_h_max
        )
      } else {
        fold_error >= as.numeric(
          config$thresholds$endpoint_fold_error_min
        ) && fold_error <= as.numeric(
          config$thresholds$endpoint_fold_error_max
        )
      }
      secondary_rows[[length(secondary_rows) + 1]] <- data.frame(
        repeat_id = repeat_id,
        candidate_id = candidate_id,
        permeability_dm_per_min = permeability,
        endpoint = endpoint,
        predicted_value = predicted_value,
        observed_value = observed_value,
        unit = spec$unit,
        fold_error = fold_error,
        absolute_error = absolute_error,
        accepted = accepted,
        metric_id = spec$metric_id,
        source_artifact_id = spec$source_artifact_id,
        source_location = spec$source_location,
        stringsAsFactors = FALSE
      )
      endpoint_rows[[length(endpoint_rows) + 1]] <- data.frame(
        repeat_id = repeat_id,
        candidate_id = candidate_id,
        permeability_dm_per_min = permeability,
        scenario_name = "EXT2_PMC572_WEIGHTED",
        endpoint = endpoint,
        predicted_value = predicted_value,
        unit = spec$unit,
        stringsAsFactors = FALSE
      )
    }
  }
  list(
    status = do.call(rbind, candidate_status_rows),
    curves = do.call(rbind, curve_rows),
    points = do.call(rbind, point_rows),
    metrics = do.call(rbind, metric_rows),
    secondary = do.call(rbind, secondary_rows),
    endpoints = do.call(rbind, endpoint_rows),
    curve_hashes = do.call(rbind, curve_hash_rows)
  )
}

deterministic_first <- run_validation_set(1)
deterministic_second <- run_validation_set(2)
status_table <- rbind(deterministic_first$status, deterministic_second$status)
curve_table <- rbind(deterministic_first$curves, deterministic_second$curves)
point_table <- rbind(deterministic_first$points, deterministic_second$points)
metric_table <- rbind(deterministic_first$metrics, deterministic_second$metrics)
secondary_table <- rbind(
  deterministic_first$secondary,
  deterministic_second$secondary
)
endpoint_table <- rbind(
  deterministic_first$endpoints,
  deterministic_second$endpoints
)
curve_hash_table <- rbind(
  deterministic_first$curve_hashes,
  deterministic_second$curve_hashes
)
v3_write_csv(
  status_table,
  file.path(validation_directory, "set_candidate_execution_status.csv")
)
v3_write_csv(
  curve_table,
  file.path(validation_directory, "set_deterministic_curves.csv")
)
v3_write_csv(
  point_table,
  file.path(validation_directory, "set_point_predictions_and_residuals.csv")
)
v3_write_csv(
  metric_table,
  file.path(validation_directory, "set_validation_metrics.csv")
)
v3_write_csv(
  secondary_table,
  file.path(validation_directory, "set_secondary_validation_metrics.csv")
)
v3_write_csv(
  endpoint_table,
  file.path(validation_directory, "set_endpoint_predictions.csv")
)
v3_write_csv(
  curve_hash_table,
  file.path(validation_directory, "set_curve_checksums.csv")
)

prediction_package_first <- deterministic_first$points[
  order(
    deterministic_first$points$candidate_id,
    deterministic_first$points$scenario_name,
    deterministic_first$points$time_min
  ),
  setdiff(names(deterministic_first$points), "repeat_id")
]
prediction_package_second <- deterministic_second$points[
  order(
    deterministic_second$points$candidate_id,
    deterministic_second$points$scenario_name,
    deterministic_second$points$time_min
  ),
  setdiff(names(deterministic_second$points), "repeat_id")
]
prediction_package_first_path <- file.path(
  validation_directory,
  "prediction_package_repeat_1.csv"
)
prediction_package_second_path <- file.path(
  validation_directory,
  "prediction_package_repeat_2.csv"
)
v3_write_csv(prediction_package_first, prediction_package_first_path)
v3_write_csv(prediction_package_second, prediction_package_second_path)

curve_hash_first <- deterministic_first$curve_hashes[
  order(deterministic_first$curve_hashes$candidate_id),
]
curve_hash_second <- deterministic_second$curve_hashes[
  order(deterministic_second$curve_hashes$candidate_id),
]
curve_checksums_repeatable <-
  nrow(curve_hash_first) == length(admissible_ids) &&
  nrow(curve_hash_second) == length(admissible_ids) &&
  identical(curve_hash_first$candidate_id, curve_hash_second$candidate_id) &&
  identical(curve_hash_first$sha256, curve_hash_second$sha256) &&
  identical(curve_hash_first$byte_size, curve_hash_second$byte_size)
prediction_packages_repeatable <-
  v3_sha256_file(prediction_package_first_path) ==
  v3_sha256_file(prediction_package_second_path)
metric_tables_repeatable <- identical(
  deterministic_first$metrics[
    , setdiff(names(deterministic_first$metrics), "repeat_id")
  ],
  deterministic_second$metrics[
    , setdiff(names(deterministic_second$metrics), "repeat_id")
  ]
) && identical(
  deterministic_first$secondary[
    , setdiff(names(deterministic_first$secondary), "repeat_id")
  ],
  deterministic_second$secondary[
    , setdiff(names(deterministic_second$secondary), "repeat_id")
  ]
)
v3_write_json(
  list(
    admissible_candidate_ids = as.list(admissible_ids),
    curve_checksums_identical = curve_checksums_repeatable,
    prediction_package_sha256_repeat_1 =
      v3_sha256_file(prediction_package_first_path),
    prediction_package_sha256_repeat_2 =
      v3_sha256_file(prediction_package_second_path),
    prediction_packages_identical = prediction_packages_repeatable,
    metric_tables_identical = metric_tables_repeatable,
    accepted = curve_checksums_repeatable &&
      prediction_packages_repeatable && metric_tables_repeatable
  ),
  file.path(validation_directory, "deterministic_repeatability.json")
)

first_endpoints <- deterministic_first$endpoints
endpoint_groups <- unique(first_endpoints[, c("scenario_name", "endpoint", "unit")])
stability_rows <- lapply(seq_len(nrow(endpoint_groups)), function(index) {
  group <- endpoint_groups[index, ]
  rows <- first_endpoints[
    first_endpoints$scenario_name == group$scenario_name &
      first_endpoints$endpoint == group$endpoint,
  ]
  minimum <- min(rows$predicted_value)
  maximum <- max(rows$predicted_value)
  span <- if (is.finite(minimum) && minimum > 0) maximum / minimum else Inf
  accepted <- nrow(rows) == length(admissible_ids) &&
    all(is.finite(rows$predicted_value)) &&
    span <= as.numeric(config$thresholds$prediction_set_fold_span_max)
  data.frame(
    scenario_name = group$scenario_name,
    endpoint = group$endpoint,
    unit = group$unit,
    candidate_count = nrow(rows),
    minimum = minimum,
    maximum = maximum,
    max_min_fold_span = span,
    threshold = as.numeric(config$thresholds$prediction_set_fold_span_max),
    accepted = accepted,
    stringsAsFactors = FALSE
  )
})
stability_table <- do.call(rbind, stability_rows)
v3_write_csv(
  stability_table,
  file.path(validation_directory, "set_endpoint_stability.csv")
)

deterministic_checks <- list(
  all_admissible_candidates_executed_twice =
    nrow(status_table) == length(admissible_ids) * 2 &&
      all(status_table$status == "ok"),
  all_five_validation_curves_pass_for_every_candidate =
    nrow(metric_table) == length(admissible_ids) * 5 * 2 &&
      all(metric_table$accepted),
  all_five_secondary_endpoints_pass_for_every_candidate =
    nrow(secondary_table) == length(admissible_ids) * 5 * 2 &&
      all(secondary_table$accepted),
  all_endpoint_pass_fail_conclusions_invariant =
    all(metric_table$accepted) && all(secondary_table$accepted),
  all_endpoint_fold_spans_at_most_1_10 =
    nrow(stability_table) > 0 && all(stability_table$accepted),
  curve_checksums_repeatable = curve_checksums_repeatable,
  prediction_packages_repeatable = prediction_packages_repeatable,
  metric_tables_repeatable = metric_tables_repeatable
)
deterministic_accepted <- all(unlist(deterministic_checks))

write_terminal_manifest <- function(scientific_status) {
  validation_manifest <- list(
    schema_version = "1.0.0",
    protocol_version = config$protocol_version,
    generated_at = v3_utc_now(),
    config_sha256 = config_sha256,
    identification_manifest_sha256 =
      v3_sha256_file(identification_manifest_path),
    absorption_set_manifest_sha256 = v3_sha256_file(set_manifest_path),
    validation_package_sha256 = v3_sha256_file(validation_package_path),
    validation_package_checksum = validation_package$checksum,
    validation_package_loaded = TRUE,
    validation_package_loaded_at = validation_package_loaded_at,
    admissible_candidate_count = length(admissible_ids),
    deterministic = list(
      checks = deterministic_checks,
      accepted = deterministic_accepted
    ),
    uncertainty = list(
      requested_sample_count = as.integer(config$uncertainty$sample_count),
      successful_sample_count = 0,
      accepted = FALSE
    ),
    parameter_identifiability = FALSE,
    prediction_identifiability = FALSE,
    scientific_status = scientific_status,
    registration_allowed = FALSE,
    artifacts = v3_artifact_inventory(
      validation_directory,
      excluded = "validation_manifest.json"
    )
  )
  v3_write_json(validation_manifest, validation_manifest_path)
  final_manifest <- list(
    schema_version = "1.0.0",
    protocol_version = config$protocol_version,
    generated_at = v3_utc_now(),
    config_sha256 = config_sha256,
    run_completed = TRUE,
    scenario_count = length(simulations),
    identification = identification_manifest$identification,
    validation = validation_manifest$deterministic,
    uncertainty = validation_manifest$uncertainty,
    validation_package_loaded = TRUE,
    parameter_identifiability = FALSE,
    prediction_identifiability = FALSE,
    scientific_status = scientific_status,
    registration_allowed = FALSE,
    artifacts = v3_artifact_inventory(
      root_output,
      excluded = "run_manifest.json"
    )
  )
  v3_write_json(final_manifest, run_manifest_path)
}

if (!deterministic_accepted) {
  v3_write_json(
    list(
      status = "completed_deterministic_failure",
      config_sha256 = config_sha256,
      identification_manifest_sha256 =
        v3_sha256_file(identification_manifest_path),
      absorption_set_manifest_sha256 = v3_sha256_file(set_manifest_path),
      validation_package_sha256 = v3_sha256_file(validation_package_path),
      uncertainty_seed = as.integer(config$uncertainty$seed),
      sample_count = as.integer(config$uncertainty$sample_count),
      completed_scenarios = as.list(validation_names),
      timestamp_utc = v3_utc_now()
    ),
    progress_manifest_path
  )
  write_terminal_manifest("quarantined_prediction_not_robust")
  quit(status = 3)
}

lhs_matrix <- function(sample_count, dimensions, seed) {
  set.seed(seed)
  output <- matrix(NA_real_, nrow = sample_count, ncol = dimensions)
  for (column in seq_len(dimensions)) {
    output[, column] <- (
      sample.int(sample_count, sample_count, replace = FALSE) - 1 +
        runif(sample_count)
    ) / sample_count
  }
  output
}

truncated_normal_quantile <- function(
  u,
  mean_value,
  sd_value,
  lower_value,
  upper_value
) {
  lower_probability <- pnorm((lower_value - mean_value) / sd_value)
  upper_probability <- pnorm((upper_value - mean_value) / sd_value)
  qnorm(lower_probability + u * (upper_probability - lower_probability)) *
    sd_value + mean_value
}

sample_count <- as.integer(config$uncertainty$sample_count)
uncertainty_seed <- as.integer(config$uncertainty$seed)
lhs <- lhs_matrix(sample_count, 4, uncertainty_seed)
clearance_sd <- sqrt(clearance_covariance$variance_original_units[[1]])
sampled_clearance <- truncated_normal_quantile(
  lhs[, 1],
  best_clearance,
  clearance_sd,
  clearance_lower,
  clearance_upper
)
log_permeability_lower <- log10(min(admissible_values))
log_permeability_upper <- log10(max(admissible_values))
sampled_permeability <- 10^(
  log_permeability_lower +
    lhs[, 2] * (log_permeability_upper - log_permeability_lower)
)
sampled_fu <- truncated_normal_quantile(
  lhs[, 3],
  as.numeric(config$uncertainty$fraction_unbound_mean),
  as.numeric(config$uncertainty$fraction_unbound_sd),
  0,
  1
)
renal_values <- as.numeric(unlist(
  config$uncertainty$renal_clearance_bootstrap_values
))
if (length(renal_values) != 6) {
  stop("Renal bootstrap source values drifted.")
}
set.seed(uncertainty_seed + 1L)
sampled_renal <- vapply(seq_len(sample_count), function(index) {
  mean(sample(renal_values, length(renal_values), replace = TRUE))
}, numeric(1))
uncertainty_samples <- data.frame(
  sample_id = seq_len(sample_count),
  total_cyp1a2_specific_clearance = sampled_clearance,
  specific_intestinal_permeability_transcellular = sampled_permeability,
  caffeine_fraction_unbound_plasma = sampled_fu,
  caffeine_renal_plasma_clearance = sampled_renal,
  permeability_set_lower_dm_per_min = min(admissible_values),
  permeability_set_upper_guard_dm_per_min = max(admissible_values),
  permeability_upper_bound_right_censored = TRUE,
  stringsAsFactors = FALSE
)
uncertainty_sample_path <- file.path(
  validation_directory,
  "uncertainty_parameter_samples.csv"
)
if (file.exists(uncertainty_sample_path)) {
  existing_samples <- read.csv(
    uncertainty_sample_path,
    stringsAsFactors = FALSE
  )
  if (!isTRUE(all.equal(
    existing_samples,
    uncertainty_samples,
    tolerance = 1e-14,
    check.attributes = FALSE
  ))) {
    stop("Resumed uncertainty parameter sample matrix drifted.")
  }
} else {
  v3_write_csv(uncertainty_samples, uncertainty_sample_path)
}

uncertainty_ids <- c(
  clearance_id,
  permeability_id,
  "caffeine_fraction_unbound_plasma",
  "caffeine_renal_plasma_clearance"
)
initial_uncertainty_logical <- c(
  total_cyp1a2_specific_clearance = best_clearance,
  specific_intestinal_permeability_transcellular = min(admissible_values),
  caffeine_fraction_unbound_plasma =
    as.numeric(config$uncertainty$fraction_unbound_mean),
  caffeine_renal_plasma_clearance = mean(renal_values)
)
uncertainty_paths <- names(v3_runtime_values_for(
  initial_uncertainty_logical,
  parameters,
  uncertainty_ids
))
if (length(uncertainty_paths) != 6 || anyDuplicated(uncertainty_paths)) {
  stop("Uncertainty runtime path set drifted.")
}
uncertainty_batches <- make_batches(all_expected_names, uncertainty_paths)
sample_curves <- list()
sample_times <- list()
successful_samples <- rep(TRUE, sample_count)
batch_size <- as.integer(config$uncertainty$batch_size)
uncertainty_progress_path <- file.path(
  validation_directory,
  "uncertainty_progress.json"
)
checkpoint_directory <- file.path(validation_directory, "checkpoints")
dir.create(checkpoint_directory, recursive = TRUE, showWarnings = FALSE)
completed_scenarios <- character()

for (scenario_name in all_expected_names) {
  checkpoint_path <- file.path(
    checkpoint_directory,
    paste0(scenario_name, ".rds")
  )
  if (file.exists(checkpoint_path)) {
    checkpoint <- readRDS(checkpoint_path)
    scenario_matrix <- checkpoint$concentration_umol_per_l
    scenario_time <- checkpoint$time_min
    if (
      !identical(as.character(checkpoint$scenario_name), scenario_name) ||
        !identical(as.character(checkpoint$config_sha256), config_sha256) ||
        !identical(
          as.character(checkpoint$identification_manifest_sha256),
          v3_sha256_file(identification_manifest_path)
        ) ||
        !identical(
          as.character(checkpoint$validation_package_sha256),
          v3_sha256_file(validation_package_path)
        ) ||
        !is.matrix(scenario_matrix) ||
        nrow(scenario_matrix) != sample_count ||
        length(scenario_time) != ncol(scenario_matrix) ||
        any(!is.finite(scenario_matrix)) ||
        any(scenario_matrix < 0) ||
        any(!is.finite(scenario_time)) ||
        any(diff(scenario_time) <= 0)
    ) {
      stop(paste("Invalid uncertainty checkpoint:", scenario_name))
    }
  } else {
    batch <- uncertainty_batches[[scenario_name]]
    scenario_matrix <- NULL
    scenario_time <- NULL
    for (first_index in seq(1, sample_count, by = batch_size)) {
      indices <- first_index:min(first_index + batch_size - 1, sample_count)
      result_ids <- character(length(indices))
      for (offset in seq_along(indices)) {
        sample_index <- indices[[offset]]
        logical_values <- c(
          total_cyp1a2_specific_clearance = sampled_clearance[[sample_index]],
          specific_intestinal_permeability_transcellular =
            sampled_permeability[[sample_index]],
          caffeine_fraction_unbound_plasma = sampled_fu[[sample_index]],
          caffeine_renal_plasma_clearance = sampled_renal[[sample_index]]
        )
        values_by_path <- v3_runtime_values_for(
          logical_values,
          parameters,
          uncertainty_ids
        )
        values <- batch_value_vector(batch, values_by_path)
        result_ids[[offset]] <- batch$addRunValues(parameterValues = values)
      }
      batch_results <- runSimulationBatches(
        list(batch),
        silentMode = TRUE,
        stopIfFails = TRUE
      )
      flattened <- v3_flatten_batch_results(batch_results)
      for (offset in seq_along(indices)) {
        sample_index <- indices[[offset]]
        curve <- v3_curve_from_result(
          v3_result_for_id(flattened, result_ids[[offset]]),
          target_output
        )
        if (is.null(scenario_time)) {
          scenario_time <- curve$time_min
          scenario_matrix <- matrix(
            NA_real_,
            nrow = sample_count,
            ncol = length(scenario_time)
          )
        }
        if (!identical(scenario_time, curve$time_min)) {
          stop(paste("Uncertainty output grid drifted:", scenario_name))
        }
        scenario_matrix[sample_index, ] <- curve$concentration_umol_per_l
      }
      v3_write_json(
        list(
          scenario_name = scenario_name,
          completed_samples = max(indices),
          total_samples = sample_count,
          timestamp_utc = v3_utc_now(),
          validation_package_loaded = TRUE
        ),
        uncertainty_progress_path
      )
    }
    v3_write_rds(
      list(
        scenario_name = scenario_name,
        time_min = scenario_time,
        concentration_umol_per_l = scenario_matrix,
        config_sha256 = config_sha256,
        identification_manifest_sha256 =
          v3_sha256_file(identification_manifest_path),
        validation_package_sha256 = v3_sha256_file(validation_package_path)
      ),
      checkpoint_path
    )
  }
  sample_curves[[scenario_name]] <- scenario_matrix
  sample_times[[scenario_name]] <- scenario_time
  completed_scenarios <- c(completed_scenarios, scenario_name)
  v3_write_json(
    list(
      status = "uncertainty_in_progress",
      config_sha256 = config_sha256,
      identification_manifest_sha256 =
        v3_sha256_file(identification_manifest_path),
      absorption_set_manifest_sha256 = v3_sha256_file(set_manifest_path),
      validation_package_sha256 = v3_sha256_file(validation_package_path),
      uncertainty_seed = uncertainty_seed,
      sample_count = sample_count,
      completed_scenarios = as.list(completed_scenarios),
      timestamp_utc = v3_utc_now()
    ),
    progress_manifest_path
  )
}

quantile_probabilities <- as.numeric(unlist(config$uncertainty$quantiles))
quantile_rows <- list()
uncertainty_point_rows <- list()
uncertainty_endpoint_rows <- list()
for (scenario_name in all_expected_names) {
  matrix_values <- sample_curves[[scenario_name]]
  times <- sample_times[[scenario_name]]
  quantiles <- apply(
    matrix_values,
    2,
    quantile,
    probs = quantile_probabilities,
    names = FALSE,
    type = 8
  )
  quantile_rows[[length(quantile_rows) + 1]] <- data.frame(
    scenario_name = scenario_name,
    dataset_id = scenarios[[scenario_name]]$dataset_id,
    split = scenarios[[scenario_name]]$split,
    time_min = times,
    p05_umol_per_l = quantiles[1, ],
    p50_umol_per_l = quantiles[2, ],
    p95_umol_per_l = quantiles[3, ],
    stringsAsFactors = FALSE
  )
  observations <- v3_observation_frame(scenarios[[scenario_name]])
  if (nrow(observations) > 0) {
    for (observation_index in seq_len(nrow(observations))) {
      predictions <- apply(matrix_values, 1, function(values) {
        approx(
          times,
          values,
          xout = observations$time_min[[observation_index]],
          ties = "ordered"
        )$y
      })
      prediction_quantiles <- quantile(
        predictions,
        probs = quantile_probabilities,
        names = FALSE,
        type = 8
      )
      observed <- observations$observed_umol_per_l[[observation_index]]
      uncertainty_point_rows[[length(uncertainty_point_rows) + 1]] <- data.frame(
        scenario_name = scenario_name,
        dataset_id = observations$dataset_id[[observation_index]],
        split = observations$split[[observation_index]],
        observation_id = observations$observation_id[[observation_index]],
        time_min = observations$time_min[[observation_index]],
        observed_umol_per_l = observed,
        p05_umol_per_l = prediction_quantiles[[1]],
        p50_umol_per_l = prediction_quantiles[[2]],
        p95_umol_per_l = prediction_quantiles[[3]],
        covered_90_percent = observed >= prediction_quantiles[[1]] &&
          observed <= prediction_quantiles[[3]],
        included = observations$included[[observation_index]],
        stringsAsFactors = FALSE
      )
    }
  }
  for (sample_index in seq_len(sample_count)) {
    endpoints <- curve_endpoints(data.frame(
      time_min = times,
      concentration_umol_per_l = matrix_values[sample_index, ]
    ))
    uncertainty_endpoint_rows[[length(uncertainty_endpoint_rows) + 1]] <- data.frame(
      sample_id = sample_index,
      scenario_name = scenario_name,
      dataset_id = scenarios[[scenario_name]]$dataset_id,
      auc_0_24_h_umol_h_per_l =
        endpoints[["auc_0_24_h_umol_h_per_l"]],
      cmax_umol_per_l = endpoints[["cmax_umol_per_l"]],
      tmax_h = endpoints[["tmax_h"]],
      stringsAsFactors = FALSE
    )
  }
}
quantile_table <- do.call(rbind, quantile_rows)
uncertainty_point_table <- do.call(rbind, uncertainty_point_rows)
uncertainty_endpoint_table <- do.call(rbind, uncertainty_endpoint_rows)
v3_write_csv(
  quantile_table,
  file.path(validation_directory, "visual_predictive_check_quantiles.csv")
)
v3_write_csv(
  uncertainty_point_table,
  file.path(validation_directory, "uncertainty_observation_coverage.csv")
)
v3_write_csv(
  uncertainty_endpoint_table,
  file.path(validation_directory, "uncertainty_endpoint_samples.csv")
)

male_times <- sample_times[["EXT2_PMC572_PO_MALE"]]
female_times <- sample_times[["EXT2_PMC572_PO_FEMALE"]]
if (!identical(male_times, female_times)) {
  stop("PMC572 uncertainty sex-specific output grids differ.")
}
secondary_uncertainty_rows <- list()
for (sample_index in seq_len(sample_count)) {
  curve <- data.frame(
    time_min = male_times,
    concentration_umol_per_l =
      0.5 * sample_curves[["EXT2_PMC572_PO_MALE"]][sample_index, ] +
      0.5 * sample_curves[["EXT2_PMC572_PO_FEMALE"]][sample_index, ]
  )
  values <- secondary_endpoints(curve)
  secondary_uncertainty_rows[[sample_index]] <- data.frame(
    sample_id = sample_index,
    auc_0_t = values[["auc_0_t"]],
    auc_0_inf = values[["auc_0_inf"]],
    cmax = values[["cmax"]],
    tmax = values[["tmax"]],
    half_life = values[["half_life"]],
    stringsAsFactors = FALSE
  )
}
secondary_uncertainty <- do.call(rbind, secondary_uncertainty_rows)
v3_write_csv(
  secondary_uncertainty,
  file.path(validation_directory, "secondary_uncertainty_endpoints.csv")
)

partial_spearman <- function(parameters_frame, endpoint) {
  if (length(unique(endpoint)) < 2) {
    return(list(
      values = rep(NA_real_, ncol(parameters_frame)),
      status = "endpoint_constant"
    ))
  }
  ranked_x <- apply(parameters_frame, 2, rank, ties.method = "average")
  ranked_y <- rank(endpoint, ties.method = "average")
  values <- vapply(seq_len(ncol(ranked_x)), function(index) {
    controls <- ranked_x[, -index, drop = FALSE]
    residual_x <- residuals(lm(ranked_x[, index] ~ controls))
    residual_y <- residuals(lm(ranked_y ~ controls))
    cor(residual_x, residual_y, method = "pearson")
  }, numeric(1))
  list(
    values = values,
    status = ifelse(all(is.finite(values)), "ok", "singular")
  )
}

parameter_matrix <- uncertainty_samples[, uncertainty_ids]
prcc_rows <- list()
for (scenario_name in all_expected_names) {
  rows <- uncertainty_endpoint_table[
    uncertainty_endpoint_table$scenario_name == scenario_name,
  ]
  rows <- rows[order(rows$sample_id), ]
  for (endpoint in c(
    "auc_0_24_h_umol_h_per_l",
    "cmax_umol_per_l",
    "tmax_h"
  )) {
    correlations <- partial_spearman(parameter_matrix, rows[[endpoint]])
    prcc_rows[[length(prcc_rows) + 1]] <- data.frame(
      scenario_name = scenario_name,
      endpoint = endpoint,
      parameter_id = uncertainty_ids,
      prcc = correlations$values,
      status = correlations$status,
      stringsAsFactors = FALSE
    )
  }
}
prcc_table <- do.call(rbind, prcc_rows)
v3_write_csv(
  prcc_table,
  file.path(validation_directory, "uncertainty_prcc.csv")
)

external_coverage_rows <- uncertainty_point_table[
  uncertainty_point_table$split == "external_validation" &
    uncertainty_point_table$included,
]
external_coverage <- mean(external_coverage_rows$covered_90_percent)
uncertainty_checks <- list(
  five_hundred_twelve_samples_successful =
    all(successful_samples) && length(successful_samples) == 512,
  permeability_samples_within_admissible_numerical_set =
    min(sampled_permeability) >= min(admissible_values) &&
      max(sampled_permeability) <= max(admissible_values),
  external_observation_coverage_at_least_80_percent =
    is.finite(external_coverage) &&
      external_coverage >= as.numeric(config$thresholds$external_coverage_min),
  all_prcc_values_finite =
    all(prcc_table$status == "ok") && all(is.finite(prcc_table$prcc)),
  all_seven_scenario_checkpoints_present =
    length(list.files(checkpoint_directory, pattern = "[.]rds$")) == 7
)
uncertainty_accepted <- all(unlist(uncertainty_checks))
fully_validated <- deterministic_accepted && uncertainty_accepted
scientific_status <- if (fully_validated) {
  "externally_validated_with_set_identified_permeability"
} else {
  "quarantined_uncertainty_failed"
}
v3_write_json(
  list(
    status = "completed",
    config_sha256 = config_sha256,
    identification_manifest_sha256 =
      v3_sha256_file(identification_manifest_path),
    absorption_set_manifest_sha256 = v3_sha256_file(set_manifest_path),
    validation_package_sha256 = v3_sha256_file(validation_package_path),
    uncertainty_seed = uncertainty_seed,
    sample_count = sample_count,
    completed_scenarios = as.list(completed_scenarios),
    timestamp_utc = v3_utc_now()
  ),
  progress_manifest_path
)

validation_manifest <- list(
  schema_version = "1.0.0",
  protocol_version = config$protocol_version,
  generated_at = v3_utc_now(),
  config_sha256 = config_sha256,
  identification_manifest_sha256 =
    v3_sha256_file(identification_manifest_path),
  absorption_set_manifest_sha256 = v3_sha256_file(set_manifest_path),
  validation_package_sha256 = v3_sha256_file(validation_package_path),
  validation_package_checksum = validation_package$checksum,
  validation_package_loaded = TRUE,
  validation_package_loaded_at = validation_package_loaded_at,
  admissible_candidate_count = length(admissible_ids),
  deterministic = list(
    checks = deterministic_checks,
    accepted = deterministic_accepted
  ),
  uncertainty = list(
    requested_sample_count = sample_count,
    successful_sample_count = sum(successful_samples),
    external_observation_coverage_90 = external_coverage,
    permeability_sampling =
      "stratified_log_uniform_over_admissible_numerical_set",
    permeability_upper_bound_semantics =
      "right_censored_at_numerical_guard",
    checks = uncertainty_checks,
    accepted = uncertainty_accepted
  ),
  parameter_identifiability = FALSE,
  prediction_identifiability = fully_validated,
  scientific_status = scientific_status,
  registration_allowed = fully_validated,
  artifacts = v3_artifact_inventory(
    validation_directory,
    excluded = "validation_manifest.json"
  )
)
v3_write_json(validation_manifest, validation_manifest_path)
final_manifest <- list(
  schema_version = "1.0.0",
  protocol_version = config$protocol_version,
  generated_at = v3_utc_now(),
  config_sha256 = config_sha256,
  run_completed = TRUE,
  scenario_count = length(simulations),
  identification = identification_manifest$identification,
  validation = validation_manifest$deterministic,
  uncertainty = validation_manifest$uncertainty,
  validation_package_loaded = TRUE,
  parameter_identifiability = FALSE,
  prediction_identifiability = fully_validated,
  scientific_status = scientific_status,
  registration_allowed = fully_validated,
  artifacts = v3_artifact_inventory(
    root_output,
    excluded = "run_manifest.json"
  )
)
v3_write_json(final_manifest, run_manifest_path)
quit(status = ifelse(fully_validated, 0, 3))
