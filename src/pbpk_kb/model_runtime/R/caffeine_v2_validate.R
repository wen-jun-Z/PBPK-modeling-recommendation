suppressPackageStartupMessages(library(ospsuite))
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2 || args[[2]] != "validate") {
  stop("Usage: caffeine_v2_validate.R <calibration-config.json> validate")
}

config_path <- normalizePath(args[[1]], mustWork = TRUE)
config <- fromJSON(config_path, simplifyVector = FALSE)
if (!isTRUE(config$accepted)) {
  stop("Calibration config is not accepted.")
}

utc_now <- function() {
  format(Sys.time(), "%Y-%m-%dT%H:%M:%OS6Z", tz = "UTC")
}

sha256_file <- function(path) {
  output <- system2("sha256sum", path, stdout = TRUE, stderr = TRUE)
  status <- attr(output, "status")
  if (!is.null(status) && status != 0) {
    stop(paste("sha256sum failed for", path, paste(output, collapse = " ")))
  }
  strsplit(output[[1]], "[[:space:]]+")[[1]][[1]]
}

write_json <- function(value, path) {
  temporary <- paste0(path, ".tmp")
  jsonlite::write_json(
    value,
    temporary,
    auto_unbox = TRUE,
    pretty = TRUE,
    na = "null",
    null = "null",
    digits = 17
  )
  if (!file.rename(temporary, path)) {
    stop(paste("Could not atomically write", path))
  }
}

write_csv <- function(value, path) {
  temporary <- paste0(path, ".tmp")
  write.csv(value, temporary, row.names = FALSE, na = "")
  if (!file.rename(temporary, path)) {
    stop(paste("Could not atomically write", path))
  }
}

write_rds <- function(value, path) {
  temporary <- paste0(path, ".tmp")
  saveRDS(value, temporary, version = 3, compress = "xz")
  if (!file.rename(temporary, path)) {
    stop(paste("Could not atomically write", path))
  }
}

relative_path <- function(path, root) {
  normalized_path <- normalizePath(path, mustWork = TRUE)
  normalized_root <- normalizePath(root, mustWork = TRUE)
  prefix <- paste0(normalized_root, .Platform$file.sep)
  if (!startsWith(normalized_path, prefix)) {
    stop(paste("Artifact escapes output directory:", normalized_path))
  }
  substring(normalized_path, nchar(prefix) + 1)
}

artifact_inventory <- function(root, excluded = character()) {
  paths <- list.files(root, recursive = TRUE, full.names = TRUE)
  paths <- paths[file.info(paths)$isdir %in% FALSE]
  paths <- paths[!(basename(paths) %in% excluded)]
  paths <- sort(paths)
  lapply(paths, function(path) {
    list(
      relative_path = relative_path(path, root),
      sha256 = sha256_file(path),
      byte_size = unname(file.info(path)$size)
    )
  })
}

nearly_equal <- function(left, right, tolerance = 1e-8) {
  abs(left - right) <= tolerance * max(1, abs(left), abs(right))
}

config_sha256 <- sha256_file(config_path)
if (as.character(packageVersion("ospsuite")) != config$ospsuite_r_version) {
  stop("OSPSuite-R version mismatch.")
}

root_output <- normalizePath(config$output_directory, mustWork = TRUE)
calibration_directory <- file.path(root_output, "01_calibration")
calibration_manifest_path <- file.path(
  calibration_directory,
  "calibration_manifest.json"
)
best_fit_path <- file.path(calibration_directory, "best_fit_parameters.csv")
covariance_path <- file.path(
  calibration_directory,
  "calibration_covariance_original_units.csv"
)
for (required_path in c(
  calibration_manifest_path,
  best_fit_path,
  covariance_path
)) {
  if (!file.exists(required_path)) {
    stop(paste("Accepted calibration artifact is missing:", required_path))
  }
}
calibration_manifest <- fromJSON(
  calibration_manifest_path,
  simplifyVector = FALSE
)
calibration_checks <- unlist(calibration_manifest$calibration$checks)
if (
  calibration_manifest$config_sha256 != config_sha256 ||
    calibration_manifest$phase != "calibrate" ||
    !isTRUE(calibration_manifest$run_completed) ||
    calibration_manifest$scientific_status !=
      "calibration_accepted_validation_locked" ||
    !isTRUE(calibration_manifest$calibration$accepted) ||
    length(calibration_checks) == 0 ||
    !all(calibration_checks) ||
    isTRUE(calibration_manifest$validation_package_loaded) ||
    isTRUE(calibration_manifest$validation_scenarios_loaded) ||
    isTRUE(calibration_manifest$registration_allowed)
) {
  stop("Calibration manifest does not unlock validation.")
}

parameters <- config$parameters
names(parameters) <- vapply(parameters, function(x) x$parameter_id, character(1))
calibrated_ids <- c(
  "total_cyp1a2_specific_clearance",
  "specific_intestinal_permeability_transcellular"
)
best_table <- read.csv(best_fit_path, stringsAsFactors = FALSE)
if (!identical(best_table$parameter_id, calibrated_ids)) {
  stop("Best-fit parameter identity or order drifted.")
}
best_theta <- as.numeric(best_table$fitted_value)
names(best_theta) <- calibrated_ids
lower <- vapply(
  parameters[calibrated_ids],
  function(x) as.numeric(x$lower_bound),
  numeric(1)
)
upper <- vapply(
  parameters[calibrated_ids],
  function(x) as.numeric(x$upper_bound),
  numeric(1)
)
if (any(!is.finite(best_theta)) || any(best_theta <= lower) || any(best_theta >= upper)) {
  stop("Best-fit parameters are outside strict locked bounds.")
}

covariance_table <- read.csv(covariance_path, stringsAsFactors = FALSE)
if (nrow(covariance_table) != 4) {
  stop("Calibration covariance must contain a 2 x 2 matrix.")
}
covariance <- matrix(
  NA_real_,
  nrow = 2,
  ncol = 2,
  dimnames = list(calibrated_ids, calibrated_ids)
)
for (row_index in seq_len(nrow(covariance_table))) {
  covariance[
    covariance_table$row_parameter[[row_index]],
    covariance_table$column_parameter[[row_index]]
  ] <- covariance_table$value[[row_index]]
}
if (
  any(!is.finite(covariance)) ||
    !isTRUE(all.equal(covariance, t(covariance), tolerance = 1e-12)) ||
    any(eigen(covariance, symmetric = TRUE, only.values = TRUE)$values <= 0)
) {
  stop("Calibration covariance is not finite symmetric positive definite.")
}

validation_package_path <- normalizePath(
  config$locked_validation_package_path,
  mustWork = TRUE
)
if (sha256_file(validation_package_path) != config$locked_validation_package_sha256) {
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
validation_package_loaded_at <- utc_now()

all_expected_names <- c(
  "CAL_BLANCHARD_IV",
  "CAL_BLANCHARD_PO",
  "HOLD_WAHLLANDER_IV",
  "EXT_EUDRACT_PO",
  "EXT_PMC535_PO",
  "EXT2_PMC572_PO_MALE",
  "EXT2_PMC572_PO_FEMALE"
)
scenarios <- c(config$calibration_scenarios, validation_package$scenarios)
names(scenarios) <- vapply(scenarios, function(x) x$name, character(1))
if (!identical(names(scenarios), all_expected_names)) {
  stop("Unlocked scenario order or identity drifted.")
}
if (
  sum(vapply(scenarios, function(x) length(x$observations), integer(1))) != 50
) {
  stop("Unlocked observation count drifted.")
}

mapping_rows <- list()
for (parameter_id in names(parameters)) {
  spec <- parameters[[parameter_id]]
  if (length(spec$runtime_mappings) == 0) {
    next
  }
  for (mapping in spec$runtime_mappings) {
    mapping_rows[[length(mapping_rows) + 1]] <- list(
      parameter_id = parameter_id,
      runtime_path = mapping$runtime_path,
      multiplier = as.numeric(mapping$multiplier),
      mapping_rule = mapping$mapping_rule,
      unit = spec$unit
    )
  }
}
mapping_paths <- vapply(mapping_rows, function(x) x$runtime_path, character(1))
if (length(mapping_rows) != 10 || anyDuplicated(mapping_paths)) {
  stop("Runtime mapping set drifted.")
}
mapping_by_path <- mapping_rows
names(mapping_by_path) <- mapping_paths

normalize_unit <- function(unit) {
  switch(
    unit,
    "umol/l" = "µmol/l",
    "ml/min/kg" = "ml/min/kg",
    "mg/l" = "mg/l",
    unit
  )
}

to_base_value <- function(parameter, value, unit) {
  if (unit %in% c("dimensionless", "fraction")) {
    return(as.numeric(value))
  }
  toBaseUnit(parameter, as.numeric(value), normalize_unit(unit))
}

runtime_values_for <- function(logical_values, logical_ids = names(logical_values)) {
  output <- numeric()
  for (parameter_id in logical_ids) {
    spec <- parameters[[parameter_id]]
    for (mapping in spec$runtime_mappings) {
      output[[mapping$runtime_path]] <-
        as.numeric(logical_values[[parameter_id]]) *
          as.numeric(mapping$multiplier)
    }
  }
  output
}

target_output <- config$target_output
simulations <- list()
preflight_rows <- list()
for (scenario_name in all_expected_names) {
  scenario <- scenarios[[scenario_name]]
  pkml_path <- file.path(config$runtime_root, scenario$pkml$relative_path)
  if (sha256_file(pkml_path) != scenario$pkml$sha256) {
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
    expected_base <- to_base_value(
      runtime_parameter,
      mapped_value,
      mapping$unit
    )
    actual_base <- as.numeric(runtime_parameter$value)
    valid <- nearly_equal(actual_base, expected_base)
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
  progress_manifest <- fromJSON(
    progress_manifest_path,
    simplifyVector = FALSE
  )
  if (
    !identical(as.character(progress_manifest$config_sha256), config_sha256) ||
      !identical(
        as.character(progress_manifest$calibration_manifest_sha256),
        sha256_file(calibration_manifest_path)
      ) ||
      !identical(
        as.character(progress_manifest$validation_package_sha256),
        sha256_file(validation_package_path)
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
  write_json(
    list(
      status = "in_progress",
      config_sha256 = config_sha256,
      calibration_manifest_sha256 = sha256_file(calibration_manifest_path),
      validation_package_sha256 = sha256_file(validation_package_path),
      uncertainty_seed = as.integer(config$uncertainty$seed),
      sample_count = as.integer(config$uncertainty$sample_count),
      completed_scenarios = list(),
      timestamp_utc = utc_now()
    ),
    progress_manifest_path
  )
}
write_csv(
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
    to_base_value(runtime_parameter, values_by_path[[path]], mapping$unit)
  }, numeric(1))
}

make_batches <- function(parameter_paths) {
  output <- lapply(all_expected_names, function(name) {
    createSimulationBatch(
      simulation = simulations[[name]],
      parametersOrPaths = parameter_paths
    )
  })
  names(output) <- all_expected_names
  output
}

flatten_batch_results <- function(results) {
  unlist(results, recursive = FALSE)
}

result_for_id <- function(flattened, result_id) {
  matches <- which(grepl(result_id, names(flattened), fixed = TRUE))
  if (length(matches) != 1) {
    stop(paste("Could not uniquely map SimulationBatch result", result_id))
  }
  flattened[[matches[[1]]]]
}

curve_from_result <- function(result) {
  values <- getOutputValues(result, quantitiesOrPaths = target_output)
  metadata <- values$metaData[[target_output]]
  if (metadata$unit != "µmol/l") {
    stop(paste("Unexpected target concentration unit:", metadata$unit))
  }
  data <- values$data[, c("Time", target_output)]
  names(data) <- c("time_min", "concentration_umol_per_l")
  data$time_min <- as.numeric(data$time_min)
  data$concentration_umol_per_l <- as.numeric(data$concentration_umol_per_l)
  if (
    nrow(data) < 2 ||
      any(!is.finite(data$time_min)) ||
      any(!is.finite(data$concentration_umol_per_l)) ||
      any(data$concentration_umol_per_l < 0) ||
      any(diff(data$time_min) <= 0)
  ) {
    stop("Simulation output is non-finite, negative, empty, or not increasing.")
  }
  data
}

run_parameter_set <- function(batches, values_by_path) {
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
  flattened <- flatten_batch_results(results)
  output <- lapply(names(batches), function(name) {
    curve_from_result(result_for_id(flattened, result_ids[[name]]))
  })
  names(output) <- names(batches)
  output
}

interpolate_curve <- function(curve, times_min) {
  if (any(times_min < min(curve$time_min)) || any(times_min > max(curve$time_min))) {
    stop("Observation time lies outside the simulation output grid.")
  }
  approx(
    x = curve$time_min,
    y = curve$concentration_umol_per_l,
    xout = times_min,
    method = "linear",
    ties = "ordered",
    rule = 1
  )$y
}

observation_frame <- function(scenario) {
  if (length(scenario$observations) == 0) {
    return(data.frame())
  }
  do.call(rbind, lapply(scenario$observations, function(x) {
    data.frame(
      observation_id = x$observation_id,
      dataset_id = x$dataset_id,
      split = x$split,
      time_min = as.numeric(x$time_min),
      observed_umol_per_l = as.numeric(x$value_umol_per_l),
      included = isTRUE(x$use_in_objective),
      exclusion_reason = ifelse(is.null(x$exclusion_reason), "", x$exclusion_reason),
      source_artifact_id = x$source_artifact_id,
      source_location = x$source_location,
      stringsAsFactors = FALSE
    )
  }))
}

calibrated_paths <- names(runtime_values_for(best_theta))
deterministic_batches <- make_batches(calibrated_paths)
deterministic_curves <- run_parameter_set(
  deterministic_batches,
  runtime_values_for(best_theta)
)
curve_rows <- list()
point_rows <- list()
for (scenario_name in all_expected_names) {
  curve <- deterministic_curves[[scenario_name]]
  scenario <- scenarios[[scenario_name]]
  curve_rows[[length(curve_rows) + 1]] <- data.frame(
    scenario_name = scenario_name,
    dataset_id = scenario$dataset_id,
    split = scenario$split,
    time_min = curve$time_min,
    concentration_umol_per_l = curve$concentration_umol_per_l,
    stringsAsFactors = FALSE
  )
  observations <- observation_frame(scenario)
  if (nrow(observations) > 0) {
    predicted <- interpolate_curve(curve, observations$time_min)
    ratio <- ifelse(
      observations$observed_umol_per_l > 0,
      predicted / observations$observed_umol_per_l,
      NA_real_
    )
    point_rows[[length(point_rows) + 1]] <- data.frame(
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
  }
}
deterministic_curve_table <- do.call(rbind, curve_rows)
point_table <- do.call(rbind, point_rows)
write_csv(
  deterministic_curve_table,
  file.path(validation_directory, "deterministic_curves.csv")
)
write_csv(
  point_table,
  file.path(validation_directory, "point_predictions_and_residuals.csv")
)

thresholds <- config$thresholds
concentration_metrics <- function(rows) {
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

dataset_ids <- unique(point_table$dataset_id)
validation_metrics <- do.call(rbind, lapply(dataset_ids, function(dataset_id) {
  concentration_metrics(point_table[point_table$dataset_id == dataset_id, ])
}))
write_csv(
  validation_metrics,
  file.path(validation_directory, "validation_metrics.csv")
)

residual_diagnostics <- do.call(rbind, lapply(dataset_ids, function(dataset_id) {
  rows <- point_table[
    point_table$dataset_id == dataset_id &
      point_table$included &
      is.finite(point_table$signed_log_residual),
  ]
  data.frame(
    dataset_id = dataset_id,
    split = rows$split[[1]],
    residual_count = nrow(rows),
    mean_signed_log_residual = mean(rows$signed_log_residual),
    sd_signed_log_residual = sd(rows$signed_log_residual),
    residual_time_spearman = ifelse(
      nrow(rows) >= 3,
      cor(rows$time_min, rows$signed_log_residual, method = "spearman"),
      NA_real_
    ),
    stringsAsFactors = FALSE
  )
}))
write_csv(
  residual_diagnostics,
  file.path(validation_directory, "residual_diagnostics.csv")
)

trapezoid_auc <- function(time_h, concentration) {
  sum(diff(time_h) * (head(concentration, -1) + tail(concentration, -1)) / 2)
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
        concentration_umol_per_l = interpolate_curve(curve, end_min)
      )
    )
  }
  concentration_ng_per_ml <-
    within$concentration_umol_per_l * molecular_weight
  auc_0_t <- trapezoid_auc(within$time_min / 60, concentration_ng_per_ml)
  maximum_index <- which.max(concentration_ng_per_ml)
  cmax <- concentration_ng_per_ml[[maximum_index]]
  tmax <- within$time_min[[maximum_index]] / 60
  terminal_times_h <- as.numeric(unlist(
    config$uncertainty$pmc572_terminal_times_h
  ))
  terminal_concentrations <- interpolate_curve(
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
  c24 <- interpolate_curve(curve, end_min) * molecular_weight
  c(
    auc_0_t = auc_0_t,
    auc_0_inf = auc_0_t + c24 / lambda_z,
    cmax = cmax,
    tmax = tmax,
    half_life = log(2) / lambda_z
  )
}

secondary_predicted <- secondary_endpoints(
  weighted_secondary_curve(deterministic_curves)
)
secondary_observed <- validation_package$secondary_metrics
names(secondary_observed) <- vapply(
  secondary_observed,
  function(x) x$endpoint,
  character(1)
)
secondary_rows <- lapply(names(secondary_predicted), function(endpoint) {
  spec <- secondary_observed[[endpoint]]
  predicted <- unname(secondary_predicted[[endpoint]])
  observed <- as.numeric(spec$observed_value)
  fold_error <- predicted / observed
  absolute_error <- abs(predicted - observed)
  accepted <- if (endpoint == "tmax") {
    absolute_error <= as.numeric(thresholds$tmax_absolute_error_h_max)
  } else {
    fold_error >= as.numeric(thresholds$endpoint_fold_error_min) &&
      fold_error <= as.numeric(thresholds$endpoint_fold_error_max)
  }
  data.frame(
    endpoint = endpoint,
    predicted_value = predicted,
    observed_value = observed,
    unit = spec$unit,
    fold_error = fold_error,
    absolute_error = absolute_error,
    accepted = accepted,
    metric_id = spec$metric_id,
    source_artifact_id = spec$source_artifact_id,
    source_location = spec$source_location,
    stringsAsFactors = FALSE
  )
})
secondary_table <- do.call(rbind, secondary_rows)
write_csv(
  secondary_table,
  file.path(validation_directory, "secondary_validation_metrics.csv")
)

metric_accepted <- function(dataset_id) {
  row <- validation_metrics[validation_metrics$dataset_id == dataset_id, ]
  nrow(row) == 1 && isTRUE(row$accepted[[1]])
}
internal_holdout_accepted <- metric_accepted("wahllander_1989_iv_3mgkg")
external_deterministic_accepted <-
  metric_accepted("eudract_2011_002291_16_oral_50mg") &&
    metric_accepted("pmc5350996_synthetic_200mg") &&
    all(secondary_table$accepted)

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

truncated_normal_quantile <- function(u, mean_value, sd_value, lower_value, upper_value) {
  lower_probability <- pnorm((lower_value - mean_value) / sd_value)
  upper_probability <- pnorm((upper_value - mean_value) / sd_value)
  qnorm(lower_probability + u * (upper_probability - lower_probability)) *
    sd_value + mean_value
}

sample_count <- as.integer(config$uncertainty$sample_count)
uncertainty_seed <- as.integer(config$uncertainty$seed)
lhs <- lhs_matrix(sample_count, 4, uncertainty_seed)
fit_sd <- sqrt(diag(covariance))
fit_correlation <- covariance / outer(fit_sd, fit_sd)
fit_correlation <- (fit_correlation + t(fit_correlation)) / 2
latent <- qnorm(lhs[, 1:2, drop = FALSE]) %*% chol(fit_correlation)
correlated_u <- pnorm(latent)
sampled_total_clearance <- truncated_normal_quantile(
  correlated_u[, 1], best_theta[[1]], fit_sd[[1]], lower[[1]], upper[[1]]
)
sampled_permeability <- truncated_normal_quantile(
  correlated_u[, 2], best_theta[[2]], fit_sd[[2]], lower[[2]], upper[[2]]
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
  total_cyp1a2_specific_clearance = sampled_total_clearance,
  specific_intestinal_permeability_transcellular = sampled_permeability,
  caffeine_fraction_unbound_plasma = sampled_fu,
  caffeine_renal_plasma_clearance = sampled_renal,
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
  write_csv(uncertainty_samples, uncertainty_sample_path)
}

uncertainty_ids <- c(
  "total_cyp1a2_specific_clearance",
  "specific_intestinal_permeability_transcellular",
  "caffeine_fraction_unbound_plasma",
  "caffeine_renal_plasma_clearance"
)
initial_uncertainty_logical <- c(
  best_theta,
  caffeine_fraction_unbound_plasma =
    as.numeric(config$uncertainty$fraction_unbound_mean),
  caffeine_renal_plasma_clearance = mean(renal_values)
)
uncertainty_paths <- names(runtime_values_for(
  initial_uncertainty_logical,
  uncertainty_ids
))
if (length(uncertainty_paths) != 6 || anyDuplicated(uncertainty_paths)) {
  stop("Uncertainty runtime path set drifted.")
}
uncertainty_batches <- make_batches(uncertainty_paths)
sample_curves <- list()
sample_times <- list()
successful_samples <- rep(TRUE, sample_count)
batch_size <- as.integer(config$uncertainty$batch_size)
progress_path <- file.path(validation_directory, "uncertainty_progress.json")
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
          as.character(checkpoint$validation_package_sha256),
          sha256_file(validation_package_path)
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
          total_cyp1a2_specific_clearance =
            sampled_total_clearance[[sample_index]],
          specific_intestinal_permeability_transcellular =
            sampled_permeability[[sample_index]],
          caffeine_fraction_unbound_plasma = sampled_fu[[sample_index]],
          caffeine_renal_plasma_clearance = sampled_renal[[sample_index]]
        )
        values_by_path <- runtime_values_for(logical_values, uncertainty_ids)
        values <- batch_value_vector(batch, values_by_path)
        result_ids[[offset]] <- batch$addRunValues(parameterValues = values)
      }
      batch_results <- runSimulationBatches(
        list(batch),
        silentMode = TRUE,
        stopIfFails = TRUE
      )
      flattened <- flatten_batch_results(batch_results)
      for (offset in seq_along(indices)) {
        sample_index <- indices[[offset]]
        curve <- curve_from_result(
          result_for_id(flattened, result_ids[[offset]])
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
      write_json(
        list(
          scenario_name = scenario_name,
          completed_samples = max(indices),
          total_samples = sample_count,
          timestamp_utc = utc_now(),
          validation_package_loaded = TRUE
        ),
        progress_path
      )
    }
    write_rds(
      list(
        scenario_name = scenario_name,
        time_min = scenario_time,
        concentration_umol_per_l = scenario_matrix,
        config_sha256 = config_sha256,
        validation_package_sha256 = sha256_file(validation_package_path)
      ),
      checkpoint_path
    )
  }
  sample_curves[[scenario_name]] <- scenario_matrix
  sample_times[[scenario_name]] <- scenario_time
  completed_scenarios <- c(completed_scenarios, scenario_name)
  write_json(
    list(
      status = "in_progress",
      config_sha256 = config_sha256,
      calibration_manifest_sha256 = sha256_file(calibration_manifest_path),
      validation_package_sha256 = sha256_file(validation_package_path),
      uncertainty_seed = uncertainty_seed,
      sample_count = sample_count,
      completed_scenarios = as.list(completed_scenarios),
      timestamp_utc = utc_now()
    ),
    progress_manifest_path
  )
}

quantile_probabilities <- as.numeric(unlist(config$uncertainty$quantiles))
quantile_rows <- list()
uncertainty_point_rows <- list()
endpoint_rows <- list()
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
  observations <- observation_frame(scenarios[[scenario_name]])
  if (nrow(observations) > 0) {
    for (observation_index in seq_len(nrow(observations))) {
      predictions <- apply(
        matrix_values,
        1,
        function(values) approx(
          times,
          values,
          xout = observations$time_min[[observation_index]],
          ties = "ordered"
        )$y
      )
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
    curve <- data.frame(
      time_min = times,
      concentration_umol_per_l = matrix_values[sample_index, ]
    )
    end_min <- min(24 * 60, max(times))
    within <- curve[curve$time_min <= end_min, ]
    if (tail(within$time_min, 1) < end_min) {
      within <- rbind(
        within,
        data.frame(
          time_min = end_min,
          concentration_umol_per_l = interpolate_curve(curve, end_min)
        )
      )
    }
    endpoint_rows[[length(endpoint_rows) + 1]] <- data.frame(
      sample_id = sample_index,
      scenario_name = scenario_name,
      dataset_id = scenarios[[scenario_name]]$dataset_id,
      auc_0_24_h_umol_h_per_l = trapezoid_auc(
        within$time_min / 60,
        within$concentration_umol_per_l
      ),
      cmax_umol_per_l = max(within$concentration_umol_per_l),
      tmax_h = within$time_min[[which.max(
        within$concentration_umol_per_l
      )]] / 60,
      stringsAsFactors = FALSE
    )
  }
}
quantile_table <- do.call(rbind, quantile_rows)
uncertainty_point_table <- do.call(rbind, uncertainty_point_rows)
endpoint_table <- do.call(rbind, endpoint_rows)
write_csv(
  quantile_table,
  file.path(validation_directory, "visual_predictive_check_quantiles.csv")
)
write_csv(
  uncertainty_point_table,
  file.path(validation_directory, "uncertainty_observation_coverage.csv")
)
write_csv(
  endpoint_table,
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
write_csv(
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
  rows <- endpoint_table[endpoint_table$scenario_name == scenario_name, ]
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
write_csv(
  prcc_table,
  file.path(validation_directory, "uncertainty_prcc.csv")
)

external_coverage_rows <- uncertainty_point_table[
  uncertainty_point_table$split == "external_validation" &
    uncertainty_point_table$included,
]
external_coverage <- mean(external_coverage_rows$covered_90_percent)
uncertainty_accepted <-
  all(successful_samples) &&
  is.finite(external_coverage) &&
  external_coverage >= as.numeric(thresholds$external_coverage_min) &&
  all(prcc_table$status == "ok") &&
  all(is.finite(prcc_table$prcc))
fully_validated <-
  internal_holdout_accepted &&
  external_deterministic_accepted &&
  uncertainty_accepted
scientific_status <- if (fully_validated) {
  "external_study_validated"
} else {
  "abstain_validation_failed"
}
write_json(
  list(
    status = "completed",
    config_sha256 = config_sha256,
    calibration_manifest_sha256 = sha256_file(calibration_manifest_path),
    validation_package_sha256 = sha256_file(validation_package_path),
    uncertainty_seed = uncertainty_seed,
    sample_count = sample_count,
    completed_scenarios = as.list(completed_scenarios),
    timestamp_utc = utc_now()
  ),
  progress_manifest_path
)

validation_manifest <- list(
  schema_version = "1.0.0",
  protocol_version = config$protocol_version,
  generated_at = utc_now(),
  config_sha256 = config_sha256,
  calibration_manifest_sha256 = sha256_file(calibration_manifest_path),
  validation_package_sha256 = sha256_file(validation_package_path),
  validation_package_checksum = validation_package$checksum,
  validation_package_loaded = TRUE,
  validation_package_loaded_at = validation_package_loaded_at,
  scenario_count = length(simulations),
  deterministic = list(
    internal_holdout_accepted = internal_holdout_accepted,
    external_deterministic_accepted = external_deterministic_accepted,
    pmc572_secondary_accepted = all(secondary_table$accepted)
  ),
  uncertainty = list(
    requested_sample_count = sample_count,
    successful_sample_count = sum(successful_samples),
    external_observation_coverage_90 = external_coverage,
    accepted = uncertainty_accepted
  ),
  scientific_status = scientific_status,
  registration_allowed = fully_validated,
  artifacts = artifact_inventory(
    validation_directory,
    excluded = "validation_manifest.json"
  )
)
write_json(
  validation_manifest,
  file.path(validation_directory, "validation_manifest.json")
)
final_manifest <- list(
  schema_version = "1.0.0",
  protocol_version = config$protocol_version,
  generated_at = utc_now(),
  config_sha256 = config_sha256,
  run_completed = TRUE,
  scenario_count = length(simulations),
  calibration = calibration_manifest$calibration,
  validation = validation_manifest$deterministic,
  uncertainty = validation_manifest$uncertainty,
  validation_package_loaded = TRUE,
  scientific_status = scientific_status,
  registration_allowed = fully_validated,
  artifacts = artifact_inventory(root_output, excluded = "run_manifest.json")
)
write_json(final_manifest, file.path(root_output, "run_manifest.json"))
quit(status = ifelse(fully_validated, 0, 3))
