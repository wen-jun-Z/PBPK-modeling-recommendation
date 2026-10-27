suppressPackageStartupMessages(library(ospsuite))
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1 || length(args) > 3) {
  stop(
    paste(
      "Usage: caffeine_fit_validate.R <run-config.json>",
      "[full|preflight|diagnostic] [diagnostic-output-directory]"
    )
  )
}

config_path <- normalizePath(args[[1]], mustWork = TRUE)
execution_mode <- if (length(args) >= 2) args[[2]] else "full"
if (!(execution_mode %in% c("full", "preflight", "diagnostic"))) {
  stop(paste("Unknown execution mode:", execution_mode))
}
if (execution_mode == "diagnostic" && length(args) != 3) {
  stop("Diagnostic mode requires a separate output directory.")
}
config <- fromJSON(config_path, simplifyVector = FALSE)
source_output_directory <- config$output_directory
output_directory <- if (execution_mode == "diagnostic") {
  normalizePath(args[[3]], mustWork = FALSE)
} else {
  source_output_directory
}
if (
  execution_mode == "diagnostic" &&
    dir.exists(output_directory) &&
    length(list.files(output_directory, all.files = TRUE, no.. = TRUE)) > 0
) {
  stop("Diagnostic output directory must be empty.")
}
dir.create(output_directory, recursive = TRUE, showWarnings = FALSE)

target_output <- config$target_output
penalty_objective <- 1e12

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

append_csv <- function(value, path) {
  write.table(
    value,
    path,
    append = file.exists(path),
    sep = ",",
    row.names = FALSE,
    col.names = !file.exists(path),
    qmethod = "double"
  )
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

artifact_inventory <- function(root) {
  paths <- list.files(root, recursive = TRUE, full.names = TRUE)
  paths <- paths[file.info(paths)$isdir %in% FALSE]
  paths <- paths[basename(paths) != "run_manifest.json"]
  paths <- sort(paths)
  lapply(paths, function(path) {
    list(
      relative_path = relative_path(path, root),
      sha256 = sha256_file(path),
      byte_size = unname(file.info(path)$size)
    )
  })
}

config_sha256 <- sha256_file(config_path)
if (as.character(packageVersion("ospsuite")) != config$ospsuite_r_version) {
  stop(
    paste(
      "OSPSuite-R version mismatch:",
      as.character(packageVersion("ospsuite")),
      "!=",
      config$ospsuite_r_version
    )
  )
}

parameters <- config$parameters
names(parameters) <- vapply(parameters, function(x) x$parameter_id, character(1))
path_to_parameter <- parameters
names(path_to_parameter) <- vapply(
  parameters,
  function(x) x$runtime_path,
  character(1)
)

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

nearly_equal <- function(left, right, tolerance = 1e-8) {
  abs(left - right) <= tolerance * max(1, abs(left), abs(right))
}

scenarios <- config$scenarios
names(scenarios) <- vapply(scenarios, function(x) x$name, character(1))
all_expected_names <- c(
  "CAL_BLANCHARD_IV",
  "CAL_BLANCHARD_PO",
  "HOLD_WAHLLANDER_IV",
  "EXT_EUDRACT_PO",
  "EXT_PMC535_PO",
  "EXT2_PMC572_PO_MALE",
  "EXT2_PMC572_PO_FEMALE"
)
calibration_names <- c("CAL_BLANCHARD_IV", "CAL_BLANCHARD_PO")
if (!identical(names(scenarios), all_expected_names)) {
  stop("Scenario order or identity drifted from the locked protocol.")
}
expected_names <- if (execution_mode == "diagnostic") {
  calibration_names
} else {
  all_expected_names
}

simulations <- list()
preflight_rows <- list()
for (scenario_name in expected_names) {
  scenario <- scenarios[[scenario_name]]
  pkml_path <- file.path(config$runtime_root, scenario$pkml$relative_path)
  if (sha256_file(pkml_path) != scenario$pkml$sha256) {
    stop(paste("PKML checksum mismatch:", scenario_name))
  }
  simulation <- loadSimulation(
    pkml_path,
    loadFromCache = FALSE,
    addToCache = FALSE,
    resetIds = FALSE
  )
  setOutputs(target_output, simulation, stopIfNotFound = TRUE)
  for (spec in parameters) {
    runtime_parameter <- getParameter(spec$runtime_path, simulation)
    expected_base <- to_base_value(runtime_parameter, spec$value, spec$unit)
    actual_base <- as.numeric(runtime_parameter$value)
    valid <- nearly_equal(actual_base, expected_base)
    preflight_rows[[length(preflight_rows) + 1]] <- data.frame(
      scenario_name = scenario_name,
      parameter_id = spec$parameter_id,
      runtime_path = spec$runtime_path,
      expected_value = as.numeric(spec$value),
      expected_unit = spec$unit,
      expected_base_value = expected_base,
      actual_base_value = actual_base,
      actual_base_unit = ifelse(is.null(runtime_parameter$unit), "", runtime_parameter$unit),
      valid = valid,
      stringsAsFactors = FALSE
    )
    if (!valid) {
      stop(
        paste(
          "Runtime parameter value drifted:",
          scenario_name,
          spec$parameter_id,
          actual_base,
          "!=",
          expected_base
        )
      )
    }
  }
  pka_count <- getParameter("Caffeine|pKa_Bases_Count", simulation)
  if (!nearly_equal(as.numeric(pka_count$value), 1)) {
    stop(paste("Caffeine base pKa count is not one:", scenario_name))
  }
  simulations[[scenario_name]] <- simulation
}
preflight_table <- do.call(rbind, preflight_rows)
write_csv(preflight_table, file.path(output_directory, "runtime_parameter_preflight.csv"))
if (execution_mode == "preflight") {
  preflight_manifest <- list(
    schema_version = "1.0.0",
    protocol_version = config$protocol_version,
    generated_at = utc_now(),
    config_sha256 = config_sha256,
    execution_mode = execution_mode,
    scenario_count = length(simulations),
    parameter_check_count = nrow(preflight_table),
    accepted = all(preflight_table$valid),
    artifacts = artifact_inventory(output_directory)
  )
  write_json(
    preflight_manifest,
    file.path(output_directory, "preflight_manifest.json")
  )
  quit(status = ifelse(preflight_manifest$accepted, 0, 2))
}

parameter_paths_for <- function(ids) {
  vapply(parameters[ids], function(x) x$runtime_path, character(1))
}

calibrated_ids <- c(
  "caffeine_cyp1a2_vmax_effective_n3",
  "caffeine_effective_lipophilicity"
)
calibrated_paths <- parameter_paths_for(calibrated_ids)

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

runtime_parameters_by_path <- lapply(
  names(path_to_parameter),
  function(path) getParameter(path, simulations[[expected_names[[1]]]])
)
names(runtime_parameters_by_path) <- names(path_to_parameter)

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
    spec <- path_to_parameter[[path]]
    runtime_parameter <- runtime_parameters_by_path[[path]]
    to_base_value(runtime_parameter, values_by_path[[path]], spec$unit)
  }, numeric(1))
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

contract_batch <- createSimulationBatch(
  simulation = simulations[[expected_names[[1]]]],
  parametersOrPaths = calibrated_paths
)
contract_human_values <- vapply(
  calibrated_ids,
  function(id) as.numeric(parameters[[id]]$value),
  numeric(1)
)
names(contract_human_values) <- calibrated_paths
contract_values <- batch_value_vector(contract_batch, contract_human_values)
contract_result_id <- contract_batch$addRunValues(
  parameterValues = contract_values
)
contract_results <- runSimulationBatches(
  list(contract_batch),
  silentMode = TRUE,
  stopIfFails = TRUE
)
contract_curve <- curve_from_result(
  result_for_id(
    flatten_batch_results(contract_results),
    contract_result_id
  )
)
batch_contract <- list(
  schema_version = "1.0.0",
  protocol_version = config$protocol_version,
  generated_at = utc_now(),
  scenario_name = expected_names[[1]],
  variable_parameter_paths = as.list(
    contract_batch$getVariableParameters()
  ),
  output_path = target_output,
  output_unit = "µmol/l",
  row_count = nrow(contract_curve),
  time_min = min(contract_curve$time_min),
  time_max = max(contract_curve$time_min),
  accepted = nrow(contract_curve) > 1 &&
    all(is.finite(contract_curve$concentration_umol_per_l)) &&
    all(contract_curve$concentration_umol_per_l >= 0)
)
write_json(
  batch_contract,
  file.path(output_directory, "batch_api_contract.json")
)
if (!batch_contract$accepted) {
  stop("SimulationBatch API contract run failed.")
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
      use_in_objective = isTRUE(x$use_in_objective),
      exclusion_reason = ifelse(is.null(x$exclusion_reason), "", x$exclusion_reason),
      source_artifact_id = x$source_artifact_id,
      source_location = x$source_location,
      stringsAsFactors = FALSE
    )
  }))
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

all_two_parameter_batches <- make_batches(expected_names, calibrated_paths)
calibration_batches <- all_two_parameter_batches[calibration_names]
calibration_observations <- lapply(
  calibration_names,
  function(name) observation_frame(scenarios[[name]])
)
names(calibration_observations) <- calibration_names

evaluation_path <- file.path(output_directory, "calibration_evaluations.csv")
evaluation_index <- 0L
evaluation_cache <- new.env(hash = TRUE, parent = emptyenv())

parameter_values_from_theta <- function(theta) {
  values <- as.numeric(theta)
  names(values) <- calibrated_paths
  values
}

evaluate_objective <- function(theta, start_index, phase, use_cache = TRUE) {
  evaluation_index <<- evaluation_index + 1L
  key <- paste(sprintf("%.17g", as.numeric(theta)), collapse = "|")
  if (use_cache && exists(key, envir = evaluation_cache, inherits = FALSE)) {
    cached <- get(key, envir = evaluation_cache, inherits = FALSE)
    append_csv(
      data.frame(
        evaluation_index = evaluation_index,
        start_index = start_index,
        phase = phase,
        timestamp_utc = utc_now(),
        vmax = theta[[1]],
        lipophilicity = theta[[2]],
        mse_iv = cached$mse_iv,
        mse_oral = cached$mse_oral,
        objective = cached$objective,
        status = "cached",
        error = "",
        stringsAsFactors = FALSE
      ),
      evaluation_path
    )
    return(cached)
  }
  evaluated <- tryCatch({
    curves <- run_parameter_set(
      calibration_batches,
      parameter_values_from_theta(theta)
    )
    mse <- vapply(calibration_names, function(name) {
      observations <- calibration_observations[[name]]
      observations <- observations[observations$use_in_objective, ]
      if (any(observations$observed_umol_per_l <= 0)) {
        stop("Calibration observations must be positive for log-MSE.")
      }
      predicted <- interpolate_curve(curves[[name]], observations$time_min)
      if (any(predicted <= 0)) {
        stop("Calibration prediction is non-positive.")
      }
      mean((log(predicted) - log(observations$observed_umol_per_l))^2)
    }, numeric(1))
    list(
      objective = 0.5 * mse[[1]] + 0.5 * mse[[2]],
      mse_iv = mse[[1]],
      mse_oral = mse[[2]],
      curves = curves,
      status = "ok",
      error = ""
    )
  }, error = function(error) {
    list(
      objective = penalty_objective,
      mse_iv = penalty_objective,
      mse_oral = penalty_objective,
      curves = NULL,
      status = "failed",
      error = conditionMessage(error)
    )
  })
  append_csv(
    data.frame(
      evaluation_index = evaluation_index,
      start_index = start_index,
      phase = phase,
      timestamp_utc = utc_now(),
      vmax = theta[[1]],
      lipophilicity = theta[[2]],
      mse_iv = evaluated$mse_iv,
      mse_oral = evaluated$mse_oral,
      objective = evaluated$objective,
      status = evaluated$status,
      error = evaluated$error,
      stringsAsFactors = FALSE
    ),
    evaluation_path
  )
  if (evaluated$status == "ok") {
    assign(
      key,
      evaluated[c("objective", "mse_iv", "mse_oral")],
      envir = evaluation_cache
    )
  }
  evaluated
}

estimated <- parameters[calibrated_ids]
lower <- vapply(estimated, function(x) as.numeric(x$lower_bound), numeric(1))
upper <- vapply(estimated, function(x) as.numeric(x$upper_bound), numeric(1))

if (execution_mode == "diagnostic") {
  source_manifest_path <- file.path(source_output_directory, "run_manifest.json")
  source_best_path <- file.path(source_output_directory, "best_fit_parameters.csv")
  if (!file.exists(source_manifest_path) || !file.exists(source_best_path)) {
    stop("Diagnostic source run is missing its manifest or best-fit parameters.")
  }
  source_manifest <- fromJSON(source_manifest_path, simplifyVector = FALSE)
  if (
    source_manifest$config_sha256 != config_sha256 ||
      !(source_manifest$scientific_status %in% c(
        "calibration_failed",
        "parameter_unidentifiable"
      )) ||
      isTRUE(source_manifest$registration_allowed) ||
      !is.null(source_manifest$validation)
  ) {
    stop("Diagnostic source run is not an isolated calibration failure.")
  }
  best_table <- read.csv(source_best_path, stringsAsFactors = FALSE)
  if (!identical(best_table$parameter_id, calibrated_ids)) {
    stop("Diagnostic best-fit parameter identity or order drifted.")
  }
  best_theta <- as.numeric(best_table$fitted_value)
  if (
    any(!is.finite(best_theta)) ||
      any(best_theta < lower) ||
      any(best_theta > upper)
  ) {
    stop("Diagnostic best-fit parameters are outside locked bounds.")
  }
  best_result <- evaluate_objective(
    best_theta,
    start_index = 0,
    phase = "diagnostic_best_fit",
    use_cache = FALSE
  )
  if (best_result$status != "ok") {
    stop(paste("Diagnostic best-fit simulation failed:", best_result$error))
  }

  curve_rows <- list()
  point_rows <- list()
  metric_rows <- list()
  for (scenario_name in calibration_names) {
    curve <- best_result$curves[[scenario_name]]
    observations <- calibration_observations[[scenario_name]]
    predicted <- interpolate_curve(curve, observations$time_min)
    ratio <- ifelse(
      observations$observed_umol_per_l > 0,
      predicted / observations$observed_umol_per_l,
      NA_real_
    )
    signed_log_residual <- ifelse(
      observations$observed_umol_per_l > 0 & predicted > 0,
      log(ratio),
      NA_real_
    )
    curve_rows[[length(curve_rows) + 1]] <- data.frame(
      scenario_name = scenario_name,
      time_min = curve$time_min,
      concentration_umol_per_l = curve$concentration_umol_per_l,
      stringsAsFactors = FALSE
    )
    point_rows[[length(point_rows) + 1]] <- data.frame(
      scenario_name = scenario_name,
      dataset_id = observations$dataset_id,
      observation_id = observations$observation_id,
      time_min = observations$time_min,
      observed_umol_per_l = observations$observed_umol_per_l,
      predicted_umol_per_l = predicted,
      predicted_to_observed_ratio = ratio,
      signed_log_residual = signed_log_residual,
      use_in_objective = observations$use_in_objective,
      source_artifact_id = observations$source_artifact_id,
      source_location = observations$source_location,
      stringsAsFactors = FALSE
    )
    included <- observations$use_in_objective &
      observations$observed_umol_per_l > 0 & predicted > 0
    ratios <- ratio[included]
    logs <- signed_log_residual[included]
    metric_rows[[length(metric_rows) + 1]] <- data.frame(
      scenario_name = scenario_name,
      dataset_id = observations$dataset_id[[1]],
      included_count = sum(included),
      gmfe = exp(mean(abs(logs))),
      geometric_bias = exp(mean(logs)),
      log_rmse = sqrt(mean(logs^2)),
      two_fold_fraction = mean(ratios >= 0.5 & ratios <= 2),
      mse_component = ifelse(
        scenario_name == calibration_names[[1]],
        best_result$mse_iv,
        best_result$mse_oral
      ),
      stringsAsFactors = FALSE
    )
  }
  write_csv(
    do.call(rbind, curve_rows),
    file.path(output_directory, "calibration_diagnostic_curves.csv")
  )
  write_csv(
    do.call(rbind, point_rows),
    file.path(output_directory, "calibration_diagnostic_points.csv")
  )
  write_csv(
    do.call(rbind, metric_rows),
    file.path(output_directory, "calibration_diagnostic_metrics.csv")
  )

  profile_rows <- list()
  profile_summaries <- list()
  for (parameter_index in seq_along(calibrated_ids)) {
    values <- seq(
      lower[[parameter_index]],
      upper[[parameter_index]],
      length.out = 21
    )
    parameter_rows <- list()
    for (grid_index in seq_along(values)) {
      theta <- best_theta
      theta[[parameter_index]] <- values[[grid_index]]
      result <- evaluate_objective(
        theta,
        start_index = 0,
        phase = paste0("diagnostic_profile_", calibrated_ids[[parameter_index]]),
        use_cache = TRUE
      )
      result_status <- if (is.null(result$status)) "cached" else result$status
      result_error <- if (is.null(result$error)) "" else result$error
      parameter_rows[[grid_index]] <- data.frame(
        parameter_id = calibrated_ids[[parameter_index]],
        grid_index = grid_index,
        parameter_value = values[[grid_index]],
        fixed_other_parameter_value = best_theta[[-parameter_index]],
        mse_iv = result$mse_iv,
        mse_oral = result$mse_oral,
        objective = result$objective,
        status = result_status,
        error = result_error,
        stringsAsFactors = FALSE
      )
    }
    parameter_table <- do.call(rbind, parameter_rows)
    profile_rows[[parameter_index]] <- parameter_table
    finite_objectives <- parameter_table$objective[
      parameter_table$status %in% c("ok", "cached") &
        is.finite(parameter_table$objective)
    ]
    deltas <- diff(finite_objectives)
    scale <- max(1, max(abs(finite_objectives)))
    tolerance <- 1e-10 * scale
    monotonicity <- if (all(deltas <= tolerance)) {
      "nonincreasing_across_locked_range"
    } else if (all(deltas >= -tolerance)) {
      "nondecreasing_across_locked_range"
    } else {
      "nonmonotonic_across_locked_range"
    }
    boundary_position <- if (
      nearly_equal(best_theta[[parameter_index]], lower[[parameter_index]])
    ) {
      "lower_bound"
    } else if (
      nearly_equal(best_theta[[parameter_index]], upper[[parameter_index]])
    ) {
      "upper_bound"
    } else {
      "interior"
    }
    profile_summaries[[parameter_index]] <- list(
      parameter_id = calibrated_ids[[parameter_index]],
      best_value = best_theta[[parameter_index]],
      lower_bound = lower[[parameter_index]],
      upper_bound = upper[[parameter_index]],
      boundary_position = boundary_position,
      monotonicity = monotonicity,
      minimum_profile_objective = min(finite_objectives)
    )
  }
  write_csv(
    do.call(rbind, profile_rows),
    file.path(output_directory, "calibration_parameter_profiles.csv")
  )
  write_json(
    list(
      schema_version = "1.0.0",
      protocol_version = config$protocol_version,
      diagnostic_protocol_version = "caffeine-calibration-diagnostic-protocol-1.0.0",
      generated_at = utc_now(),
      config_sha256 = config_sha256,
      source_run_manifest_sha256 = sha256_file(source_manifest_path),
      source_scientific_status = source_manifest$scientific_status,
      loaded_scenarios = as.list(calibration_names),
      holdout_or_external_scenarios_loaded = FALSE,
      holdout_or_external_scenarios_run = FALSE,
      best_objective = best_result$objective,
      parameter_profiles = profile_summaries,
      checks = list(
        source_is_isolated_calibration_failure = TRUE,
        calibration_scenarios_only = identical(names(simulations), calibration_names),
        best_fit_simulation_succeeded = TRUE,
        both_parameter_profiles_complete =
          nrow(do.call(rbind, profile_rows)) == 42 &&
            all(do.call(rbind, profile_rows)$status %in% c("ok", "cached"))
      ),
      artifacts = artifact_inventory(output_directory)
    ),
    file.path(output_directory, "diagnostic_manifest.json")
  )
  quit(status = 0)
}

start_grid <- do.call(rbind, lapply(config$start_grid, unlist))
if (!identical(dim(start_grid), c(9L, 2L))) {
  stop("Locked 3 x 3 calibration start grid is invalid.")
}

multi_start_rows <- list()
optim_results <- list()
for (start_index in seq_len(nrow(start_grid))) {
  initial <- as.numeric(start_grid[start_index, ])
  result <- optim(
    par = initial,
    fn = function(theta) {
      evaluate_objective(
        theta,
        start_index = start_index,
        phase = "optimization",
        use_cache = TRUE
      )$objective
    },
    method = "L-BFGS-B",
    lower = lower,
    upper = upper,
    control = list(
      maxit = 250,
      factr = 1e7,
      pgtol = 1e-8,
      parscale = c(250, 1),
      ndeps = c(1e-4, 1e-4)
    )
  )
  optim_results[[start_index]] <- result
  multi_start_rows[[start_index]] <- data.frame(
    start_index = start_index,
    initial_vmax = initial[[1]],
    initial_lipophilicity = initial[[2]],
    fitted_vmax = result$par[[1]],
    fitted_lipophilicity = result$par[[2]],
    objective = result$value,
    convergence_code = result$convergence,
    function_evaluations = unname(result$counts[["function"]]),
    gradient_evaluations = unname(result$counts[["gradient"]]),
    message = ifelse(is.null(result$message), "", result$message),
    stringsAsFactors = FALSE
  )
  write_csv(
    do.call(rbind, multi_start_rows),
    file.path(output_directory, "calibration_multi_start_results.csv")
  )
}
multi_start <- do.call(rbind, multi_start_rows)
evaluation_table <- read.csv(evaluation_path, stringsAsFactors = FALSE)
successful_evaluation_count <- sum(evaluation_table$status == "ok")
converged <- multi_start[
  multi_start$convergence_code == 0 &
    is.finite(multi_start$objective) &
    multi_start$objective < penalty_objective,
]
converged <- converged[order(converged$objective, converged$start_index), ]

if (successful_evaluation_count == 0 || nrow(converged) == 0) {
  manifest <- list(
    schema_version = "1.0.0",
    protocol_version = config$protocol_version,
    generated_at = utc_now(),
    config_sha256 = config_sha256,
    run_completed = FALSE,
    scenario_count = 2,
    calibration = list(
      start_count = nrow(start_grid),
      converged_start_count = nrow(converged),
      successful_evaluation_count = successful_evaluation_count,
      accepted = FALSE,
      checks = list(any_successful_evaluation = FALSE)
    ),
    validation = NULL,
    uncertainty = list(successful_sample_count = 0),
    scientific_status = "calibration_runtime_failed",
    registration_allowed = FALSE,
    artifacts = artifact_inventory(output_directory)
  )
  write_json(manifest, file.path(output_directory, "run_manifest.json"))
  quit(status = 2)
}

thresholds <- config$thresholds
minimum_converged <- as.integer(thresholds$minimum_converged_starts)
enough_converged <- nrow(converged) >= minimum_converged
if (nrow(converged) > 0) {
  best_row <- converged[1, ]
} else {
  best_row <- multi_start[which.min(multi_start$objective), ]
}
best_theta <- c(best_row$fitted_vmax, best_row$fitted_lipophilicity)

top_three_objective_reproducible <- FALSE
top_three_parameters_reproducible <- FALSE
if (nrow(converged) >= 3) {
  top_three <- converged[1:3, ]
  relative_objective_difference <- max(
    abs(top_three$objective - top_three$objective[[1]]) /
      max(abs(top_three$objective[[1]]), 1e-12)
  )
  vmax_relative_difference <- max(
    abs(top_three$fitted_vmax - top_three$fitted_vmax[[1]]) /
      max(abs(top_three$fitted_vmax[[1]]), 1e-12)
  )
  logp_range_difference <- (
    max(top_three$fitted_lipophilicity) -
      min(top_three$fitted_lipophilicity)
  ) / (upper[[2]] - lower[[2]])
  top_three_objective_reproducible <-
    relative_objective_difference <=
      as.numeric(thresholds$top_three_objective_relative_tolerance)
  top_three_parameters_reproducible <-
    vmax_relative_difference <=
      as.numeric(thresholds$top_three_parameter_relative_tolerance) &&
      logp_range_difference <=
      as.numeric(thresholds$top_three_parameter_relative_tolerance)
}

margin <- as.numeric(thresholds$boundary_margin_fraction) * (upper - lower)
not_near_bound <- all(best_theta > lower + margin & best_theta < upper - margin)

numeric_hessian <- function(theta) {
  step <- pmax((upper - lower) * 1e-4, 1e-6)
  dimension <- length(theta)
  output <- matrix(0, nrow = dimension, ncol = dimension)
  center <- evaluate_objective(
    theta,
    start_index = 0,
    phase = "hessian",
    use_cache = TRUE
  )$objective
  for (left in seq_len(dimension)) {
    plus <- theta
    minus <- theta
    plus[[left]] <- plus[[left]] + step[[left]]
    minus[[left]] <- minus[[left]] - step[[left]]
    f_plus <- evaluate_objective(
      plus, 0, "hessian", use_cache = TRUE
    )$objective
    f_minus <- evaluate_objective(
      minus, 0, "hessian", use_cache = TRUE
    )$objective
    output[left, left] <- (f_plus - 2 * center + f_minus) / step[[left]]^2
  }
  if (dimension > 1) {
    for (left in seq_len(dimension - 1)) {
      for (right in (left + 1):dimension) {
        pp <- pm <- mp <- mm <- theta
        pp[[left]] <- pp[[left]] + step[[left]]
        pp[[right]] <- pp[[right]] + step[[right]]
        pm[[left]] <- pm[[left]] + step[[left]]
        pm[[right]] <- pm[[right]] - step[[right]]
        mp[[left]] <- mp[[left]] - step[[left]]
        mp[[right]] <- mp[[right]] + step[[right]]
        mm[[left]] <- mm[[left]] - step[[left]]
        mm[[right]] <- mm[[right]] - step[[right]]
        value <- (
          evaluate_objective(pp, 0, "hessian", TRUE)$objective -
            evaluate_objective(pm, 0, "hessian", TRUE)$objective -
            evaluate_objective(mp, 0, "hessian", TRUE)$objective +
            evaluate_objective(mm, 0, "hessian", TRUE)$objective
        ) / (4 * step[[left]] * step[[right]])
        output[left, right] <- value
        output[right, left] <- value
      }
    }
  }
  dimnames(output) <- list(calibrated_ids, calibrated_ids)
  output
}

hessian <- if (not_near_bound) {
  numeric_hessian(best_theta)
} else {
  matrix(
    NA_real_,
    nrow = length(calibrated_ids),
    ncol = length(calibrated_ids),
    dimnames = list(calibrated_ids, calibrated_ids)
  )
}
hessian_finite <- all(is.finite(hessian))
hessian_eigenvalues <- if (hessian_finite) {
  eigen(hessian, symmetric = TRUE, only.values = TRUE)$values
} else {
  c(NA_real_, NA_real_)
}
hessian_positive <- hessian_finite && all(hessian_eigenvalues > 0)
hessian_condition <- if (hessian_positive) {
  max(hessian_eigenvalues) / min(hessian_eigenvalues)
} else {
  Inf
}
hessian_identifiable <- hessian_positive &&
  hessian_condition <= as.numeric(thresholds$hessian_condition_number_max)

best_first <- evaluate_objective(
  best_theta, 0, "repeatability_1", use_cache = FALSE
)
best_second <- evaluate_objective(
  best_theta, 0, "repeatability_2", use_cache = FALSE
)
objective_repeatable <- nearly_equal(
  best_first$objective,
  best_second$objective,
  as.numeric(thresholds$repeatability_relative_tolerance)
)
prediction_repeatable <- objective_repeatable &&
  best_first$status == "ok" && best_second$status == "ok"
if (prediction_repeatable) {
  for (name in calibration_names) {
    left <- best_first$curves[[name]]
    right <- best_second$curves[[name]]
    prediction_repeatable <- prediction_repeatable &&
      identical(left$time_min, right$time_min) &&
      all(abs(left$concentration_umol_per_l - right$concentration_umol_per_l) <=
        as.numeric(thresholds$repeatability_relative_tolerance) *
          pmax(1, abs(left$concentration_umol_per_l)))
  }
}

calibration_checks <- list(
  minimum_three_converged = enough_converged,
  top_three_objective_reproducible = top_three_objective_reproducible,
  top_three_parameters_reproducible = top_three_parameters_reproducible,
  optimum_not_near_bounds = not_near_bound,
  hessian_finite_positive_identifiable = hessian_identifiable,
  objective_repeatable = objective_repeatable,
  prediction_repeatable = prediction_repeatable
)
calibration_accepted <- all(unlist(calibration_checks))

write_csv(
  data.frame(
    parameter_id = calibrated_ids,
    fitted_value = best_theta,
    unit = vapply(estimated, function(x) x$unit, character(1)),
    lower_bound = lower,
    upper_bound = upper,
    stringsAsFactors = FALSE
  ),
  file.path(output_directory, "best_fit_parameters.csv")
)
write_csv(
  data.frame(
    row_parameter = rep(calibrated_ids, each = length(calibrated_ids)),
    column_parameter = rep(calibrated_ids, times = length(calibrated_ids)),
    value = as.vector(t(hessian)),
    stringsAsFactors = FALSE
  ),
  file.path(output_directory, "calibration_hessian.csv")
)

best_parameter_values <- parameter_values_from_theta(best_theta)
if (!calibration_accepted) {
  manifest <- list(
    schema_version = "1.0.0",
    protocol_version = config$protocol_version,
    generated_at = utc_now(),
    config_sha256 = config_sha256,
    run_completed = FALSE,
    scenario_count = 2,
    calibration = list(
      start_count = 9,
      accepted = FALSE,
      checks = calibration_checks,
      best_objective = best_row$objective,
      hessian_condition_number = hessian_condition
    ),
    validation = NULL,
    uncertainty = list(successful_sample_count = 0),
    scientific_status = ifelse(
      hessian_identifiable,
      "calibration_failed",
      "parameter_unidentifiable"
    ),
    registration_allowed = FALSE,
    artifacts = artifact_inventory(output_directory)
  )
  write_json(manifest, file.path(output_directory, "run_manifest.json"))
  quit(status = 2)
}

residual_count <- sum(vapply(calibration_observations, nrow, integer(1)))
covariance <- (
  2 * best_row$objective / (residual_count - length(best_theta))
) * solve(hessian)
dimnames(covariance) <- list(calibrated_ids, calibrated_ids)
write_csv(
  data.frame(
    row_parameter = rep(calibrated_ids, each = length(calibrated_ids)),
    column_parameter = rep(calibrated_ids, times = length(calibrated_ids)),
    value = as.vector(t(covariance)),
    stringsAsFactors = FALSE
  ),
  file.path(output_directory, "calibration_covariance.csv")
)

deterministic_curves <- run_parameter_set(
  all_two_parameter_batches,
  best_parameter_values
)
curve_rows <- list()
point_rows <- list()
for (scenario_name in expected_names) {
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
    point_rows[[length(point_rows) + 1]] <- data.frame(
      scenario_name = scenario_name,
      dataset_id = observations$dataset_id,
      split = observations$split,
      observation_id = observations$observation_id,
      time_min = observations$time_min,
      observed_umol_per_l = observations$observed_umol_per_l,
      predicted_umol_per_l = predicted,
      included = observations$use_in_objective,
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
  file.path(output_directory, "deterministic_curves.csv")
)
write_csv(point_table, file.path(output_directory, "point_predictions.csv"))

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
  file.path(output_directory, "validation_metrics.csv")
)

trapezoid_auc <- function(time_h, concentration) {
  sum(diff(time_h) * (head(concentration, -1) + tail(concentration, -1)) / 2)
}

weighted_secondary_curve <- function() {
  male <- deterministic_curves[["EXT2_PMC572_PO_MALE"]]
  female <- deterministic_curves[["EXT2_PMC572_PO_FEMALE"]]
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

secondary_predicted <- secondary_endpoints(weighted_secondary_curve())
secondary_observed <- config$secondary_metrics
names(secondary_observed) <- vapply(
  secondary_observed,
  function(x) x$endpoint,
  character(1)
)
secondary_rows <- lapply(names(secondary_predicted), function(endpoint) {
  spec <- secondary_observed[[endpoint]]
  predicted <- unname(secondary_predicted[[endpoint]])
  observed <- as.numeric(spec$observed_value)
  if (endpoint == "tmax") {
    absolute_error <- abs(predicted - observed)
    fold_error <- predicted / observed
    accepted <- absolute_error <=
      as.numeric(thresholds$tmax_absolute_error_h_max)
  } else {
    fold_error <- predicted / observed
    absolute_error <- abs(predicted - observed)
    accepted <- fold_error >=
      as.numeric(thresholds$endpoint_fold_error_min) &&
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
  file.path(output_directory, "secondary_validation_metrics.csv")
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
sampled_vmax <- truncated_normal_quantile(
  correlated_u[, 1], best_theta[[1]], fit_sd[[1]], lower[[1]], upper[[1]]
)
sampled_logp <- truncated_normal_quantile(
  correlated_u[, 2], best_theta[[2]], fit_sd[[2]], lower[[2]], upper[[2]]
)
fu_spec <- parameters[["caffeine_fraction_unbound_plasma"]]
sampled_fu <- truncated_normal_quantile(
  lhs[, 3],
  as.numeric(fu_spec$value),
  as.numeric(fu_spec$uncertainty),
  0,
  1
)
km_spec <- parameters[["caffeine_cyp1a2_km_n3_demethylation"]]
sampled_km <- as.numeric(km_spec$reported_lower) +
  lhs[, 4] * (
    as.numeric(km_spec$reported_upper) -
      as.numeric(km_spec$reported_lower)
  )
renal_values <- as.numeric(unlist(
  parameters[["caffeine_renal_plasma_clearance"]]$bootstrap_values
))
set.seed(uncertainty_seed + 1L)
sampled_renal <- vapply(seq_len(sample_count), function(index) {
  mean(sample(renal_values, length(renal_values), replace = TRUE))
}, numeric(1))
uncertainty_samples <- data.frame(
  sample_id = seq_len(sample_count),
  caffeine_cyp1a2_vmax_effective_n3 = sampled_vmax,
  caffeine_effective_lipophilicity = sampled_logp,
  caffeine_fraction_unbound_plasma = sampled_fu,
  caffeine_renal_plasma_clearance = sampled_renal,
  caffeine_cyp1a2_km_n3_demethylation = sampled_km,
  stringsAsFactors = FALSE
)
write_csv(
  uncertainty_samples,
  file.path(output_directory, "uncertainty_parameter_samples.csv")
)

uncertainty_ids <- c(
  "caffeine_cyp1a2_vmax_effective_n3",
  "caffeine_effective_lipophilicity",
  "caffeine_fraction_unbound_plasma",
  "caffeine_renal_plasma_clearance",
  "caffeine_cyp1a2_km_n3_demethylation"
)
uncertainty_paths <- parameter_paths_for(uncertainty_ids)
uncertainty_batches <- make_batches(expected_names, uncertainty_paths)
sample_curves <- list()
sample_times <- list()
successful_samples <- rep(TRUE, sample_count)
batch_size <- as.integer(config$uncertainty$batch_size)
progress_path <- file.path(output_directory, "uncertainty_progress.json")

for (scenario_name in expected_names) {
  batch <- uncertainty_batches[[scenario_name]]
  scenario_matrix <- NULL
  scenario_time <- NULL
  for (first_index in seq(1, sample_count, by = batch_size)) {
    indices <- first_index:min(first_index + batch_size - 1, sample_count)
    result_ids <- character(length(indices))
    for (offset in seq_along(indices)) {
      sample_index <- indices[[offset]]
      human_values <- c(
        sampled_vmax[[sample_index]],
        sampled_logp[[sample_index]],
        sampled_fu[[sample_index]],
        sampled_renal[[sample_index]],
        sampled_km[[sample_index]]
      )
      names(human_values) <- uncertainty_paths
      values <- batch_value_vector(batch, human_values)
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
        timestamp_utc = utc_now()
      ),
      progress_path
    )
  }
  sample_curves[[scenario_name]] <- scenario_matrix
  sample_times[[scenario_name]] <- scenario_time
}

quantile_probabilities <- as.numeric(unlist(config$uncertainty$quantiles))
quantile_rows <- list()
uncertainty_point_rows <- list()
endpoint_rows <- list()
for (scenario_name in expected_names) {
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
        included = observations$use_in_objective[[observation_index]],
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
    generic_auc <- trapezoid_auc(
      within$time_min / 60,
      within$concentration_umol_per_l
    )
    generic_cmax <- max(within$concentration_umol_per_l)
    generic_tmax <- within$time_min[[which.max(
      within$concentration_umol_per_l
    )]] / 60
    endpoint_rows[[length(endpoint_rows) + 1]] <- data.frame(
      sample_id = sample_index,
      scenario_name = scenario_name,
      dataset_id = scenarios[[scenario_name]]$dataset_id,
      auc_0_24_h_umol_h_per_l = generic_auc,
      cmax_umol_per_l = generic_cmax,
      tmax_h = generic_tmax,
      stringsAsFactors = FALSE
    )
  }
}
quantile_table <- do.call(rbind, quantile_rows)
uncertainty_point_table <- do.call(rbind, uncertainty_point_rows)
endpoint_table <- do.call(rbind, endpoint_rows)
write_csv(
  quantile_table,
  file.path(output_directory, "uncertainty_curve_quantiles.csv")
)
write_csv(
  uncertainty_point_table,
  file.path(output_directory, "uncertainty_observation_coverage.csv")
)
write_csv(
  endpoint_table,
  file.path(output_directory, "uncertainty_endpoint_samples.csv")
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
  file.path(output_directory, "secondary_uncertainty_endpoints.csv")
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
for (scenario_name in expected_names) {
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
write_csv(prcc_table, file.path(output_directory, "uncertainty_prcc.csv"))

external_coverage_rows <- uncertainty_point_table[
  uncertainty_point_table$split == "external_validation" &
    uncertainty_point_table$included,
]
external_coverage <- mean(external_coverage_rows$covered_90_percent)
uncertainty_accepted <-
  all(successful_samples) &&
  is.finite(external_coverage) &&
  external_coverage >= as.numeric(thresholds$external_coverage_min) &&
  !any(prcc_table$status == "singular")

fully_validated <- calibration_accepted &&
  internal_holdout_accepted &&
  external_deterministic_accepted &&
  uncertainty_accepted
scientific_status <- if (fully_validated) {
  "external_study_validated"
} else {
  "abstain_validation_failed"
}

manifest_without_artifacts <- list(
  schema_version = "1.0.0",
  protocol_version = config$protocol_version,
  generated_at = utc_now(),
  config_sha256 = config_sha256,
  run_completed = TRUE,
  scenario_count = length(expected_names),
  calibration = list(
    start_count = nrow(start_grid),
    converged_start_count = nrow(converged),
    accepted = calibration_accepted,
    checks = calibration_checks,
    best_objective = best_row$objective,
    best_vmax = best_theta[[1]],
    best_lipophilicity = best_theta[[2]],
    hessian_condition_number = hessian_condition,
    covariance_method = "2*objective/(n-p)*inverse(finite_difference_hessian)"
  ),
  validation = list(
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
  registration_allowed = fully_validated
)
manifest <- c(
  manifest_without_artifacts,
  list(artifacts = artifact_inventory(output_directory))
)
write_json(manifest, file.path(output_directory, "run_manifest.json"))

if (!fully_validated) {
  quit(status = 3)
}
