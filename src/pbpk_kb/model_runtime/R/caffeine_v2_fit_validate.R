suppressPackageStartupMessages(library(ospsuite))
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2 || length(args) > 3) {
  stop(
    paste(
      "Usage: caffeine_v2_fit_validate.R <calibration-config.json>",
      "<preflight|calibrate|validate> [phase-output-directory]"
    )
  )
}

config_path <- normalizePath(args[[1]], mustWork = TRUE)
phase <- args[[2]]
if (!(phase %in% c("preflight", "calibrate", "validate"))) {
  stop(paste("Unknown phase:", phase))
}
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

ensure_new_directory <- function(path) {
  if (
    dir.exists(path) &&
      length(list.files(path, all.files = TRUE, no.. = TRUE)) > 0
  ) {
    stop(paste("Immutable output directory is not empty:", path))
  }
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  normalizePath(path, mustWork = TRUE)
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

if (phase == "preflight") {
  if (length(args) != 3) {
    stop("Preflight requires a separate immutable output directory.")
  }
  output_directory <- ensure_new_directory(args[[3]])
} else if (phase == "calibrate") {
  if (length(args) != 2) {
    stop("Calibration does not accept an output override.")
  }
  root_output <- config$output_directory
  if (dir.exists(root_output)) {
    stop(paste("Immutable calibration run already exists:", root_output))
  }
  dir.create(root_output, recursive = TRUE, showWarnings = FALSE)
  output_directory <- ensure_new_directory(
    file.path(root_output, "01_calibration")
  )
} else {
  stop("Validation phase is locked until the calibration implementation passes.")
}

target_output <- config$target_output
penalty_objective <- 1e12
calibration_names <- c("CAL_BLANCHARD_IV", "CAL_BLANCHARD_PO")
scenarios <- config$calibration_scenarios
names(scenarios) <- vapply(scenarios, function(x) x$name, character(1))
if (!identical(names(scenarios), calibration_names)) {
  stop("Calibration config contains a non-calibration scenario.")
}
if (any(vapply(scenarios, function(x) x$split, character(1)) != "calibration")) {
  stop("Calibration config split isolation failed.")
}

parameters <- config$parameters
names(parameters) <- vapply(parameters, function(x) x$parameter_id, character(1))
calibrated_ids <- c(
  "total_cyp1a2_specific_clearance",
  "specific_intestinal_permeability_transcellular"
)
estimated_ids <- names(parameters)[vapply(
  parameters,
  function(x) isTRUE(x$calibrate),
  logical(1)
)]
if (!identical(estimated_ids, calibrated_ids)) {
  stop("Estimated parameter identity or order drifted.")
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
      unit = spec$unit,
      calibrate = isTRUE(spec$calibrate)
    )
  }
}
mapping_paths <- vapply(mapping_rows, function(x) x$runtime_path, character(1))
if (length(mapping_rows) != 10 || anyDuplicated(mapping_paths)) {
  stop("Runtime mapping count or uniqueness drifted.")
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

nearly_equal <- function(left, right, tolerance = 1e-8) {
  abs(left - right) <= tolerance * max(1, abs(left), abs(right))
}

logical_bounds <- lapply(parameters[calibrated_ids], function(x) {
  c(as.numeric(x$lower_bound), as.numeric(x$upper_bound))
})
lower <- vapply(logical_bounds, function(x) x[[1]], numeric(1))
upper <- vapply(logical_bounds, function(x) x[[2]], numeric(1))
span <- upper - lower
if (any(!is.finite(span)) || any(span <= 0)) {
  stop("Calibrated parameter bounds are invalid.")
}
names(lower) <- names(upper) <- names(span) <- calibrated_ids

theta_from_z <- function(z) {
  values <- lower + as.numeric(z) * span
  names(values) <- calibrated_ids
  values
}

z_from_theta <- function(theta) {
  values <- (as.numeric(theta) - lower) / span
  names(values) <- calibrated_ids
  values
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

simulations <- list()
preflight_rows <- list()
for (scenario_name in calibration_names) {
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
      stop(
        paste(
          "Runtime parameter value drifted:",
          scenario_name,
          mapping$runtime_path,
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
write_csv(
  preflight_table,
  file.path(output_directory, "runtime_parameter_preflight.csv")
)

calibrated_paths <- unlist(lapply(
  parameters[calibrated_ids],
  function(x) vapply(
    x$runtime_mappings,
    function(mapping) mapping$runtime_path,
    character(1)
  )
))
if (length(calibrated_paths) != 4 || anyDuplicated(calibrated_paths)) {
  stop("Calibrated runtime path set drifted.")
}

make_batches <- function(simulation_set, parameter_paths) {
  output <- lapply(names(simulation_set), function(name) {
    createSimulationBatch(
      simulation = simulation_set[[name]],
      parametersOrPaths = parameter_paths
    )
  })
  names(output) <- names(simulation_set)
  output
}

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
    to_base_value(runtime_parameter, values_by_path[[path]], mapping$unit)
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

calibration_batches <- make_batches(simulations, calibrated_paths)
initial_logical <- vapply(
  calibrated_ids,
  function(id) as.numeric(parameters[[id]]$value),
  numeric(1)
)
names(initial_logical) <- calibrated_ids
contract_runtime_values <- runtime_values_for(initial_logical)
contract_batch <- createSimulationBatch(
  simulation = simulations[[calibration_names[[1]]]],
  parametersOrPaths = calibrated_paths
)
contract_values <- batch_value_vector(contract_batch, contract_runtime_values)
contract_result_id <- contract_batch$addRunValues(
  parameterValues = contract_values
)
contract_results <- runSimulationBatches(
  list(contract_batch),
  silentMode = TRUE,
  stopIfFails = TRUE
)
contract_curve <- curve_from_result(
  result_for_id(flatten_batch_results(contract_results), contract_result_id)
)
batch_contract <- list(
  schema_version = "1.0.0",
  protocol_version = config$protocol_version,
  generated_at = utc_now(),
  scenario_name = calibration_names[[1]],
  logical_parameter_ids = as.list(calibrated_ids),
  variable_parameter_paths = as.list(contract_batch$getVariableParameters()),
  mapped_values = as.list(contract_runtime_values),
  output_path = target_output,
  output_unit = "µmol/l",
  row_count = nrow(contract_curve),
  time_min = min(contract_curve$time_min),
  time_max = max(contract_curve$time_min),
  accepted = nrow(contract_curve) > 1 &&
    all(is.finite(contract_curve$concentration_umol_per_l)) &&
    all(contract_curve$concentration_umol_per_l >= 0)
)
write_json(batch_contract, file.path(output_directory, "batch_api_contract.json"))
if (!batch_contract$accepted) {
  stop("SimulationBatch API contract run failed.")
}

if (phase == "preflight") {
  manifest <- list(
    schema_version = "1.0.0",
    protocol_version = config$protocol_version,
    generated_at = utc_now(),
    config_sha256 = config_sha256,
    phase = phase,
    scenario_count = length(simulations),
    parameter_mapping_check_count = nrow(preflight_table),
    validation_package_loaded = FALSE,
    validation_scenarios_loaded = FALSE,
    accepted = all(preflight_table$valid) && batch_contract$accepted,
    artifacts = artifact_inventory(
      output_directory,
      excluded = "preflight_manifest.json"
    )
  )
  write_json(manifest, file.path(output_directory, "preflight_manifest.json"))
  quit(status = ifelse(manifest$accepted, 0, 2))
}

observation_frame <- function(scenario) {
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

calibration_observations <- lapply(
  calibration_names,
  function(name) observation_frame(scenarios[[name]])
)
names(calibration_observations) <- calibration_names
if (sum(vapply(calibration_observations, nrow, integer(1))) != 28) {
  stop("Calibration observation count drifted.")
}

evaluation_path <- file.path(output_directory, "calibration_evaluations.csv")
evaluation_index <- 0L
evaluation_cache <- new.env(hash = TRUE, parent = emptyenv())

evaluate_objective <- function(z, start_index, evaluation_phase, use_cache = TRUE) {
  evaluation_index <<- evaluation_index + 1L
  if (any(!is.finite(z)) || any(z < 0) || any(z > 1)) {
    return(list(
      objective = penalty_objective,
      mse_iv = penalty_objective,
      mse_oral = penalty_objective,
      curves = NULL,
      status = "outside_bounds",
      error = "normalized parameter outside [0,1]"
    ))
  }
  key <- paste(sprintf("%.17g", as.numeric(z)), collapse = "|")
  theta <- theta_from_z(z)
  runtime_values <- runtime_values_for(theta)
  if (use_cache && exists(key, envir = evaluation_cache, inherits = FALSE)) {
    cached <- get(key, envir = evaluation_cache, inherits = FALSE)
    status <- "cached"
    evaluated <- cached
  } else {
    evaluated <- tryCatch({
      curves <- run_parameter_set(calibration_batches, runtime_values)
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
    status <- evaluated$status
    if (evaluated$status == "ok") {
      assign(
        key,
        evaluated[c("objective", "mse_iv", "mse_oral")],
        envir = evaluation_cache
      )
    }
  }
  append_csv(
    data.frame(
      evaluation_index = evaluation_index,
      start_index = start_index,
      phase = evaluation_phase,
      timestamp_utc = utc_now(),
      z_total_clearance = z[[1]],
      z_permeability = z[[2]],
      total_cyp1a2_specific_clearance = theta[[1]],
      specific_intestinal_permeability_transcellular = theta[[2]],
      cl_n3_paraxanthine = runtime_values[[
        "Caffeine-CYP1A2-N3-Paraxanthine|Specific clearance"
      ]],
      cl_n1_theobromine = runtime_values[[
        "Caffeine-CYP1A2-N1-Theobromine|Specific clearance"
      ]],
      cl_n7_theophylline = runtime_values[[
        "Caffeine-CYP1A2-N7-Theophylline|Specific clearance"
      ]],
      mse_iv = evaluated$mse_iv,
      mse_oral = evaluated$mse_oral,
      objective = evaluated$objective,
      status = status,
      error = ifelse(is.null(evaluated$error), "", evaluated$error),
      stringsAsFactors = FALSE
    ),
    evaluation_path
  )
  evaluated
}

start_grid_theta <- do.call(rbind, lapply(config$start_grid, unlist))
if (!identical(dim(start_grid_theta), c(25L, 2L))) {
  stop("Locked 5 x 5 calibration start grid is invalid.")
}
start_grid_z <- t(apply(start_grid_theta, 1, z_from_theta))
if (any(start_grid_z <= 0) || any(start_grid_z >= 1)) {
  stop("Calibration starts must lie strictly inside normalized bounds.")
}

multi_start_rows <- list()
for (start_index in seq_len(nrow(start_grid_z))) {
  initial_z <- as.numeric(start_grid_z[start_index, ])
  initial_theta <- theta_from_z(initial_z)
  result <- optim(
    par = initial_z,
    fn = function(z) {
      evaluate_objective(
        z,
        start_index = start_index,
        evaluation_phase = "optimization",
        use_cache = TRUE
      )$objective
    },
    method = "L-BFGS-B",
    lower = c(0, 0),
    upper = c(1, 1),
    control = list(
      maxit = 250,
      factr = 1e7,
      pgtol = 1e-8,
      parscale = c(1, 1),
      ndeps = c(1e-4, 1e-4)
    )
  )
  fitted_theta <- theta_from_z(result$par)
  multi_start_rows[[start_index]] <- data.frame(
    start_index = start_index,
    initial_z_total_clearance = initial_z[[1]],
    initial_z_permeability = initial_z[[2]],
    initial_total_clearance = initial_theta[[1]],
    initial_permeability = initial_theta[[2]],
    fitted_z_total_clearance = result$par[[1]],
    fitted_z_permeability = result$par[[2]],
    fitted_total_clearance = fitted_theta[[1]],
    fitted_permeability = fitted_theta[[2]],
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
  failed_manifest <- list(
    schema_version = "1.0.0",
    protocol_version = config$protocol_version,
    generated_at = utc_now(),
    config_sha256 = config_sha256,
    phase = "calibrate",
    run_completed = FALSE,
    scenario_count = length(simulations),
    loaded_scenarios = as.list(names(simulations)),
    validation_package_sha256_locked = config$locked_validation_package_sha256,
    validation_package_loaded = FALSE,
    validation_scenarios_loaded = FALSE,
    calibration = list(
      start_count = nrow(start_grid_z),
      converged_start_count = nrow(converged),
      successful_evaluation_count = successful_evaluation_count,
      accepted = FALSE,
      checks = list(any_successful_converged_start = FALSE)
    ),
    validation = NULL,
    uncertainty = list(successful_sample_count = 0),
    scientific_status = "calibration_runtime_failed",
    registration_allowed = FALSE,
    artifacts = artifact_inventory(
      output_directory,
      excluded = "calibration_manifest.json"
    )
  )
  write_json(
    failed_manifest,
    file.path(output_directory, "calibration_manifest.json")
  )
  quit(status = 2)
}
if (nrow(converged) > 0) {
  best_row <- converged[1, ]
} else {
  best_row <- multi_start[which.min(multi_start$objective), ]
}
best_z <- c(
  best_row$fitted_z_total_clearance,
  best_row$fitted_z_permeability
)
best_theta <- theta_from_z(best_z)

thresholds <- config$thresholds
top_count <- as.integer(thresholds$top_run_count)
enough_converged <- nrow(converged) >=
  as.integer(thresholds$minimum_converged_starts)
top_objective_reproducible <- FALSE
top_parameters_reproducible <- FALSE
if (nrow(converged) >= top_count) {
  top <- converged[seq_len(top_count), ]
  relative_objective_difference <- max(
    abs(top$objective - top$objective[[1]]) /
      max(abs(top$objective[[1]]), 1e-12)
  )
  z_total_range <- max(top$fitted_z_total_clearance) -
    min(top$fitted_z_total_clearance)
  z_permeability_range <- max(top$fitted_z_permeability) -
    min(top$fitted_z_permeability)
  top_objective_reproducible <- relative_objective_difference <=
    as.numeric(thresholds$top_objective_relative_tolerance)
  top_parameters_reproducible <-
    z_total_range <= as.numeric(thresholds$top_parameter_relative_tolerance) &&
      z_permeability_range <=
        as.numeric(thresholds$top_parameter_relative_tolerance)
}

margin <- as.numeric(thresholds$boundary_margin_fraction)
not_near_bound <- all(best_z > margin & best_z < 1 - margin)

numeric_hessian_z <- function(z) {
  step <- rep(1e-4, length(z))
  dimension <- length(z)
  output <- matrix(0, nrow = dimension, ncol = dimension)
  center <- evaluate_objective(z, 0, "hessian", TRUE)$objective
  for (left in seq_len(dimension)) {
    plus <- minus <- z
    plus[[left]] <- plus[[left]] + step[[left]]
    minus[[left]] <- minus[[left]] - step[[left]]
    output[left, left] <- (
      evaluate_objective(plus, 0, "hessian", TRUE)$objective -
        2 * center +
        evaluate_objective(minus, 0, "hessian", TRUE)$objective
    ) / step[[left]]^2
  }
  for (left in seq_len(dimension - 1)) {
    for (right in (left + 1):dimension) {
      pp <- pm <- mp <- mm <- z
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
  dimnames(output) <- list(calibrated_ids, calibrated_ids)
  output
}

hessian_z <- if (not_near_bound) {
  numeric_hessian_z(best_z)
} else {
  matrix(
    NA_real_,
    nrow = 2,
    ncol = 2,
    dimnames = list(calibrated_ids, calibrated_ids)
  )
}
hessian_finite <- all(is.finite(hessian_z))
hessian_eigenvalues <- if (hessian_finite) {
  eigen(hessian_z, symmetric = TRUE, only.values = TRUE)$values
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

best_first <- evaluate_objective(best_z, 0, "repeatability_1", FALSE)
best_second <- evaluate_objective(best_z, 0, "repeatability_2", FALSE)
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
      all(
        abs(left$concentration_umol_per_l - right$concentration_umol_per_l) <=
          as.numeric(thresholds$repeatability_relative_tolerance) *
            pmax(1, abs(left$concentration_umol_per_l))
      )
  }
}

curve_rows <- list()
point_rows <- list()
metric_rows <- list()
for (scenario_name in calibration_names) {
  curve <- best_first$curves[[scenario_name]]
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
    included = observations$use_in_objective,
    source_artifact_id = observations$source_artifact_id,
    source_location = observations$source_location,
    stringsAsFactors = FALSE
  )
  included <- observations$use_in_objective &
    observations$observed_umol_per_l > 0 & predicted > 0
  ratios <- ratio[included]
  logs <- signed_log_residual[included]
  zero_rows <- observations$use_in_objective &
    observations$observed_umol_per_l == 0
  zero_pass <- !any(zero_rows) || all(
    predicted[zero_rows] <=
      as.numeric(thresholds$zero_baseline_max_umol_per_l)
  )
  gmfe <- exp(mean(abs(logs)))
  geometric_bias <- exp(mean(logs))
  two_fold_fraction <- mean(ratios >= 0.5 & ratios <= 2)
  accepted <- all(is.finite(predicted)) && all(predicted >= 0) &&
    zero_pass &&
    gmfe <= as.numeric(thresholds$gmfe_max) &&
    geometric_bias >= as.numeric(thresholds$geometric_bias_min) &&
    geometric_bias <= as.numeric(thresholds$geometric_bias_max) &&
    two_fold_fraction >= as.numeric(thresholds$minimum_two_fold_fraction)
  metric_rows[[length(metric_rows) + 1]] <- data.frame(
    scenario_name = scenario_name,
    dataset_id = observations$dataset_id[[1]],
    included_count = sum(observations$use_in_objective),
    gmfe = gmfe,
    geometric_bias = geometric_bias,
    log_rmse = sqrt(mean(logs^2)),
    two_fold_fraction = two_fold_fraction,
    zero_baseline_pass = zero_pass,
    accepted = accepted,
    stringsAsFactors = FALSE
  )
}
calibration_metrics <- do.call(rbind, metric_rows)
write_csv(
  do.call(rbind, curve_rows),
  file.path(output_directory, "calibration_curves.csv")
)
write_csv(
  do.call(rbind, point_rows),
  file.path(output_directory, "calibration_point_predictions.csv")
)
write_csv(
  calibration_metrics,
  file.path(output_directory, "calibration_metrics.csv")
)

profile_threshold <- best_first$objective * (
  1 + qf(
    as.numeric(thresholds$profile_confidence_level),
    df1 = 1,
    df2 = 28 - length(calibrated_ids)
  ) / (28 - length(calibrated_ids))
)

crossing_nearest_best <- function(rows, best_value, side, threshold) {
  selected <- if (side == "left") {
    rows[rows$fixed_z <= best_value, ]
  } else {
    rows[rows$fixed_z >= best_value, ]
  }
  selected <- selected[order(selected$fixed_z), ]
  if (side == "left") {
    selected <- selected[nrow(selected):1, ]
  }
  if (nrow(selected) < 2) {
    return(NA_real_)
  }
  delta <- selected$objective - threshold
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
          (threshold - left$objective) *
            (right$fixed_z - left$fixed_z) /
            (right$objective - left$objective)
      )
    }
  }
  NA_real_
}

profile_rows <- list()
profile_summaries <- list()
grid_count <- as.integer(thresholds$profile_grid_points)
for (parameter_index in seq_along(calibrated_ids)) {
  other_index <- setdiff(seq_along(calibrated_ids), parameter_index)
  grid <- seq(0, 1, length.out = grid_count)
  grid[[which.min(abs(grid - best_z[[parameter_index]]))]] <-
    best_z[[parameter_index]]
  grid <- sort(grid)
  parameter_rows <- list()
  for (grid_index in seq_along(grid)) {
    fixed_z <- grid[[grid_index]]
    result <- optim(
      par = best_z[[other_index]],
      fn = function(other_z) {
        z <- best_z
        z[[parameter_index]] <- fixed_z
        z[[other_index]] <- other_z[[1]]
        evaluate_objective(z, 0, "profile", TRUE)$objective
      },
      method = "L-BFGS-B",
      lower = 0,
      upper = 1,
      control = list(maxit = 150, factr = 1e7, pgtol = 1e-8)
    )
    z <- best_z
    z[[parameter_index]] <- fixed_z
    z[[other_index]] <- result$par[[1]]
    theta <- theta_from_z(z)
    parameter_rows[[grid_index]] <- data.frame(
      parameter_id = calibrated_ids[[parameter_index]],
      grid_index = grid_index,
      fixed_z = fixed_z,
      fixed_parameter_value = theta[[parameter_index]],
      optimized_other_z = result$par[[1]],
      optimized_other_parameter_value = theta[[other_index]],
      objective = result$value,
      threshold = profile_threshold,
      convergence_code = result$convergence,
      stringsAsFactors = FALSE
    )
  }
  table <- do.call(rbind, parameter_rows)
  profile_rows[[parameter_index]] <- table
  left_crossing <- crossing_nearest_best(
    table,
    best_z[[parameter_index]],
    "left",
    profile_threshold
  )
  right_crossing <- crossing_nearest_best(
    table,
    best_z[[parameter_index]],
    "right",
    profile_threshold
  )
  profile_summaries[[parameter_index]] <- list(
    parameter_id = calibrated_ids[[parameter_index]],
    best_z = best_z[[parameter_index]],
    best_value = best_theta[[parameter_index]],
    threshold = profile_threshold,
    left_crossing_z = left_crossing,
    right_crossing_z = right_crossing,
    left_crossing_value = ifelse(
      is.finite(left_crossing),
      theta_from_z(replace(best_z, parameter_index, left_crossing))[[
        parameter_index
      ]],
      NA_real_
    ),
    right_crossing_value = ifelse(
      is.finite(right_crossing),
      theta_from_z(replace(best_z, parameter_index, right_crossing))[[
        parameter_index
      ]],
      NA_real_
    ),
    accepted = all(is.finite(table$objective)) &&
      all(table$convergence_code == 0) &&
      is.finite(left_crossing) && is.finite(right_crossing) &&
      left_crossing > 0 && right_crossing < 1 &&
      left_crossing < best_z[[parameter_index]] &&
      right_crossing > best_z[[parameter_index]]
  )
}
profile_table <- do.call(rbind, profile_rows)
write_csv(
  profile_table,
  file.path(output_directory, "calibration_profile_likelihood.csv")
)
write_json(
  list(
    method = "21-point fixed parameter with bounded reoptimization",
    coordinate_system = "normalized_[0,1]",
    threshold_method = "RSS_min*(1+F_0.95(1,n-p)/(n-p))",
    observation_count = 28,
    parameter_count = 2,
    threshold = profile_threshold,
    parameters = profile_summaries
  ),
  file.path(output_directory, "calibration_profile_summary.json")
)
profiles_accepted <- all(vapply(
  profile_summaries,
  function(x) isTRUE(x$accepted),
  logical(1)
))

calibration_checks <- list(
  at_least_five_converged = enough_converged,
  top_five_objective_reproducible = top_objective_reproducible,
  top_five_parameters_reproducible = top_parameters_reproducible,
  optimum_not_near_bounds = not_near_bound,
  normalized_hessian_finite_positive_identifiable = hessian_identifiable,
  both_profile_likelihood_intervals_finite = profiles_accepted,
  both_calibration_arms_meet_curve_standards = all(calibration_metrics$accepted),
  objective_repeatable = objective_repeatable,
  prediction_repeatable = prediction_repeatable,
  validation_package_not_loaded = TRUE,
  validation_scenarios_not_loaded = identical(names(simulations), calibration_names)
)
calibration_accepted <- all(unlist(calibration_checks))

write_csv(
  data.frame(
    parameter_id = calibrated_ids,
    fitted_value = best_theta,
    normalized_value = best_z,
    unit = vapply(parameters[calibrated_ids], function(x) x$unit, character(1)),
    lower_bound = lower,
    upper_bound = upper,
    stringsAsFactors = FALSE
  ),
  file.path(output_directory, "best_fit_parameters.csv")
)
write_csv(
  data.frame(
    row_parameter = rep(calibrated_ids, each = 2),
    column_parameter = rep(calibrated_ids, times = 2),
    normalized_hessian_value = as.vector(t(hessian_z)),
    stringsAsFactors = FALSE
  ),
  file.path(output_directory, "calibration_hessian_normalized.csv")
)

covariance_theta <- matrix(NA_real_, nrow = 2, ncol = 2)
if (calibration_accepted) {
  covariance_z <- (
    2 * best_first$objective / (28 - length(best_z))
  ) * solve(hessian_z)
  jacobian <- diag(span)
  covariance_theta <- jacobian %*% covariance_z %*% jacobian
  dimnames(covariance_theta) <- list(calibrated_ids, calibrated_ids)
  write_csv(
    data.frame(
      row_parameter = rep(calibrated_ids, each = 2),
      column_parameter = rep(calibrated_ids, times = 2),
      value = as.vector(t(covariance_theta)),
      stringsAsFactors = FALSE
    ),
    file.path(output_directory, "calibration_covariance_original_units.csv")
  )
}

scientific_status <- if (calibration_accepted) {
  "calibration_accepted_validation_locked"
} else if (!hessian_identifiable || !profiles_accepted) {
  "parameter_unidentifiable"
} else {
  "calibration_failed"
}
manifest_without_artifacts <- list(
  schema_version = "1.0.0",
  protocol_version = config$protocol_version,
  generated_at = utc_now(),
  config_sha256 = config_sha256,
  phase = "calibrate",
  run_completed = TRUE,
  scenario_count = length(simulations),
  loaded_scenarios = as.list(names(simulations)),
  validation_package_sha256_locked = config$locked_validation_package_sha256,
  validation_package_loaded = FALSE,
  validation_scenarios_loaded = FALSE,
  calibration = list(
    start_count = nrow(start_grid_z),
    converged_start_count = nrow(converged),
    successful_evaluation_count = successful_evaluation_count,
    accepted = calibration_accepted,
    checks = calibration_checks,
    best_objective = best_first$objective,
    best_total_cyp1a2_specific_clearance = best_theta[[1]],
    best_specific_intestinal_permeability_transcellular = best_theta[[2]],
    normalized_hessian_condition_number = hessian_condition,
    profile_threshold = profile_threshold,
    covariance_method = paste(
      "normalized Hessian inverse with residual scale and",
      "bound-span Jacobian to original units"
    )
  ),
  validation = NULL,
  uncertainty = list(successful_sample_count = 0),
  scientific_status = scientific_status,
  registration_allowed = FALSE
)
manifest <- c(
  manifest_without_artifacts,
  list(
    artifacts = artifact_inventory(
      output_directory,
      excluded = "calibration_manifest.json"
    )
  )
)
write_json(
  manifest,
  file.path(output_directory, "calibration_manifest.json")
)
quit(status = ifelse(calibration_accepted, 0, 2))
