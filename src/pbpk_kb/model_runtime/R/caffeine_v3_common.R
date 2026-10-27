v3_utc_now <- function() {
  format(Sys.time(), "%Y-%m-%dT%H:%M:%OS6Z", tz = "UTC")
}

v3_sha256_file <- function(path) {
  output <- system2("sha256sum", path, stdout = TRUE, stderr = TRUE)
  status <- attr(output, "status")
  if (!is.null(status) && status != 0) {
    stop(paste("sha256sum failed for", path, paste(output, collapse = " ")))
  }
  strsplit(output[[1]], "[[:space:]]+")[[1]][[1]]
}

v3_write_json <- function(value, path) {
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

v3_write_csv <- function(value, path) {
  temporary <- paste0(path, ".tmp")
  write.csv(value, temporary, row.names = FALSE, na = "")
  if (!file.rename(temporary, path)) {
    stop(paste("Could not atomically write", path))
  }
}

v3_append_csv <- function(value, path) {
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

v3_write_rds <- function(value, path) {
  temporary <- paste0(path, ".tmp")
  saveRDS(value, temporary, version = 3, compress = "xz")
  if (!file.rename(temporary, path)) {
    stop(paste("Could not atomically write", path))
  }
}

v3_relative_path <- function(path, root) {
  normalized_path <- normalizePath(path, mustWork = TRUE)
  normalized_root <- normalizePath(root, mustWork = TRUE)
  prefix <- paste0(normalized_root, .Platform$file.sep)
  if (!startsWith(normalized_path, prefix)) {
    stop(paste("Artifact escapes output directory:", normalized_path))
  }
  substring(normalized_path, nchar(prefix) + 1)
}

v3_artifact_inventory <- function(root, excluded = character()) {
  paths <- list.files(root, recursive = TRUE, full.names = TRUE)
  paths <- paths[file.info(paths)$isdir %in% FALSE]
  paths <- paths[!(basename(paths) %in% excluded)]
  paths <- sort(paths)
  lapply(paths, function(path) {
    list(
      relative_path = v3_relative_path(path, root),
      sha256 = v3_sha256_file(path),
      byte_size = unname(file.info(path)$size)
    )
  })
}

v3_ensure_new_directory <- function(path) {
  if (
    dir.exists(path) &&
      length(list.files(path, all.files = TRUE, no.. = TRUE)) > 0
  ) {
    stop(paste("Immutable output directory is not empty:", path))
  }
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  normalizePath(path, mustWork = TRUE)
}

v3_nearly_equal <- function(left, right, tolerance = 1e-8) {
  abs(left - right) <= tolerance * max(1, abs(left), abs(right))
}

v3_normalize_unit <- function(unit) {
  switch(
    unit,
    "umol/l" = "µmol/l",
    "ml/min/kg" = "ml/min/kg",
    "mg/l" = "mg/l",
    unit
  )
}

v3_to_base_value <- function(parameter, value, unit) {
  if (unit %in% c("dimensionless", "fraction")) {
    return(as.numeric(value))
  }
  toBaseUnit(parameter, as.numeric(value), v3_normalize_unit(unit))
}

v3_mapping_rows <- function(parameters) {
  rows <- list()
  for (parameter_id in names(parameters)) {
    spec <- parameters[[parameter_id]]
    if (length(spec$runtime_mappings) == 0) {
      next
    }
    for (mapping in spec$runtime_mappings) {
      rows[[length(rows) + 1]] <- list(
        parameter_id = parameter_id,
        runtime_path = mapping$runtime_path,
        multiplier = as.numeric(mapping$multiplier),
        mapping_rule = mapping$mapping_rule,
        unit = spec$unit
      )
    }
  }
  rows
}

v3_runtime_values_for <- function(logical_values, parameters, logical_ids) {
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

v3_flatten_batch_results <- function(results) {
  unlist(results, recursive = FALSE)
}

v3_result_for_id <- function(flattened, result_id) {
  matches <- which(grepl(result_id, names(flattened), fixed = TRUE))
  if (length(matches) != 1) {
    stop(paste("Could not uniquely map SimulationBatch result", result_id))
  }
  flattened[[matches[[1]]]]
}

v3_curve_from_result <- function(result, target_output) {
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

v3_interpolate_curve <- function(curve, times_min) {
  if (
    any(times_min < min(curve$time_min)) ||
      any(times_min > max(curve$time_min))
  ) {
    stop("Requested time lies outside the simulation output grid.")
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

v3_observation_frame <- function(scenario) {
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
      exclusion_reason = ifelse(
        is.null(x$exclusion_reason),
        "",
        x$exclusion_reason
      ),
      source_artifact_id = x$source_artifact_id,
      source_location = x$source_location,
      stringsAsFactors = FALSE
    )
  }))
}

v3_trapezoid_auc_0_24 <- function(curve) {
  end_min <- 24 * 60
  if (max(curve$time_min) < end_min) {
    stop("Simulation does not cover AUC0-24h.")
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
  sum(
    diff(within$time_min / 60) *
      (head(within$concentration_umol_per_l, -1) +
        tail(within$concentration_umol_per_l, -1)) / 2
  )
}

v3_symmetric_relative_delta <- function(left, right) {
  denominator <- abs(left) + abs(right)
  ifelse(denominator == 0, 0, 2 * abs(left - right) / denominator)
}
