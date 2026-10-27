# Build and execute one explicit project snapshot through the native Linux engine.
# This worker never edits an input snapshot or changes registry/scientific status.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Expected snapshot path and new output directory")
suppressPackageStartupMessages(library(ospsuite))
input <- normalizePath(args[[1]], mustWork = TRUE)
output <- normalizePath(args[[2]], mustWork = TRUE)
runSimulationsFromSnapshot(input, output = output, RunForAllOutputs = FALSE,
                           exportCSV = TRUE, exportPKML = TRUE,
                           exportJSON = FALSE, exportXML = FALSE)
pkmls <- sort(list.files(output, pattern = "\\.pkml$", recursive = TRUE, full.names = TRUE))
if (length(pkmls) == 0L) stop("Native runner exported no PKML model")
records <- list()
models <- list()
patterns <- c("Organism|**|Volume", "Organism|**|Blood flow rate", "Organism|Weight", "Organism|Age",
              "**|Molecular weight", "**|Fraction unbound (plasma, reference value)", "**|Plasma clearance")
for (pkml in pkmls) {
  simulation <- loadSimulation(pkml, loadFromCache = FALSE, addToCache = FALSE, resetIds = FALSE)
  parameters <- getAllParametersMatching(patterns, simulation)
  paths <- vapply(parameters, function(p) p$path, character(1))
  parameters <- parameters[!duplicated(paths)]
  total_matched <- length(parameters)
  parameters <- head(parameters, 512L)
  first_record <- length(records)
  for (parameter in parameters) {
    value <- tryCatch(as.numeric(parameter$value), error = function(e) NA_real_)
    finite <- length(value) == 1L && is.finite(value)
    unit <- tryCatch(parameter$unit, error = function(e) NA_character_)
    dimension <- tryCatch(parameter$dimension, error = function(e) NA_character_)
    records[[length(records) + 1L]] <- data.frame(
      model_file = basename(pkml), parameter_path = parameter$path,
      value = if (finite) value else NA_real_,
      unit = if (is.null(unit)) "" else as.character(unit),
      dimension = if (is.null(dimension)) "" else as.character(dimension),
      status = if (finite) "evaluated_native_value" else "not_evaluable",
      stringsAsFactors = FALSE)
  }
  models[[length(models) + 1L]] <- list(
    model_file = basename(pkml), simulation_name = simulation$name,
    total_parameter_count = length(getAllParameterPathsIn(simulation)),
    matched_parameter_count = total_matched,
    exported_parameter_count = length(records) - first_record,
    extraction_truncated = total_matched > 512L)
}
if (length(records) == 0L) stop("No native physiological or compound parameter values were found")
values <- do.call(rbind, records)
write.csv(values, file.path(output, "native_parameters.csv"), row.names = FALSE, na = "")
jsonlite::write_json(list(
  extraction_scope = "selected_native_physiology_and_drug_parameters_in_base_units",
  models = models, parameter_rows = nrow(values),
  finite_parameter_rows = sum(values$status == "evaluated_native_value"),
  ospsuite_version = as.character(packageVersion("ospsuite")),
  rsharp_version = as.character(packageVersion("rSharp")), r_version = R.version.string,
  scientific_accepted = FALSE
), file.path(output, "native_inventory.json"), auto_unbox = TRUE, pretty = TRUE, na = "null")
cat("BLUEPRINT_NATIVE_EXECUTION_FINISHED\n")
