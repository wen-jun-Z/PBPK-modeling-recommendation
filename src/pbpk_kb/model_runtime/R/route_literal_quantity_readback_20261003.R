# Preserve the frozen solver worker; resolve only literal-wildcard output metadata.
# No model, formula, output values, times, monitors or numerical tolerances change.
literal_quantity_metadata <- function(path, result_paths, find_quantities) {
  if (sum(result_paths == path) != 1L) stop("Exact output result path required")
  matches <- find_quantities(path)
  exact <- Filter(function(q) identical(q$path, path), matches)
  if (length(exact) != 1L) stop("One exact native output entity required")
  q <- exact[[1L]]
  if (!is.character(q$id) || length(q$id)!=1L || !nzchar(q$id)) stop("Native quantity identity required")
  unit <- q$unit; dimension <- q$dimension
  if (is.null(unit)) unit <- ""
  if (!is.character(unit) || length(unit)!=1L || !is.character(dimension) || length(dimension)!=1L)
    stop("Native entity unit and dimension required")
  list(unit=unit,dimension=dimension)
}
route_getOutputValues <- function(simulationResults, quantitiesOrPaths=NULL, population=NULL,
                                 individualIds=NULL, stopIfNotFound=TRUE, addMetaData=TRUE) {
  original <- ospsuite::getOutputValues
  if (!is.character(quantitiesOrPaths) || !isTRUE(addMetaData) || !any(grepl("*",quantitiesOrPaths,fixed=TRUE)))
    return(original(simulationResults,quantitiesOrPaths,population,individualIds,stopIfNotFound,addMetaData))
  if (!is.null(population) || !is.null(individualIds) || !isTRUE(stopIfNotFound) || anyDuplicated(quantitiesOrPaths))
    stop("Exact registered individual output request required")
  paths <- quantitiesOrPaths
  result <- original(simulationResults,paths,population,individualIds,stopIfNotFound,addMetaData=FALSE)
  ordinary <- paths[!grepl("*",paths,fixed=TRUE)]
  if (!length(ordinary)) stop("Registered ordinary concentration output required")
  ordinary_result <- original(simulationResults,ordinary,population,individualIds,stopIfNotFound,addMetaData=TRUE)
  for (path in c("IndividualId","Time",ordinary))
    if (!identical(result$data[[path]],ordinary_result$data[[path]])) stop("Original output data changed")
  metadata <- ordinary_result$metaData
  for (path in paths[grepl("*",paths,fixed=TRUE)]) {
    metadata[[path]] <- literal_quantity_metadata(path,simulationResults$allQuantityPaths,
      function(p) ospsuite::getAllQuantitiesMatching(p,simulationResults$simulation))
  }
  result$metaData <- metadata[c(paths,"Time")]
  result
}
original_worker <- file.path(getwd(),"src/pbpk_kb/model_runtime/R/recommendation_pkml_run.R")
worker_hash <- system2("sha256sum",shQuote(original_worker),stdout=TRUE,stderr=TRUE)
if (!is.null(attr(worker_hash,"status")) || length(worker_hash)!=1L ||
    strsplit(worker_hash,"[[:space:]]+")[[1L]][[1L]]!="42e2e9d09abc63eb09d4a56610a809b0d3894265306b4b2b31bcffe93e52dad1")
  stop("Original frozen native worker changed")
worker_environment <- new.env(parent=globalenv())
worker_environment$getOutputValues <- route_getOutputValues
source(original_worker,local=worker_environment)
