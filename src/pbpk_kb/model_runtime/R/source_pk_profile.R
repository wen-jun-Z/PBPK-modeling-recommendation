# Read complete PKML and existing results; never call an ODE solver or mutate inputs.
suppressPackageStartupMessages(library(ospsuite))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2L) stop("Expected request and SHA256")
sha <- function(path) {
  output <- system2("sha256sum", shQuote(path), stdout=TRUE, stderr=TRUE)
  if (!is.null(attr(output,"status")) || !length(output)) stop("Cannot hash input")
  strsplit(output[[1]], "[[:space:]]+")[[1]][[1]]
}
if (sha(args[[1]]) != args[[2]]) stop("Request hash changed")
request <- fromJSON(args[[1]], simplifyVector=FALSE)
if (request$version != "source-pk-profile-v1") stop("Unknown request version")
if (sha(request$pkml_path) != request$pkml_sha256 || sha(request$csv_path) != request$csv_sha256) stop("Artifact hash changed")
if (file.exists(request$result_path)) stop("Output already exists")
simulation <- loadSimulation(request$pkml_path, loadFromCache=FALSE, addToCache=FALSE, resetIds=FALSE)
paths <- getAllParameterPathsIn(simulation)
parameters <- lapply(request$parameters, function(spec) {
  result <- list(path=spec$path, requested_unit=spec$unit, status="failed", value=NULL,
                 native_value=NULL, native_unit=NULL, dimension=NULL, error=NULL)
  tryCatch({
    if (sum(paths == spec$path) != 1L) stop("Native path missing or ambiguous")
    p <- getParameter(spec$path, simulation)
    value <- as.numeric(p$value)
    if (length(value)!=1L || !is.finite(value)) stop("Nonfinite native parameter")
    result$native_value <- value
    result$native_unit <- if (is.null(p$unit)) "" else p$unit
    result$dimension <- p$dimension
    if (spec$unit == "fraction") {
      if (!result$native_unit %in% c("", "1", "fraction", "dimensionless")) stop("Not dimensionless")
      result$value <- value
    } else {
      factor <- toBaseUnit(p, 1.0, spec$unit)
      if (length(factor)!=1L || !is.finite(factor) || factor<=0) stop("Unit conversion failed")
      result$value <- value / factor
    }
    result$status <- "completed"
    result
  }, error=function(e) {result$error <- conditionMessage(e); result})
})
analyses <- tryCatch({
  imported <- importResultsFromCSV(simulation, request$csv_path)
  pk <- calculatePKAnalyses(imported)
  rows <- lapply(pk$allPKParametersFor(request$target_output), function(parameter) {
    list(name=parameter$name, values=as.list(as.numeric(parameter$values)),
         unit=parameter$unit, dimension=parameter$dimension)
  })
  list(status="completed", parameters=rows, error=NULL)
}, error=function(e) list(status="failed", parameters=list(), error=conditionMessage(e)))
if (sha(request$pkml_path)!=request$pkml_sha256 || sha(request$csv_path)!=request$csv_sha256 || sha(args[[1]])!=args[[2]]) stop("Inputs changed during readback")
result <- list(version=request$version, request_sha256=args[[2]], pkml_sha256=request$pkml_sha256,
               csv_sha256=request$csv_sha256, target_output=request$target_output,
               ospsuite_version=as.character(packageVersion("ospsuite")), solver_calls=0L,
               model_modified=FALSE, parameters=parameters, nca=analyses)
write_json(result, request$result_path, auto_unbox=TRUE, pretty=TRUE, digits=17, na="null", null="null")
cat("SOURCE_PK_PROFILE_READBACK_FINISHED\n")
