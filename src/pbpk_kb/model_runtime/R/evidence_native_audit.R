# Read-only bounded inspection of a newly evidence-built PKML. No ODE calls.
suppressPackageStartupMessages(library(ospsuite))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2L) stop("Expected request path and checksum")
sha256 <- function(path) {
  result <- system2("sha256sum", shQuote(path), stdout=TRUE, stderr=TRUE)
  status <- attr(result, "status")
  if ((!is.null(status) && status != 0) || length(result) != 1L) stop("Hash failed")
  strsplit(result[[1]], "[[:space:]]+")[[1]][[1]]
}
request_path <- normalizePath(args[[1]], mustWork=TRUE)
if (sha256(request_path) != args[[2]]) stop("Request checksum mismatch")
request <- fromJSON(request_path, simplifyVector=FALSE)
if (request$version != "evidence-native-input-capacity-audit-v1") stop("Unsupported audit version")
if (sha256(request$pkml_path) != request$pkml_sha256) stop("PKML checksum mismatch")
if (file.exists(request$result_path) || length(request$parameters) < 1L || length(request$parameters) > 128L) stop("Invalid output/parameter count")
simulation <- loadSimulation(request$pkml_path, loadFromCache=FALSE, addToCache=FALSE, resetIds=FALSE)
paths <- getAllParameterPathsIn(simulation)
if (anyDuplicated(paths)) stop("Ambiguous native paths")
rows <- lapply(request$parameters, function(spec) {
  row <- list(path=spec$path, unit=spec$unit, status="failed", value=NULL, error=NULL)
  tryCatch({
    if (sum(paths == spec$path) != 1L) stop("Requested native parameter is missing")
    p <- getParameter(spec$path, simulation)
    value <- as.numeric(p$value)
    if (length(value) != 1L || !is.finite(value)) stop("Native parameter is not finite")
    if (spec$unit == "fraction") {
      if (!is.null(p$unit) && !p$unit %in% c("", "fraction", "dimensionless", "1")) stop("Dimensionless request targets a dimensional value")
    } else {
      value <- as.numeric(toUnit(p, value, spec$unit))
    }
    if (length(value) != 1L || !is.finite(value)) stop("Requested unit conversion failed")
    row$value <- value
    row$status <- "evaluated"
    row
  }, error=function(e) {row$error <- conditionMessage(e); row})
})
if (sha256(request$pkml_path) != request$pkml_sha256 || sha256(request_path) != args[[2]]) stop("Audit inputs changed")
result <- list(request_sha256=args[[2]], pkml_sha256=request$pkml_sha256, parameters=rows,
               solver_calls=0L, model_modified=FALSE, ospsuite_version=as.character(packageVersion("ospsuite")))
temporary <- paste0(request$result_path, ".tmp")
write_json(result, temporary, auto_unbox=TRUE, pretty=TRUE, digits=17, null="null", na="null")
if (!file.rename(temporary, request$result_path)) stop("Could not persist audit values")
