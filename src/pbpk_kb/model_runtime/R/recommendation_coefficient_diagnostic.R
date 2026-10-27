# Execute a complete source PKML; only explicit administration values may change.
suppressPackageStartupMessages(library(ospsuite))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2L) stop("Expected request and SHA256")
sha <- function(path) {
  out <- system2("sha256sum", shQuote(path), stdout=TRUE, stderr=TRUE)
  if (!is.null(attr(out,"status")) || !length(out)) stop("Cannot hash input")
  strsplit(out[[1]], "[[:space:]]+")[[1]][[1]]
}
if (sha(args[[1]]) != args[[2]]) stop("Request checksum differs")
request <- fromJSON(args[[1]], simplifyVector=FALSE)
if (request$version != "recommendation-complete-pkml-v1") stop("Unknown request")
if (!request$action %in% c("inspect", "run")) stop("Unknown action")
if (sha(request$pkml_path) != request$pkml_sha256) stop("Original model checksum differs")
if (file.exists(request$result_path)) stop("Never replace a result")
simulation <- loadSimulation(request$pkml_path, loadFromCache=FALSE, addToCache=FALSE, resetIds=FALSE)
paths <- getAllParameterPathsIn(simulation)
containers <- getAllContainerPathsIn(simulation)
if (anyDuplicated(paths) || anyDuplicated(containers)) stop("Ambiguous native paths")
read_parameter <- function(path) {
  if (sum(paths == path) != 1L) stop("Exactly one native parameter is required")
  p <- getParameter(path, simulation)
  value <- as.numeric(p$value)
  if (length(value) != 1L || !is.finite(value)) stop(paste("Parameter is not finite:",path))
  list(path=path, id=p$id, value=value, unit=if (is.null(p$unit)) "" else p$unit,
       dimension=if (is.null(p$dimension)) "" else p$dimension)
}
values <- lapply(request$proof_parameter_requirements,function(requirement) {
  tryCatch(list(requirement=requirement,actual=read_parameter(requirement$path)),error=function(e)list(requirement=requirement,error=conditionMessage(e)))
})
write_json(values,request$result_path,auto_unbox=TRUE,pretty=TRUE,null="null",digits=NA)
