# Exact native parameter evaluation only. No ODE, setParameter*, or saveSimulation.
suppressPackageStartupMessages(library(ospsuite))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2L) stop("Expected request and hash")
sha <- function(path) {
  value <- system2("sha256sum",shQuote(path),stdout=TRUE,stderr=TRUE)
  if (!is.null(attr(value,"status")) || length(value)!=1L) stop("Hash failed")
  strsplit(value[[1]],"[[:space:]]+")[[1]][[1]]
}
if (sha(args[[1]])!=args[[2]]) stop("Request hash differs")
cfg <- fromJSON(args[[1]],simplifyVector=FALSE)
if (cfg$version!="native-perfusion-physiology-1" || sha(cfg$pkml_path)!=cfg$pkml_sha256) stop("Source differs")
if (file.exists(cfg$result_path)) stop("Result already exists")
simulation <- loadSimulation(cfg$pkml_path,loadFromCache=FALSE,addToCache=FALSE,resetIds=FALSE)
paths <- getAllParameterPathsIn(simulation)
requested <- vapply(cfg$parameters,function(x)x$path,character(1))
if (anyDuplicated(requested) || !all(requested %in% paths)) stop("Native parameter plan mismatch")
rows <- lapply(cfg$parameters,function(spec) {
  p <- getParameter(spec$path,simulation)
  native <- as.numeric(p$value)
  unit <- if (is.null(p$unit)) "" else p$unit
  if (length(native)!=1L || !is.finite(native)) stop(paste("Nonfinite native value",spec$path))
  if (spec$unit=="fraction") {
    if (!unit %in% c("","1","fraction","dimensionless")) stop("Dimensionless quantity mismatch")
    value <- native
  } else value <- as.numeric(toUnit(p,native,spec$unit))
  if (length(value)!=1L || !is.finite(value)) stop("Canonical conversion failed")
  list(path=spec$path,value=value,unit=spec$unit,native_value=native,native_unit=unit,dimension=p$dimension)
})
if (sha(cfg$pkml_path)!=cfg$pkml_sha256 || sha(args[[1]])!=args[[2]]) stop("Source changed during export")
result <- list(version=cfg$version,request_sha256=args[[2]],pkml_sha256=cfg$pkml_sha256,
               solver_calls=0L,model_modified=FALSE,parameters=rows)
write_json(result,cfg$result_path,pretty=TRUE,auto_unbox=TRUE,digits=17,na="null",null="null")
cat("NATIVE_PERFUSION_PHYSIOLOGY_EXPORTED\n")
