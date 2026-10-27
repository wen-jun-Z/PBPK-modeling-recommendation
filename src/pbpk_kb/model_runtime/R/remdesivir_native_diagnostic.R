# Output-only diagnostic of an already selected development model.
suppressPackageStartupMessages(library(ospsuite))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2L) stop("Expected request and hash")
sha <- function(path) {
  value <- system2("sha256sum", shQuote(path), stdout=TRUE, stderr=TRUE)
  if (!is.null(attr(value,"status")) || length(value)!=1L) stop("Hash failed")
  strsplit(value[[1]],"[[:space:]]+")[[1]][[1]]
}
if (sha(args[[1]])!=args[[2]]) stop("Request hash differs")
cfg <- fromJSON(args[[1]],simplifyVector=FALSE)
if (cfg$version!="remdesivir-native-diagnostic-1" || sha(cfg$pkml_path)!=cfg$pkml_sha256) stop("Source differs")
result <- list(status="failed",actual_native_solver_calls=0L,request_sha256=args[[2]],
               source_pkml_sha256=cfg$pkml_sha256,parameter_changes=list(),saved_model=FALSE)
tryCatch({
  sim <- loadSimulation(cfg$pkml_path,loadFromCache=FALSE,addToCache=FALSE,resetIds=FALSE)
  paths <- vapply(cfg$quantities,function(x)x$path,character(1))
  if (anyDuplicated(paths) || !all(paths %in% getAllQuantityPathsIn(sim))) stop("Missing/duplicate native quantity")
  before <- vapply(getAllParametersMatching("*",sim),function(p)p$value,numeric(1))
  protocol <- lapply(cfg$protocol,function(path) {
    p <- getParameter(path,sim)
    unit <- if (endsWith(path,"|Dose")) "mg" else "min"
    list(path=path,value=as.numeric(toUnit(p,p$value,unit)),unit=unit)
  })
  names(protocol) <- vapply(cfg$protocol,function(p)tail(strsplit(p,"|",fixed=TRUE)[[1]],1),character(1))
  if (protocol$Dose$value!=150 || protocol$`Start time`$value!=0 || protocol$`Infusion time`$value!=60) stop("Not selected development dosing")
  result$administration <- protocol
  result$dose_mg <- protocol$Dose$value
  mw <- getParameter("Remdesivir|Molecular weight",sim)
  result$molecular_weight_g_mol <- as.numeric(toUnit(mw,mw$value,"g/mol"))
  setOutputs(paths,sim,stopIfNotFound=TRUE)
  after <- vapply(getAllParametersMatching("*",sim),function(p)p$value,numeric(1))
  if (!identical(before,after)) stop("Output selection changed native parameters")
  result$native_parameters_unchanged <- TRUE
  write_json(list(actual_native_solver_calls=1L,request_sha256=args[[2]]),file.path(cfg$output_dir,"solver_receipt.json"),auto_unbox=TRUE)
  result$actual_native_solver_calls <- 1L
  options <- SimulationRunOptions$new(numberOfCores=as.integer(cfg$native_cpu_cores),showProgress=FALSE)
  solved <- runSimulations(sim,simulationRunOptions=options,silentMode=TRUE,stopIfFails=TRUE)[[1]]
  values <- getOutputValues(solved,quantitiesOrPaths=paths,stopIfNotFound=TRUE,addMetaData=TRUE)
  if (values$metaData$Time$unit!="min" || length(unique(values$data$IndividualId))!=1L) stop("Time or individual mismatch")
  result$quantities <- lapply(cfg$quantities,function(spec) {
    meta <- values$metaData[[spec$path]]
    expected <- if (spec$dimension=="Amount") "µmol" else "µmol/l"
    if (meta$dimension!=spec$dimension || meta$unit!=expected) stop(paste("Unverified native output unit",spec$path))
    list(path=spec$path,kind=spec$kind,unit=meta$unit,dimension=meta$dimension)
  })
  if (any(!is.finite(as.matrix(values$data[,c("Time",paths)]))) || any(diff(values$data$Time)<=0)) stop("Nonfinite or unordered native results")
  write.csv(values$data,file.path(cfg$output_dir,"outputs.csv"),row.names=FALSE)
  result$source_unchanged <- sha(cfg$pkml_path)==cfg$pkml_sha256
  if (!result$source_unchanged || sha(args[[1]])!=args[[2]]) stop("Source/request changed")
  result$status <- "completed"
},error=function(e) result$failure <<- conditionMessage(e))
write_json(result,file.path(cfg$output_dir,"native_result.json"),auto_unbox=TRUE,pretty=TRUE,digits=17,na="null",null="null")
cat("REMDESIVIR_NATIVE_DIAGNOSTIC_FINISHED\n")
