# Read-only load probe for an exactly preserved saved simulation model.
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2L) stop("Expected preserved PKML and new inventory JSON")
suppressPackageStartupMessages(library(ospsuite))
suppressPackageStartupMessages(library(jsonlite))
if (file.exists(args[[2]])) stop("Inventory already exists")
simulation <- loadSimulation(args[[1]], loadFromCache=FALSE, addToCache=FALSE, resetIds=FALSE)
paths <- getAllParameterPathsIn(simulation)
containers <- getAllContainerPathsIn(simulation)
if (length(paths)>200000L || length(containers)>200000L) stop("Inventory bound exceeded")
selected <- paths[grepl("(^Events\\||^Applications\\||Molecular weight$|Use pH-|Reference concentration$)", paths)]
rows <- lapply(selected, function(path) {
  p <- getParameter(path, simulation)
  list(path=path, value=tryCatch(as.numeric(p$value),error=function(e) NULL), unit=p$unit, dimension=p$dimension)
})
write_json(list(simulation_name=simulation$name,parameter_count=length(paths),container_count=length(containers),
  parameter_paths=as.list(paths),container_paths=as.list(containers),selected_parameters=rows,
  solver_calls=0L,model_modified=FALSE),args[[2]],auto_unbox=TRUE,pretty=TRUE,digits=17,null="null")
cat("PRESERVED_MODEL_NATIVE_LOAD_FINISHED_WITHOUT_ODE\n")
