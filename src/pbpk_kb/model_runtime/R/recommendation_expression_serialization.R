# Frozen scalar serialization only: no model loading, ODE, observations or edits.
suppressPackageStartupMessages(library(ospsuite))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2L)stop("Frozen request and SHA required")
sha <- function(p)strsplit(system2("sha256sum",shQuote(p),stdout=TRUE),"[[:space:]]+")[[1]][[1]]
if(sha(args[[1]])!=args[[2]])stop("Request changed")
request <- fromJSON(args[[1]],simplifyVector=FALSE)
for(ref in request$serialization_runtime)if(sha(ref$path)!=ref$sha256)stop("Serializer runtime changed")
ospsuite::initPKSim()
api <- rSharp::getType("PKSim.R.Api")
flags <- rSharp::callStatic("System.Enum","Parse",rSharp::getType("System.Reflection.BindingFlags"),"NonPublic, Static")
container <- api$call("GetField","_container",flags)$call("GetValue",NULL)
methods <- rSharp::getType(container)$call("GetMethods")
resolve <- Filter(function(m)m$get("Name")=="Resolve"&&!m$get("IsGenericMethod")&&length(m$call("GetParameters"))==1L,methods)
if(length(resolve)!=1L)stop("Ambiguous resolve")
serializer <- resolve[[1]]$call("Invoke",container,list(rSharp::getType("PKSim.Core.Services.IJsonSerializer")))
dir.create(request$artifact_directory,recursive=TRUE,showWarnings=FALSE)
rows <- list()
for(i in seq_along(request$items)) {
 item <- request$items[[i]]
 dto <- rSharp::newObjectFromName("PKSim.Core.Snapshots.Parameter")
 dto$set("Name",item$path);dto$set("Value",item$value)
 if(!is.null(item$unit))dto$set("Unit",item$unit)
 if(!isTRUE(as.numeric(dto$get("Value"))==item$value))stop("DTO changed value")
 destination <- file.path(request$artifact_directory,sprintf("parameter_%03d.json",i))
 serializer$call("Serialize",dto,destination)$call("Wait")
 rows[[i]] <- list(item=item,artifact=list(path=substring(destination,nchar(request$project_root)+2L),sha256=sha(destination)))
}
if(sha(args[[1]])!=args[[2]])stop("Request changed during serialization")
for(ref in request$serialization_runtime)if(sha(ref$path)!=ref$sha256)stop("Serializer runtime changed")
write_json(list(request_sha256=args[[2]],serializer="PKSim.Core.Services.IJsonSerializer",
 serializer_assembly=rSharp::getType(serializer)$get("Assembly")$get("FullName"),parameters=rows,
 solver_calls=0L,observations_read=FALSE,parameter_values_modified=FALSE),request$result_path,
 auto_unbox=TRUE,pretty=TRUE,digits=17,null="null",na="null")
cat("EXPRESSION_PROFILE_OFFICIAL_SERIALIZATION_NO_ODE_COMPLETE\n")
