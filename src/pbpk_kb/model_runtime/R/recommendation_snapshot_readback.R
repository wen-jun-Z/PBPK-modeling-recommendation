# Original-source JSON readback: no ODE, outcomes, fitting or parameter repair.
suppressPackageStartupMessages(library(ospsuite))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2L) stop("Expected frozen request and SHA")
sha <- function(p) strsplit(system2("sha256sum",shQuote(p),stdout=TRUE),"[[:space:]]+")[[1]][[1]]
if (sha(args[[1]])!=args[[2]]) stop("Request changed")
request <- fromJSON(args[[1]],simplifyVector=FALSE)
if (sha(request$project_path)!=request$project_sha256) stop("Native project changed")
check_serialization_runtime <- function() {
  if(length(request$serialization_runtime)!=5L)stop("Pinned serialization runtime missing")
  for(ref in request$serialization_runtime)if(sha(ref$path)!=ref$sha256)stop("Serialization runtime changed")
}
check_serialization_runtime()
ospsuite::initPKSim()
opts <- rSharp::newObjectFromName("PKSim.CLI.Core.RunOptions.ExportRunOptions")
names <- rSharp::newObjectFromName("System.Collections.Generic.List`1[System.String]")
for (item in request$simulations) names$call("Add",item$name)
opts$set("ProjectFile",request$project_path); opts$set("OutputFolder",request$preload_directory)
opts$set("Simulations",names);opts$set("ExportAllSimulationsIfListIsEmpty",FALSE)
opts$set("RunSimulation",FALSE);opts$set("ExportMode",0L)
rSharp::callStatic("PKSim.R.Api","RunSimulationExport",opts)
api <- rSharp::getType("PKSim.R.Api")
flags <- rSharp::callStatic("System.Enum","Parse",rSharp::getType("System.Reflection.BindingFlags"),"NonPublic, Static")
container <- api$call("GetField","_container",flags)$call("GetValue",NULL)
invoke <- function(object,name,arguments) {
  methods <- rSharp::getType(object)$call("GetMethods")
  matches <- Filter(function(m)m$get("Name")==name && !m$get("IsGenericMethod") && length(m$call("GetParameters"))==length(arguments),methods)
  if(length(matches)!=1L)stop("Ambiguous native member")
  matches[[1]]$call("Invoke",object,arguments)
}
resolve <- function(name) invoke(container,"Resolve",list(rSharp::getType(name)))
workspace <- resolve("PKSim.Core.ICoreWorkspace");project <- workspace$get("Project")
mapper <- resolve("OSPSuite.Core.Domain.Mappers.ISimulationToModelCoreSimulationMapper")
persistor <- resolve("OSPSuite.Core.Serialization.Exchange.ISimulationPersistor")
json_serializer <- resolve("PKSim.Core.Services.IJsonSerializer")
serialization_directory <- file.path(dirname(request$result_path),"official_parameter_serialization")
dir.create(serialization_directory,showWarnings=FALSE)
serialization_index <- 0L
official_serialization <- function(parameter,actual,check) {
  # This uses the running PK-Sim serializer and its nullable-double converter,
  # not an R rounding rule or a numerical acceptance tolerance.
  declared_unit <- if(is.null(check$unit)) parameter$unit else check$unit
  display_value <- as.numeric(toUnit(parameter,actual,targetUnit=declared_unit))
  if(length(display_value)!=1L||!is.finite(display_value))stop("Nonfinite source-unit value")
  dto <- rSharp::newObjectFromName("PKSim.Core.Snapshots.Parameter")
  dto$set("Name",check$path);dto$set("Value",display_value)
  dto$set("Unit",declared_unit)
  if(!isTRUE(as.numeric(dto$get("Value"))==display_value))stop("Snapshot DTO changed native source-unit value")
  serialization_index <<- serialization_index+1L
  destination <- file.path(serialization_directory,sprintf("parameter_%06d.json",serialization_index))
  task <- json_serializer$call("Serialize",dto,destination)
  task$call("Wait")
  encoded <- fromJSON(destination,simplifyVector=FALSE)
  if(is.null(encoded$Value)||length(encoded$Value)!=1L||!is.finite(encoded$Value))stop("Official serializer returned no finite value")
  list(status="official_snapshot_serialization_audit",serializer="PKSim.Core.Services.IJsonSerializer",
       dto_type="PKSim.Core.Snapshots.Parameter",parameter_path=check$path,
       actual_native_value=actual,native_unit=parameter$unit,dimension=parameter$dimension,
       declared_unit=declared_unit,source_unit_was_unspecified=is.null(check$unit),
       source_unit_value_before_serialization=display_value,
       serialized_value=encoded$Value,source_value=check$value,
       source_value_exactly_reproduced=isTRUE(encoded$Value==check$value),
       artifact=list(path=destination,sha256=sha(destination)),
       serializer_assembly=rSharp::getType(json_serializer)$get("Assembly")$get("FullName"))
}
all_results <- list()
for (item in request$simulations) {
  sim <- invoke(project,"BuildingBlockById",list(item$simulation_id))
  if(is.null(sim)||sim$get("Name")!=item$name||is.null(sim$get("Model")))stop("Exact built simulation missing")
  original_seed <- sim$get("Settings")$get("RandomSeed")
  # JSON sources do not declare a simulation-level seed; the controlled build
  # seed was registered before either source/mapped build. Never alter a source
  # Individual/Population seed or any model parameter.
  if(!is.null(item$registered_build_seed))sim$get("Settings")$set("RandomSeed",as.integer(item$registered_build_seed))
  cfg <- rSharp::newObjectFromName("OSPSuite.Core.Domain.Builder.SimulationConfiguration")
  cfg$set("SimulationSettings",sim$get("Settings"))
  mapped <- mapper$call("MapFrom",sim,cfg,FALSE)
  if(!isTRUE(rSharp::callStatic("System.Object","ReferenceEquals",sim$get("Model"),mapped$get("Model"))) ||
     !isTRUE(rSharp::callStatic("System.Object","ReferenceEquals",sim$get("Settings"),mapped$get("Configuration")$get("SimulationSettings"))))stop("Source objects not preserved")
  transfer <- rSharp::newObjectFromName("OSPSuite.Core.Serialization.Exchange.SimulationTransfer")
  transfer$set("Simulation",mapped);persistor$call("Save",transfer,item$output_pkml)
  loaded <- loadSimulation(item$output_pkml,loadFromCache=FALSE,addToCache=FALSE,resetIds=FALSE)
  paths <- getAllParameterPathsIn(loaded)
  checks <- lapply(item$parameters,function(check) {
    count <- sum(paths==check$path)
    if(count==0L && !is.null(check$inactive_override_registration)) {
      registration <- check$inactive_override_registration
      if(check$source_scope!=paste0("simulation:",item$name) ||
         registration$simulation!=item$name || registration$native_path!=check$path ||
         registration$eligibility!="unchanged_explicit_simulation_override_only")stop("Inactive override scope mismatch")
      # This records absence, not a successful value check. Python must verify
      # identical absence in both original/restored builds and full models.
      return(list(path=check$path,status="missing_in_native_model",accepted=FALSE,
           source_value=check$value,source_unit=check$unit,
           source_record=check$source_record,source_scope=check$source_scope))
    }
    if(count!=1L)stop(paste("Explicit source parameter path missing or ambiguous:",check$path))
    p <- getParameter(check$path,loaded);actual <- as.numeric(p$value)
    expected <- as.numeric(toBaseUnit(p,check$value,check$unit))
    if(length(actual)!=1L||length(expected)!=1L||!is.finite(actual)||!is.finite(expected))stop(paste("Nonfinite explicit source value:",check$path))
    delta <- abs(actual-expected)
    audit <- if(isTRUE(actual==expected))NULL else official_serialization(p,actual,check)
    status <- if(isTRUE(actual==expected))"present_exact" else if(isTRUE(audit$source_value_exactly_reproduced))"official_snapshot_serialization_equivalent" else "present_numeric_difference"
    list(path=check$path,status=status,
         source_value=check$value,source_unit=check$unit,actual_native_value=actual,
         expected_native_value=expected,native_unit=p$unit,dimension=p$dimension,absolute_difference=delta,
         accepted=isTRUE(actual==expected),source_record=check$source_record,source_scope=check$source_scope,
         official_serialization=audit)
  })
  all_results[[length(all_results)+1L]] <- list(name=item$name,simulation_id=item$simulation_id,
    pkml_path=item$output_pkml,pkml_sha256=sha(item$output_pkml),parameter_readbacks=checks,
    all_parameters_accepted=all(vapply(checks,function(x)isTRUE(x$accepted),logical(1))),
    original_generated_simulation_seed=original_seed,registered_build_seed=item$registered_build_seed,
    same_model_object=TRUE,same_settings_object=TRUE,solver_calls=0L)
}
if(sha(args[[1]])!=args[[2]]||sha(request$project_path)!=request$project_sha256)stop("Input changed during readback")
check_serialization_runtime()
write_json(list(request_sha256=args[[2]],project_sha256=request$project_sha256,simulations=all_results,
  solver_calls=0L,observations_read=FALSE,parameter_values_modified=FALSE),request$result_path,auto_unbox=TRUE,pretty=TRUE,digits=17,null="null",na="null")
cat("SOURCE_SNAPSHOT_NATIVE_READBACK_DONE_NO_ODE\n")
