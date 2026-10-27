# Serialize the already saved model with the official core mapper. No snapshot
# reconstruction, simulation run, population sampling, or result loading.
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=4L) stop("Expected isolated project, simulation ID, exact name, output PKML")
suppressPackageStartupMessages(library(ospsuite))
suppressPackageStartupMessages(library(jsonlite))
if (file.exists(args[[4]])) stop("Output already exists")
ospsuite::initPKSim()
# ExportMode=0 asks the public native runner to load exactly this saved model
# without invoking the UI/MoBi export guard or rebuilding its configuration.
# The Python-created project has no observation/result rows or references.
preload <- file.path(dirname(args[[4]]),"preload")
if (dir.exists(preload)) stop("Preload folder already exists")
dir.create(preload)
opts <- rSharp::newObjectFromName("PKSim.CLI.Core.RunOptions.ExportRunOptions")
names <- rSharp::newObjectFromName("System.Collections.Generic.List`1[System.String]")
names$call("Add",args[[3]])
opts$set("ProjectFile",normalizePath(args[[1]],mustWork=TRUE))
opts$set("OutputFolder",preload)
opts$set("Simulations",names)
opts$set("ExportAllSimulationsIfListIsEmpty",FALSE)
opts$set("RunSimulation",FALSE)
opts$set("ExportMode",0L)
rSharp::callStatic("PKSim.R.Api","RunSimulationExport",opts)
api <- rSharp::getType("PKSim.R.Api")
flags <- rSharp::callStatic("System.Enum","Parse",rSharp::getType("System.Reflection.BindingFlags"),"NonPublic, Static")
field <- api$call("GetField","_container",flags)
container <- field$call("GetValue",NULL)
invoke_nongeneric <- function(object,name,arguments) {
  methods <- rSharp::getType(object)$call("GetMethods")
  matches <- Filter(function(m) m$get("Name")==name && !m$get("IsGenericMethod") &&
                      length(m$call("GetParameters"))==length(arguments),methods)
  if (length(matches)!=1L) stop(paste("Ambiguous native method",name,length(matches)))
  matches[[1]]$call("Invoke",object,arguments)
}
resolve <- function(name) invoke_nongeneric(container,"Resolve",list(rSharp::getType(name)))
workspace <- resolve("PKSim.Core.ICoreWorkspace")
persistor <- resolve("PKSim.Core.Services.IWorkspacePersistor")
on.exit(persistor$call("CloseSession"),add=TRUE)
project <- workspace$get("Project")
simulation <- invoke_nongeneric(project,"BuildingBlockById",list(args[[2]]))
if (is.null(simulation) || simulation$get("Name")!=args[[3]]) stop("Exact original simulation not found")
if (is.null(simulation$get("Model"))) stop("Saved simulation model was not loaded")
mapper <- resolve("OSPSuite.Core.Domain.Mappers.ISimulationToModelCoreSimulationMapper")
configuration <- rSharp::newObjectFromName("OSPSuite.Core.Domain.Builder.SimulationConfiguration")
configuration$set("SimulationSettings",simulation$get("Settings"))
mapped <- mapper$call("MapFrom",simulation,configuration,FALSE)
same <- rSharp::callStatic("System.Object","ReferenceEquals",simulation$get("Model"),mapped$get("Model"))
if (!isTRUE(same)) stop("Core mapper did not preserve the saved model object")
same_settings <- rSharp::callStatic("System.Object","ReferenceEquals",simulation$get("Settings"),
                                      mapped$get("Configuration")$get("SimulationSettings"))
if (!isTRUE(same_settings)) stop("Core mapper did not preserve the saved settings object")
transfer <- rSharp::newObjectFromName("OSPSuite.Core.Serialization.Exchange.SimulationTransfer")
transfer$set("Simulation",mapped)
core_persistor <- resolve("OSPSuite.Core.Serialization.Exchange.ISimulationPersistor")
core_persistor$call("Save",transfer,args[[4]])
write_json(list(simulation_id=args[[2]],simulation_name=args[[3]],same_model_object=TRUE,same_settings_object=TRUE,
  model_rebuilt=FALSE,prior_result_payloads_accessed=FALSE,observations_loaded=FALSE,solver_calls=0L,
  native_preload="ExportMode=0 against isolated project with all prior outcome tables cleared",
  mapper="MapFrom(savedSimulation, configuration retaining original Settings, shouldCloneModel=false)",
  imported_flag_changed=FALSE),paste0(args[[4]],".native.json"),auto_unbox=TRUE,pretty=TRUE)
cat("SAVED_MODEL_CORE_PKML_EXPORTED_WITHOUT_REBUILD_OR_ODE\n")
