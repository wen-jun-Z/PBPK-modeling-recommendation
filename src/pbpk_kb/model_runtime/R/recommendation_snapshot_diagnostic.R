# One explicitly bounded native conversion stage; never rewrite Version.
# The Python parent freezes every source/output hash and limits total stages.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 3L) stop("Expected conversion mode, input folder and output folder")
mode <- args[[1]]
if (!mode %in% c("snapshot_to_project", "project_to_snapshot")) stop("Unknown migration stage")
input_folder <- normalizePath(args[[2]], mustWork=TRUE)
output_folder <- normalizePath(args[[3]], mustWork=TRUE)
input_pattern <- if (mode == "snapshot_to_project") "\\.json$" else "\\.pksim5$"
output_pattern <- if (mode == "snapshot_to_project") "\\.pksim5$" else "\\.json$"
if (length(list.files(input_folder, pattern=input_pattern)) != 1L) stop("One native input is required")
if (length(list.files(output_folder)) != 0L) stop("Native output folder must be empty")
cat("MIGRATION_STAGE_BEGIN ", mode, "\n", sep=""); flush.console()
suppressPackageStartupMessages(library(ospsuite))
ospsuite::initPKSim()

api <- rSharp::getType("PKSim.R.Api")
flags <- rSharp::callStatic("System.Enum","Parse",rSharp::getType("System.Reflection.BindingFlags"),"NonPublic, Static")
container <- api$call("GetField","_container",flags)$call("GetValue",NULL)
methods <- rSharp::getType(container)$call("GetMethods")
resolve_method <- Filter(function(m)m$get("Name")=="Resolve" && !m$get("IsGenericMethod") && length(m$call("GetParameters"))==1L,methods)[[1]]
task_service <- resolve_method$call("Invoke",container,list(rSharp::getType("PKSim.Core.Snapshots.Services.ISnapshotTask")))
fields <- rSharp::callStatic("System.Enum","Parse",rSharp::getType("System.Reflection.BindingFlags"),"NonPublic, Instance")
mapper <- rSharp::getType(task_service)$call("GetField","_projectMapper",fields)$call("GetValue",task_service)
logger <- rSharp::newObjectFromName("OSPSuite.R.Domain.RLogger")
fields <- rSharp::callStatic("System.Enum","Parse",rSharp::getType("System.Reflection.BindingFlags"),"NonPublic, Instance")
rSharp::getType(mapper)$call("GetField","_logger",fields)$call("SetValue",mapper,logger)

task <- task_service$call("LoadProjectFromSnapshotFileAsync",list.files(input_folder,pattern="\\.json$",full.names=TRUE)[[1]],FALSE)
project <- task$get("Result")
cat("DIRECT_SNAPSHOT_LOAD_DONE\n")
