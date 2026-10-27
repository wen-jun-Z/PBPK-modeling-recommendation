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
opts <- rSharp::newObjectFromName("PKSim.CLI.Core.RunOptions.SnapshotRunOptions")
opts$set("InputFolder", input_folder)
opts$set("OutputFolder", output_folder)
opts$set("RunSimulations", FALSE)
opts$set("ExportMode", if (mode == "snapshot_to_project") 0L else 1L)
rSharp::callStatic("PKSim.R.Api", "RunSnapshot", opts)
if (length(list.files(output_folder, pattern=output_pattern)) != 1L)
  stop(paste("Native conversion returned without the required output:", mode))
cat("MIGRATION_STAGE_FINISHED_WITHOUT_ODE ", mode, "\n", sep=""); flush.console()
