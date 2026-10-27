# Preserve an imported PK-Sim model using its native object and native serializer.
# Input MUST be the isolated outcome-free working copy prepared by source_pkml.py.
# No snapshot, configuration reconstruction, parameter change, or ODE execution.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop('Expected model-only project, simulation id, empty output directory')
suppressPackageStartupMessages(library(ospsuite))
input <- normalizePath(args[[1]], mustWork = TRUE)
output <- normalizePath(args[[3]], mustWork = TRUE)
if (length(list.files(output, all.files = TRUE, no.. = TRUE))) stop('Output must be empty')
ospsuite::initPKSim()

# Api exposes only CLI runners. Inspect its own initialized DI container, as
# defined by the pinned PKSim.R/Api.cs, to call the native persistence services.
# This neither accesses another process nor changes the original source file.
flags <- rSharp::callStatic('System.Enum', 'Parse',
                            rSharp::getType('System.Reflection.BindingFlags'), 'NonPublic, Static')
field <- rSharp::getType('PKSim.R.Api')$call('GetField', '_container', flags)
container <- field$call('GetValue', NULL)
resolve <- function(name) container$call('Resolve', rSharp::getType(name))
workspace <- resolve('PKSim.Core.ICoreWorkspace')
workspacePersistor <- resolve('PKSim.Core.Services.IWorkspacePersistor')
workspacePersistor$call('LoadSession', workspace, input)
project <- workspace$get('Project')
simulation <- project$call('BuildingBlockById', args[[2]])
if (is.null(simulation)) stop('Exact saved simulation id missing')
lazy <- resolve('PKSim.Core.Services.ILazyLoadTask')
lazy$call('Load', simulation)
# Deliberately never call LoadResults.
originalImported <- simulation$get('IsImported')
model <- simulation$get('Model')
modelId <- model$get('Id')
settings <- simulation$get('Settings')
settingsId <- settings$get('Id')
mapper <- resolve('OSPSuite.Core.Domain.Mappers.ISimulationToModelCoreSimulationMapper')
mapped <- mapper$call('MapFrom', simulation, FALSE)
if (!isTRUE(rSharp::callStatic('System.Object', 'ReferenceEquals', model, mapped$get('Model'))))
  stop('Native mapper did not retain the exact saved Model object')
if (!identical(model$get('Id'), modelId) || !identical(simulation$get('IsImported'), originalImported))
  stop('Source model identity/imported provenance changed')
transfer <- rSharp::newObjectFromName('OSPSuite.Core.Serialization.Exchange.SimulationTransfer')
transfer$set('Simulation', mapped)
persistor <- resolve('OSPSuite.Core.Serialization.Exchange.ISimulationPersistor')
target <- file.path(output, 'preserved_native_model.pkml')
persistor$call('Save', transfer, target)
if (!file.exists(target) || file.info(target)$size == 0) stop('Native serializer produced no PKML')
if (!identical(model$get('Id'), modelId) || !identical(simulation$get('Settings')$get('Id'), settingsId) ||
    !identical(simulation$get('IsImported'), originalImported)) stop('Original native model/settings/imported identity changed')
jsonlite::write_json(list(
  status = 'native_model_serialized', source_simulation_id = args[[2]],
  source_simulation_name = simulation$get('Name'), source_model_id = modelId,
  source_settings_id = settingsId, source_is_imported_before = originalImported,
  source_is_imported_after = simulation$get('IsImported'), exact_model_object_retained = TRUE,
  native_model_rebuilt = FALSE, source_parameters_modified = FALSE,
  observed_data_loaded = FALSE, old_results_loaded = FALSE, ode_calls = 0L,
  population_covariates_applied = FALSE,
  limitations = c('Preserved native reference instance; original population distribution is not executed.',
                  'Native parameter/output readback and solver validation remain required.')
), file.path(output, 'native_serialization_audit.json'), auto_unbox = TRUE, pretty = TRUE)
cat('IMPORTED_MODEL_NATIVE_SERIALIZATION_COMPLETED_WITHOUT_ODE\n')
