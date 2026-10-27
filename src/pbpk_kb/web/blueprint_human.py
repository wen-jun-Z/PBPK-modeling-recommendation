"""Rebuild a web scenario's physiology through the existing native audit chain.

Every call owns a new workspace job directory. The three native stages already
enforce two CPU threads and 180 seconds each, with output thresholds of 128,
64, and 128 MiB. The native construction solve is an engineering prerequisite;
its concentrations are not used for fitting or for the web prediction.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Callable

from pbpk_kb.blueprint_policy.perfusion_calibration import _physiology
from pbpk_kb.blueprint_policy.perfusion_export import ORGANS
from pbpk_kb.blueprints.design_runtime import _directory_bytes
from pbpk_kb.blueprints.evidence_blueprint import EvidenceBlueprintRequest
from pbpk_kb.blueprints.evidence_preserving_calibration import InfrastructurePause
from pbpk_kb.blueprints.source_workflow import inside, sha256


BUILD_TIMEOUT_SECONDS = 600
MAX_OUTPUT_BYTES = 384 * 1024**2
WORKSPACE_NATIVE_LIMIT_BYTES = 2 * 1024**3


def _native_workspace_bytes(workspace: Path, output: Path) -> int:
    """Count retained native job artifacts without traversing other resources."""
    jobs = workspace / 'blueprints'
    if jobs.is_symlink():
        raise ValueError('蓝图任务目录不能是符号链接')
    total, included = 0, False
    if jobs.exists():
        for job in jobs.iterdir():
            if job.is_symlink():
                raise ValueError('蓝图任务目录不能是符号链接')
            if not job.is_dir():
                continue
            native = job / 'human'
            if native.is_symlink():
                raise ValueError('人体输出目录不能是符号链接')
            if native.exists():
                total += _directory_bytes(native)
                included = included or native == output
    # Include a caller-owned workspace verification directory as well.
    if not included and output.exists():
        total += _directory_bytes(output)
    return total


def rebuild_physiology(
    request: dict, output: Path, root: Path,
    before_trial: Callable[[], None] | None = None,
) -> dict:
    """Return audited native physiology for the supplied human and formula.

    The caller must set ``request['partition_method']`` to the saved calibration
    formula *before* calling, and must use the same human in fitting/prediction.
    ``manifest`` in the returned document points to the newly exported JSON;
    no duplicate physiology file or frozen experimental input is written.
    """
    root = Path(root).resolve(strict=True)
    raw = Path(output)
    if not raw.is_absolute():
        raw = root / raw
    workspace = root / 'runtime/deployment/workspace'
    if '..' in raw.parts or raw == workspace or not raw.is_relative_to(workspace):
        raise ValueError('人体重建输出必须位于本项目工作台的独立任务目录')
    output = inside(raw, root)
    if not output.is_relative_to(workspace):
        raise ValueError('人体重建输出不能通过链接离开工作台目录')
    if output.exists():
        raise ValueError('人体重建需要新的任务目录，不能复用旧人体结果')

    req = EvidenceBlueprintRequest.model_validate(request).model_dump(mode='json', exclude_none=True)
    if req['dosing']['route'] != 'intravenous':
        raise ValueError('当前人体重建仅支持成人单药单次静脉给药')
    started = time.monotonic()

    def guard():
        if before_trial is not None:
            before_trial()
        if time.monotonic() - started > BUILD_TIMEOUT_SECONDS:
            raise TimeoutError('人体重建超过 600 秒时限，任务已停止且不会自动重试')
        # Native writers independently check their bounded directory every
        # 0.2 seconds; this adds a combined check between stages and at return.
        if output.exists() and _directory_bytes(output) > MAX_OUTPUT_BYTES:
            raise ValueError('人体重建输出超过大小上限，任务已停止')
        if _native_workspace_bytes(workspace, output) > WORKSPACE_NATIVE_LIMIT_BYTES:
            raise ValueError('人体任务文件已达到 2 GiB 上限，请先下载并整理已保存的任务后再生成；系统不会自动删除旧结果')

    guard()
    try:
        physiology = _physiology(req, output, root, guard)
    except InfrastructurePause as exc:
        raise ValueError('人体原生构建或校验失败，未使用旧人体替代；任务诊断已保留') from exc
    guard()

    # Bind returned physiology to this exact new native execution, not an
    # arbitrary cached or substituted document with a matching display weight.
    completion = json.loads((output / 'base/completion.json').read_text())
    if completion['request'] != req or completion.get('actual_native_solver_calls') != 1:
        raise ValueError('人体重建记录与提交条件不一致')
    reference = completion['execution_manifest']
    manifest_path = inside(reference['path'], root, exists=True)
    if (manifest_path.parent != output / 'base/native'
            or sha256(manifest_path) != reference['sha256']
            or physiology.get('source_execution_manifest_sha256') != reference['sha256']):
        raise ValueError('新人体的原生执行来源校验失败')
    execution = json.loads(manifest_path.read_text())
    snapshot_path = manifest_path.parent / 'snapshot.json'
    if (execution.get('engineering_accepted') is not True
            or sha256(snapshot_path) != execution.get('source_sha256')):
        raise ValueError('新人体原生执行或快照校验失败')
    snapshot = json.loads(snapshot_path.read_text())
    human = req['human']
    expected_origin = {
        'Species': 'Human', 'Population': human['population'], 'Gender': human['sex'],
        'Age': {'Value': human['age_years'], 'Unit': 'year(s)'},
        'Weight': {'Value': human['weight_kg'], 'Unit': 'kg'},
        'Height': {'Value': human['height_cm'], 'Unit': 'cm'},
    }
    individuals = snapshot.get('Individuals', [])
    if (len(individuals) != 1 or individuals[0].get('OriginData') != expected_origin
            or individuals[0].get('Seed') != human['seed']
            or snapshot['Simulations'][0].get('Individual') != individuals[0].get('Name')):
        raise ValueError('原生人体条件与提交的年龄、性别、身高、体重不一致')
    method = 'PK-Sim Standard' if req['partition_method'] == 'PKSimStandard' else 'Rodgers and Rowland'
    expected_method = 'Cellular partition coefficient method - ' + method
    if expected_method not in snapshot['Compounds'][0].get('CalculationMethods', []):
        raise ValueError('新人体的分配方法与已保存的校准方案不一致')
    if physiology.get('compound') != req['compound_name']:
        raise ValueError('新人体原生读回的药物与提交条件不一致')
    # Native parameter serialization introduces small floating-point differences;
    # use the same relative tolerance as the existing native physiology audits.
    if not math.isclose(float(physiology['weight_kg']), human['weight_kg'], rel_tol=1e-6):
        raise ValueError('新人体原生读回的体重与提交条件不一致')
    if (len(physiology['organs']) != len(ORGANS)
            or {organ['name'] for organ in physiology['organs']} != set(ORGANS)):
        raise ValueError('新人体原生读回的器官范围不一致')
    exported = physiology['manifest']
    exported_path = inside(exported['path'], root, exists=True)
    if (exported_path != output / 'physiology/perfusion_physiology.json'
            or sha256(exported_path) != exported['sha256']):
        raise ValueError('新人体生理参数导出校验失败')
    guard()
    return physiology
