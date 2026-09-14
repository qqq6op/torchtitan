# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
from torch.distributed.pipelining.schedules import (
    _Action,
    _PipelineScheduleRuntime,
    FORWARD,
    FULL_BACKWARD,
    get_schedule_class,
    REDUCE_GRAD,
    RESHARD,
    UNSHARD,
)

from torchtitan.components.loss import LossFunction
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.distributed.pipeline_parallel import (
    _build_get_mesh_callback,
    _build_pipeline_schedule,
    _generate_llm_fqn_per_model_part,
    _get_pipeline_metadata,
    _get_pp_rank_to_stage_indices_mapping,
    _split_module,
)
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    trace_input_preparer_keys,
)
from torchtitan.experiments.graph_trainer.graph_pp.graph_builder import (
    GraphTrainerStageGraphProvider,
)
from torchtitan.experiments.graph_trainer.graph_pp.runner import (
    FULL_FORWARD_BACKWARD,
    GraphPipelineRuntime,
    register_graph_pp_schedule,
)
from torchtitan.experiments.graph_trainer.graph_pp.stage import GraphPipelineStage
from torchtitan.experiments.graph_trainer.registry import (
    PASS_PIPELINE_REGISTRY,
    TRACE_CALL_INPUT_PREPARERS,
    TRACE_INPUT_PREPARERS,
)
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.model_spec import ParallelizeFunction


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphPPFSDPPolicy:
    extract_fsdp_param_unshard: bool
    extract_fsdp_grad_reduction: bool


def resolve_graph_pp_fsdp_policy(
    compile_config: GraphTrainerCompileConfig,
    *,
    pp_enabled: bool,
    fsdp_enabled: bool,
) -> GraphPPFSDPPolicy:
    """Resolve topology-dependent FSDP graph boundaries.

    PP=1 keeps FSDP operations in their compute graphs by default. PP>1
    extracts them into explicit schedule actions by default.
    """
    if not fsdp_enabled:
        if compile_config.fsdp_param_unshard_mode == "extracted_in_schedule_stage":
            raise ValueError("Extracted FSDP parameter unsharding requires FSDP")
        if compile_config.fsdp_gradient_sync_mode == "deferred_as_schedule_stage":
            raise ValueError("Deferred FSDP gradient synchronization requires FSDP")
        return GraphPPFSDPPolicy(
            extract_fsdp_param_unshard=False,
            extract_fsdp_grad_reduction=False,
        )

    if not pp_enabled:
        return GraphPPFSDPPolicy(
            extract_fsdp_param_unshard=(
                compile_config.fsdp_param_unshard_mode == "extracted_in_schedule_stage"
            ),
            extract_fsdp_grad_reduction=(
                compile_config.fsdp_gradient_sync_mode == "deferred_as_schedule_stage"
            ),
        )

    if compile_config.fsdp_param_unshard_mode == "in_graph":
        raise ValueError("PP>1 GraphPP requires extracted FSDP parameter unsharding")
    if compile_config.fsdp_gradient_sync_mode == "in_graph":
        raise ValueError("PP>1 GraphPP requires deferred FSDP gradient synchronization")
    return GraphPPFSDPPolicy(
        extract_fsdp_param_unshard=True,
        extract_fsdp_grad_reduction=True,
    )


def _validate_pp1_vpp1_graph_pipeline_compile_config(
    compile_config: GraphTrainerCompileConfig,
) -> None:
    if compile_config.mode != "aot_fx_trace":
        raise ValueError("GraphPipelineRuntime requires --compile.mode aot_fx_trace")
    if compile_config.precompile_artifact_dir:
        raise ValueError(
            "GraphPipelineRuntime does not support "
            "--compile.precompile_artifact_dir yet. Existing precompiled "
            "artifacts contain one monolithic train-step graph, while the "
            "runtime requires separately bound forward, backward, and FSDP graphs."
        )


def _make_pp1_vpp1_runtime_schedule(
    stage: GraphPipelineStage,
    *,
    num_microbatches: int,
    parallelism: ParallelismConfig,
    loss_fn: LossFunction,
    fsdp_enabled: bool,
    extract_fsdp_param_unshard: bool,
    extract_fsdp_grad_reduction: bool,
    use_full_forward_backward: bool,
) -> _PipelineScheduleRuntime:
    """Build the explicit action order for one PP=1/VPP=1 stage."""
    fsdp_reshard_after_forward = (
        get_fsdp_reshard_after_forward_policy(
            parallelism.fsdp_reshard_after_forward,
            pp_enabled=False,
        )
        if fsdp_enabled
        else None
    )

    def scalar_loss_fn(*args: object, **kwargs: object) -> torch.Tensor:
        loss = loss_fn(*args, **kwargs)
        return loss[0] if isinstance(loss, tuple) else loss

    schedule = _PipelineScheduleRuntime(
        [stage],
        n_microbatches=num_microbatches,
        loss_fn=scalar_loss_fn,
        scale_grads=False,
        backward_requires_autograd=False,
    )
    reuse_unsharded_parameters = (
        extract_fsdp_param_unshard and fsdp_reshard_after_forward is False
    )
    if use_full_forward_backward:
        actions: list[_Action | None] = [
            _Action(
                0,
                cast(Any, FULL_FORWARD_BACKWARD),
                0,
                (_Action(0, FORWARD, 0), _Action(0, FULL_BACKWARD, 0)),
            )
        ]
    else:
        actions = []
        if reuse_unsharded_parameters:
            actions.append(_Action(0, UNSHARD))
        for microbatch_index in range(num_microbatches):
            if extract_fsdp_param_unshard and not reuse_unsharded_parameters:
                actions.append(_Action(0, UNSHARD))
            actions.extend(
                (
                    _Action(0, FORWARD, microbatch_index),
                    _Action(0, FULL_BACKWARD, microbatch_index),
                )
            )
            if extract_fsdp_param_unshard and not reuse_unsharded_parameters:
                actions.append(_Action(0, RESHARD))
        if extract_fsdp_grad_reduction:
            actions.append(_Action(0, REDUCE_GRAD))
        if reuse_unsharded_parameters:
            actions.append(_Action(0, RESHARD))
    schedule._prepare_schedule_with_comms({0: actions}, format="compute_comms")
    return schedule


def _validate_graph_pp_config(
    *,
    compile_config: GraphTrainerCompileConfig,
    parallelism: ParallelismConfig,
) -> None:
    if compile_config.mode != "aot_fx_trace":
        raise ValueError("GraphPP requires --compile.mode aot_fx_trace")
    if compile_config.precompile_artifact_dir:
        raise ValueError(
            "GraphPP does not support --compile.precompile_artifact_dir yet. "
            "Trace and graph construction are stage-local runtime operations."
        )
    if parallelism.fsdp_reshard_after_forward == "always":
        raise ValueError(
            "GraphPP assumes ZeRO-2 style FSDP with "
            "--parallelism.fsdp_reshard_after_forward default/never, not always."
        )
    schedule_class = get_schedule_class(parallelism.pipeline_parallel_schedule)
    if not issubclass(schedule_class, _PipelineScheduleRuntime):
        raise ValueError(
            "GraphPP currently requires a runtime PP schedule such as "
            "Interleaved1F1B, ZBVZeroBubble, or DualPipeV. "
            f"Got {parallelism.pipeline_parallel_schedule}."
        )


def make_graph_pipeline_runtime(
    stages: list[GraphPipelineStage],
    *,
    num_microbatches: int,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    model_config: BaseModel.Config | None,
    loss_fn: LossFunction,
    pass_config: Any | None,
) -> GraphPipelineRuntime:
    """Build the schedule, graph provider, and runtime for GraphTrainer.

    The inline comments describe state transitions for one stage. ``stage.state``
    is cleared after each runtime step, while live ``param.grad`` persists until
    the optimizer step. Before the first action, the runtime fills ``stage.state``
    with the live parameters, buffers, and empty gradient accumulators.

    Path 0 - PP=1 conventional execution. One runtime microbatch without
    extracted FSDP boundaries uses the joint train graph:

    FULL_FORWARD_BACKWARD(0)  # stage.output_chunks.append(loss)
                              # schedule._internal_losses.append(loss)
                              # param.grad += parameter_grads

    Path 1 - PP=1 gradient accumulation without deferred FSDP synchronization.
    The schedule contains all N accumulation microbatches:

    FORWARD(0)                            # stage.fwd_cache[0] <-
                                          #     (output, saved_values_for_backward)

    FULL_BACKWARD(0, with FSDP reduce)    # Free stage.fwd_cache[0]
                                          # stage.bwd_cache[0] <- input_grads
                                          # param.grad += reduced_param_grads

    ...

    FORWARD(N-1)                          # stage.fwd_cache[N-1] <-
                                          #     (output, saved_values_for_backward)

    FULL_BACKWARD(N-1, with FSDP reduce)  # Free stage.fwd_cache[N-1]
                                          # stage.bwd_cache[N-1] <- input_grads
                                          # param.grad += reduced_param_grads

    Path 2 - PP=1 gradient accumulation with deferred FSDP synchronization.
    This path requires FSDP and
    ``fsdp_gradient_sync_mode="deferred_as_schedule_stage"``:

    FORWARD(0)                               # stage.fwd_cache[0] <-
                                             # (output, saved_values_for_backward)

    FULL_BACKWARD(0, without FSDP reduce)    # Free stage.fwd_cache[0]
                                             # stage.bwd_cache[0] <- input_grads
                                             # stage.state.unsharded_param_grads +=
                                             #     raw_param_grads

    ...

    FORWARD(N-1)                             # stage.fwd_cache[N-1] <-
                                             # (output, saved_values_for_backward)

    FULL_BACKWARD(N-1, without FSDP reduce)  # Free stage.fwd_cache[N-1]
                                             # stage.bwd_cache[N-1] <- input_grads
                                             # stage.state.unsharded_param_grads +=
                                             #     raw_param_grads

    REDUCE_GRAD                              # stage.state.sharded_param_grads <-
                                             # reduce(unsharded_param_grads)

    Each Path 2 backward pops its forward cache and accumulates parameter
    gradients in ``stage.state.unsharded_param_grads``. ``REDUCE_GRAD`` writes
    the reduced result to ``stage.state.sharded_param_grads``. At successful
    step exit, the runtime accumulates those gradients in live ``param.grad``
    before clearing ``stage.state``.

    Every split ``FULL_BACKWARD(i)`` also moves input gradients to
    ``stage.bwd_cache[i]`` for the preceding pipeline stage. A last-stage
    ``FORWARD(i)`` additionally records its output in the schedule loss list.

    In Paths 1 and 2,
    ``fsdp_param_unshard_mode="extracted_in_schedule_stage"`` extracts FSDP
    parameter unsharding. With ``fsdp_reshard_after_forward="default"`` or
    ``"always"``, each microbatch is wrapped separately:

    UNSHARD             # stage.state.unsharded_param_values <-
                        #     unshard(stage.state.flat_param_values)

    FORWARD(0)          # stage.fwd_cache[0] <-
                        #     (output, saved_values_for_backward)

    FULL_BACKWARD(0)    # Update parameter grads as in Path 1 or Path 2

    RESHARD             # stage.state.unsharded_param_values <- []

    ...

    UNSHARD             # stage.state.unsharded_param_values <-
                        #     unshard(stage.state.flat_param_values)

    FORWARD(N-1)        # stage.fwd_cache[N-1] <-
                        #     (output, saved_values_for_backward)

    FULL_BACKWARD(N-1)  # Update parameter grads as in Path 1 or Path 2

    RESHARD             # stage.state.unsharded_param_values <- []

    REDUCE_GRAD         # Path 2: stage.state.sharded_param_grads <-
                        #     reduce(stage.state.unsharded_param_grads)

    With ``fsdp_reshard_after_forward="never"``, parameters remain unsharded
    for the complete schedule:

    UNSHARD             # stage.state.unsharded_param_values <-
                        #     unshard(stage.state.flat_param_values)

    FORWARD(0)          # stage.fwd_cache[0] <-
                        #     (output, saved_values_for_backward)

    FULL_BACKWARD(0)    # Update parameter grads as in Path 1 or Path 2

    ...

    FORWARD(N-1)        # Reuse stage.state.unsharded_param_values
                        # stage.fwd_cache[N-1] <-
                        #     (output, saved_values_for_backward)

    FULL_BACKWARD(N-1)  # Update parameter grads as in Path 1 or Path 2

    REDUCE_GRAD         # Path 2: stage.state.sharded_param_grads <-
                        #     reduce(stage.state.unsharded_param_grads)

    RESHARD             # stage.state.unsharded_param_values <- []

    Path 3 - PP>1:

    - The upstream pipeline schedule owns action ordering and communication.
    - Each stage uses separate forward, backward, and optional FSDP graphs.
    - The state transitions above apply independently to every local stage.

    All paths use the resolved FSDP boundaries to configure the same graph
    provider and ``GraphPipelineRuntime``.

    Args:
        stages: Local GraphPP stages. PP=1 requires exactly one stage.
        num_microbatches: Trainer accumulation steps for PP=1, or configured
            pipeline microbatches for PP>1.
        parallel_dims: Parallel topology used to select PP=1 or PP>1 behavior.
        parallelism: Parallel configuration used to construct the schedule.
        compile_config: GraphTrainer execution-mode configuration.
        model_config: Model configuration consumed by graph passes.
        loss_fn: Loss function used by the schedule and graph provider.
        pass_config: Full Trainer configuration required by the PP=1 joint
            graph, or ``None`` for PP>1.
    """
    if num_microbatches < 1:
        raise ValueError(
            "GraphPipelineRuntime requires at least one microbatch, got "
            f"{num_microbatches}"
        )

    pp_enabled = parallel_dims.pp_enabled
    if pp_enabled:
        _validate_graph_pp_config(
            compile_config=compile_config,
            parallelism=parallelism,
        )
    else:
        _validate_pp1_vpp1_graph_pipeline_compile_config(compile_config)
        if len(stages) != 1:
            raise ValueError(f"PP=1 requires one local stage, got {len(stages)}")

    fsdp_policy = resolve_graph_pp_fsdp_policy(
        compile_config,
        pp_enabled=pp_enabled,
        fsdp_enabled=parallel_dims.fsdp_enabled,
    )
    extract_fsdp_param_unshard = fsdp_policy.extract_fsdp_param_unshard
    extract_fsdp_grad_reduction = fsdp_policy.extract_fsdp_grad_reduction

    if pp_enabled:
        schedule = _build_pipeline_schedule(
            parallelism=parallelism,
            num_microbatches=num_microbatches,
            stages=stages,  # pyrefly: ignore [bad-argument-type]
            loss_fn=loss_fn,
            backward_requires_autograd=False,
        )
    else:
        use_full_forward_backward = (
            num_microbatches == 1
            and not extract_fsdp_param_unshard
            and not extract_fsdp_grad_reduction
        )
        if not use_full_forward_backward:
            if compile_config.ep_overlap.enabled:
                raise ValueError(
                    "GraphPipelineRuntime does not support "
                    "--compile.ep_overlap.enabled yet. GraphPP stage tracing does "
                    "not apply the EP-overlap trace-input preparers."
                )
            if compile_config.memory_policy == "sac_and_offload":
                raise ValueError(
                    "GraphPipelineRuntime does not support "
                    "--compile.memory_policy sac_and_offload yet. The GraphPP "
                    "partition must preserve offload and reload pairs across the "
                    "forward/backward boundary."
                )
            if compile_config.pass_pipeline in PASS_PIPELINE_REGISTRY:
                raise ValueError(
                    "GraphPipelineRuntime does not support custom pass pipelines yet"
                )
            trace_preparer_names = set(trace_input_preparer_keys(compile_config))
            unsupported_preparers = trace_preparer_names.intersection(
                TRACE_INPUT_PREPARERS.keys() | TRACE_CALL_INPUT_PREPARERS.keys()
            )
            if unsupported_preparers:
                raise ValueError(
                    "GraphPipelineRuntime does not support trace-input preparers yet: "
                    f"{sorted(unsupported_preparers)}"
                )

        schedule = _make_pp1_vpp1_runtime_schedule(
            stages[0],
            num_microbatches=num_microbatches,
            parallelism=parallelism,
            loss_fn=loss_fn,
            fsdp_enabled=parallel_dims.fsdp_enabled,
            extract_fsdp_param_unshard=extract_fsdp_param_unshard,
            extract_fsdp_grad_reduction=extract_fsdp_grad_reduction,
            use_full_forward_backward=use_full_forward_backward,
        )

    assert isinstance(schedule, _PipelineScheduleRuntime)
    graph_provider = GraphTrainerStageGraphProvider(
        loss_fn=loss_fn,
        compile_config=compile_config,
        model_config=model_config,
        parallelism=parallelism,
        extract_fsdp_param_unshard=extract_fsdp_param_unshard,
        extract_fsdp_grad_reduction=extract_fsdp_grad_reduction,
        pass_config=pass_config,
        parallel_dims=parallel_dims,
    )
    if pp_enabled:
        graph_provider._warn_if_cudagraph_pass_requested()
    return register_graph_pp_schedule(schedule, graph_provider=graph_provider)


def make_pp1_vpp1_graph_pipeline_runtime(
    model: nn.Module,
    *,
    gradient_accumulation_steps: int,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    device: torch.device,
    model_config: BaseModel.Config | None,
    loss_fn: LossFunction,
    trainer_config: Any,
) -> GraphPipelineRuntime:
    """Wrap one model stage and build its PP=1/VPP=1 graph runtime."""
    pp_mesh = parallel_dims.get_optional_mesh("pp", include_singleton_axes=True)
    assert pp_mesh is not None
    stage = GraphPipelineStage(
        model,
        stage_index=0,
        num_stages=1,
        device=device,
        group=pp_mesh.get_group("pp"),
    )
    return make_graph_pipeline_runtime(
        [stage],
        num_microbatches=gradient_accumulation_steps,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
        compile_config=compile_config,
        model_config=model_config,
        loss_fn=loss_fn,
        pass_config=trainer_config,
    )


def graph_pipeline_llm(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
    device: torch.device,
    model_config: BaseModel.Config,
    parallelize_fn: ParallelizeFunction,
    loss_fn: LossFunction,
) -> tuple[GraphPipelineRuntime, list[nn.Module], bool, bool]:
    """Build a GraphPP pipeline schedule for GraphTrainer.

    Args:
        model: The full model before PP stage splitting.
        parallel_dims: TorchTitan parallel dimension helper.
        training: Training config used for local batch size.
        parallelism: Parallelism config used for PP schedule and module split.
        compile_config: GraphTrainer compile config.
        ac_config: Activation checkpointing config forwarded to ``parallelize_fn``.
        dump_folder: Artifact/debug output directory.
        device: Local device for the stage.
        model_config: Model config consumed by stage graph passes.
        parallelize_fn: Model-specific SPMD parallelization function.
        loss_fn: Loss function used by upstream PP metadata and GraphPP tracing.

    Returns:
        A tuple of ``(runtime, model_parts, has_first_stage, has_last_stage)``.
    """
    pp_mesh = parallel_dims.get_mesh("pp")

    (
        num_virtual_stages,
        num_layers,
        input_weight,
        output_weight,
    ) = _get_pipeline_metadata(parallel_dims, parallelism, model_config)

    module_names_per_stage = parallelism.module_fqns_per_model_part
    if module_names_per_stage is None:
        module_names_per_stage = _generate_llm_fqn_per_model_part(
            num_virtual_stages,
            num_layers,
            input_weight,
            output_weight,
        )
    for index, stage_modules in enumerate(module_names_per_stage):
        logger.debug("GraphPP stage %s modules: %s", index, stage_modules)

    get_mesh_cb = _build_get_mesh_callback(parallel_dims)
    pp_rank_to_stage_indices = _get_pp_rank_to_stage_indices_mapping(
        pp_mesh.get_local_rank(),
        pp_mesh.size(),
        parallelism.pipeline_parallel_schedule,
        len(module_names_per_stage),
    )
    model_parts: list[nn.Module] = []
    stages: list[GraphPipelineStage] = []
    for stage_index in pp_rank_to_stage_indices:
        model_part = _split_module(model, module_names_per_stage[stage_index])
        model_part = parallelize_fn(
            model_part,
            parallel_dims=parallel_dims,
            training=training,
            parallelism=parallelism,
            compile_config=compile_config,
            ac_config=ac_config,
            dump_folder=dump_folder,
        )
        logger.info(
            "PP rank %s is building GraphPP stage_idx %s with modules %s",
            pp_mesh.get_local_rank(),
            stage_index,
            module_names_per_stage[stage_index],
        )
        model_parts.append(model_part)
        stages.append(
            GraphPipelineStage(
                model_part,
                stage_index=stage_index,
                num_stages=len(module_names_per_stage),
                device=device,
                group=pp_mesh.get_group("pp"),
                get_mesh=get_mesh_cb,
            )
        )

    graph_pipeline_runtime = make_graph_pipeline_runtime(
        stages,
        num_microbatches=parallelism.num_pp_microbatches,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
        compile_config=compile_config,
        model_config=model_config,
        loss_fn=loss_fn,
        pass_config=None,
    )

    return (
        graph_pipeline_runtime,
        model_parts,
        any(stage.is_first for stage in stages),
        any(stage.is_last for stage in stages),
    )
