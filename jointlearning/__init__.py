"""Public entry point for the current COD/IML model."""
from cod_iml_evp_joint_v3.model import JointStrategyModel
from cod_iml_evp_joint_v3.optimizer import TaskAdamW

__all__ = ["JointStrategyModel", "TaskAdamW", "build_model"]


def build_model(**overrides):
    """Construct the current architecture; load generic weights separately."""
    config = dict(enhancement="evp", sharing="all_private", frequency_enabled=True,
                  frequency_fusion="sum", attention_lora_rank=8, attention_lora_alpha=8.,
                  freeze_backbone=True, private_stage_exits=True, exit_rank=4, exit_alpha=4.)
    config.update(overrides)
    return JointStrategyModel(**config)
