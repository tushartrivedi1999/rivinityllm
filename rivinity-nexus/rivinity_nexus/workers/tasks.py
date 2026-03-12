from rivinity_nexus.engine.distributed_launcher import LauncherType
from rivinity_nexus.engine.training import TrainingEngine, TrainingMethod, TrainingRuntimeConfig
from rivinity_nexus.workers.celery_app import celery_app


@celery_app.task(name="train_model")
def train_model(
    model_name_or_path: str,
    dataset_uri: str,
    max_steps: int = 100,
    config_overrides: dict | None = None,
) -> dict:
    config_overrides = config_overrides or {}
    method = TrainingMethod(config_overrides.get("method", TrainingMethod.supervised_fine_tuning.value))

    cfg = TrainingRuntimeConfig(
        model_name_or_path=model_name_or_path,
        dataset_uri=dataset_uri,
        output_dir=config_overrides.get("output_dir", f"/tmp/rivinity/training/{model_name_or_path}"),
        max_steps=max_steps,
        learning_rate=config_overrides.get("learning_rate", 2e-4),
        train_batch_size=config_overrides.get("train_batch_size", 1),
        gradient_accumulation_steps=config_overrides.get("gradient_accumulation_steps", 8),
        method=method,
        distributed=config_overrides.get("distributed", True),
        launcher=LauncherType(config_overrides.get("launcher", "torchrun")),
        deepspeed_enabled=config_overrides.get("deepspeed_enabled", True),
        fsdp_enabled=config_overrides.get("fsdp_enabled", False),
        gradient_checkpointing=config_overrides.get("gradient_checkpointing", True),
        mixed_precision=config_overrides.get("mixed_precision", "bf16"),
        save_steps=config_overrides.get("save_steps", 100),
        save_total_limit=config_overrides.get("save_total_limit", 3),
        lora_r=config_overrides.get("lora_r", 16),
        lora_alpha=config_overrides.get("lora_alpha", 32),
        lora_dropout=config_overrides.get("lora_dropout", 0.05),
        lora_target_modules=tuple(config_overrides.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
        qlora_4bit=config_overrides.get("qlora_4bit", True),
        qlora_quant_type=config_overrides.get("qlora_quant_type", "nf4"),
        qlora_double_quant=config_overrides.get("qlora_double_quant", True),
        cpu_offload=config_overrides.get("cpu_offload", True),
        future_rlhf_config=config_overrides.get("future_rlhf_config", {}),
    )

    engine = TrainingEngine()
    return engine.train(cfg)
