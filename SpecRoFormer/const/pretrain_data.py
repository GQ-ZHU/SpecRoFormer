from pathlib import Path

PRETRAIN_ROOT = Path(__file__).resolve().parents[2] / "data" / "PreTrained"

PRETRAIN_DATASETS = {
    # "3M_eng0_posh": PRETRAIN_ROOT / "3M_eng0_posh_cleaned.npy",
    "3M_eng1_posh": PRETRAIN_ROOT / "3M_eng1_posh_cleaned.npy",
    # "3M_eng2_posh": PRETRAIN_ROOT / "3M_eng2_posh_cleaned.npy",
}
