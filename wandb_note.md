# 强制重新下载全部
python scripts/download_wandb_visuals.py xxxyyymmm/wge_drift --force

# 只同步单个 run
python scripts/download_wandb_visuals.py xxxyyymmm/qam-reproduce/3assknrg

# 额外写完整 history.csv，默认不写，避免太大
python scripts/download_wandb_visuals.py xxxyyymmm/wge_drift --write-csv

# 如果哪天你又想顺手下媒体文件
python scripts/download_wandb_visuals.py xxxyyymmm/wge_drift --download-files