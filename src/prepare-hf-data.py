import pandas as pd
from config import Config
from utils import build_hf_dataset_from_pandas

if __name__ == "__main__":
    configobj = Config()
    pd_df = pd.read_parquet(configobj.data_dir / "kw_raw.parquet")
    kw_ds = build_hf_dataset_from_pandas(pd_df)
    print(f"Number of training samples: {len(kw_ds['train'])}")
    print(f"Number of validation samples: {len(kw_ds['validation'])}")
    print(f"Number of test samples: {len(kw_ds['test'])}")
    kw_ds.save_to_disk(configobj.data_dir / 'keyword_dataset')