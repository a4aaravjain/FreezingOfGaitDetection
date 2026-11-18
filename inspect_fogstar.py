"""Read-only FoG-STAR inventory; no training or automatic axis remapping."""
from pathlib import Path
import pandas as pd

def main():
    data=pd.read_csv(Path(__file__).resolve().parent/"sensor_data.csv")
    print("Shape:",data.shape)
    print("Columns:",data.columns.tolist())
    print("Missing values:",data.isna().sum().to_dict())
    print("Labels:",data.fog.value_counts().to_dict())
    print("Subjects:",sorted(data.subjectID.unique().tolist()))
    delta=data.groupby(["subjectID","sessionID","taskID"],sort=False).timestamp.diff()
    print("Within-recording timestamp steps:",delta.describe().to_dict())
    print("Local median step is approximately 1/60, consistent with seconds.")
    print("Do not infer units from the repository README alone; document the discrepancy.")

if __name__=="__main__":
    main()

