
import json
import os

nb_path = "external_validation_filtered.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Modify Cell 0 (Title)
nb["cells"][0]["source"] = [
    "# [Filtered External] PKSmart 모델 성능 검증 리포트 (Similarity Filtered)\n",
    "\n",
    "이 노트북은 **유사도 필터링된 외부 데이터(Filtered Test Set)**를 사용하여 유효성을 평가합니다. 학습 데이터와 화학적 구조가 유사한 물질(Tanimoto Similarity >= 0.5)만 포함하고 있습니다."
]

# Modify Cell 3 (Data Loading)
# Finding the cell that defines EXTERNAL_DATA_PATH
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "EXTERNAL_DATA_PATH =" in source and "All_external_test_Sets.csv" in source:
            cell["source"] = [
                "# 1. 외부 데이터 및 모델 로드 (Filtered)\n",
                "EXTERNAL_DATA_PATH = os.path.join(PROJECT_ROOT, \"validation\", \"external_test_filtered.csv\")\n",
                "MODEL_DIR = os.path.join(PROJECT_ROOT, \"models\")\n",
                "\n",
                "df_raw = pd.read_csv(EXTERNAL_DATA_PATH)\n",
                "df = df_raw.copy() # 이미 필터링됨\n",
                "\n",
                "pipeline = PKSmartPipeline(MODEL_DIR)\n",
                "print(f\"외부 데이터 로드 완료: {len(df)} compounds\")"
            ]
            break

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Patched {nb_path}")
