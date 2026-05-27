import os
import argparse
import shutil
import pandas as pd
import mlflow
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    REPO_OWNER = "Naufalrazani"
    REPO_NAME = "Eksperimen_SML_Muhammad-Naufal-Razani"
    
    print("Menghubungkan ke DagsHub...")
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
    mlflow.set_experiment("Workflow-CI-Retraining")

    if "MLFLOW_RUN_ID" in os.environ:
        del os.environ["MLFLOW_RUN_ID"]

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "dataset_clean/telco_cleaned.csv")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    with mlflow.start_run(run_name="CI_Automated_Run"):
        print(f"Melatih model dengan n_estimators={args.n_estimators}, max_depth={args.max_depth}")
        model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        mlflow.sklearn.log_model(model, "model")
        
        local_model_path = os.path.join(BASE_DIR, "mlflow_model_local")
        if os.path.exists(local_model_path):
            shutil.rmtree(local_model_path)
        mlflow.sklearn.save_model(model, local_model_path)
        print("Model lokal berhasil disimpan di:", local_model_path)