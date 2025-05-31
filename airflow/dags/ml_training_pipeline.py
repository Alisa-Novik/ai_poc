from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

mlflow.set_tracking_uri("http://mlflow:5000")

def load_data(**context):
    ds = fetch_openml(name="adult", version=2, as_frame=True)
    df = ds.frame
    context["ti"].xcom_push(key="data", value=df.to_json())

def train_and_validate(**context):
    df = pd.read_json(context["ti"].xcom_pull(key="data"))
    df = df.dropna()
    y = df["class"]
    X = df.drop(columns=["class"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    mlflow.sklearn.autolog()
    with mlflow.start_run():
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)

default_args = {"start_date": datetime(2025, 5, 31)}
with DAG("ml_training_pipeline", schedule="@once", catchup=False, default_args=default_args, tags=["mlops"]) as dag:
    load = PythonOperator(task_id="load_data", python_callable=load_data)
    train = PythonOperator(task_id="train_and_validate", python_callable=train_and_validate)
    load >> train
