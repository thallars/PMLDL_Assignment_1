from pathlib import Path
import pickle

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


X, y = make_classification(
    n_samples=500,
    n_features=4,
    n_informative=4,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42,
)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = make_pipeline(
    StandardScaler(),
    LogisticRegression(solver="liblinear", random_state=42),
)

model.fit(X_train, y_train)


preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Test accuracy: {acc:.3f}")


repo_root = Path(__file__).resolve().parents[2]
models_dir = repo_root / "models"
models_dir.mkdir(parents=True, exist_ok=True)
model_path = models_dir / "model.pkl"

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Saved trained model to: {model_path}")
