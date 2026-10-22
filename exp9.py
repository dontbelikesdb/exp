import pandas as pd
import numpy as np
from tensorflow.keras import layers, Model

# 1. Load dataset
df = pd.read_csv("events.csv")
df = df[["visitorid", "itemid"]].drop_duplicates()
df = df.groupby("visitorid").filter(lambda x: len(x) >= 3)
df = df[df["visitorid"].isin(df["visitorid"].unique()[:1000])]

# 2. Encode users and items
df["user"] = pd.factorize(df["visitorid"])[0]
df["item"] = pd.factorize(df["itemid"])[0]
nu, ni = df["user"].nunique(), df["item"].nunique()

# 3. Split into training and testing
train = df.groupby("user").head(-1)
test = df.groupby("user").tail(1)

# 4. Generate negative samples
rng = np.random.default_rng(42)
u, i = train["user"].to_numpy(), train["item"].to_numpy()
neg = rng.integers(0, ni, len(i))

X = np.r_[u, u]
Y = np.r_[i, neg]
labels = np.r_[np.ones(len(i)), np.zeros(len(i))]

# 5. Build recommendation model
user = layers.Input(shape=(1,))
item = layers.Input(shape=(1,))
a = layers.Flatten()(layers.Embedding(nu, 8)(user))
b = layers.Flatten()(layers.Embedding(ni, 8)(item))
x = layers.Concatenate()([a, b])
x = layers.Dense(16, activation="relu")(x)
out = layers.Dense(1, activation="sigmoid")(x)

model = Model([user, item], out)
model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["accuracy"])

# 6. Train model
model.fit([X, Y], labels, epochs=3, batch_size=256, verbose=1)

# 7. Recommend top 5 items and evaluate
P, R = [], []

for uid in test["user"].unique()[:100]:
    actual = set(test[test["user"] == uid]["item"])
    seen = set(train[train["user"] == uid]["item"])
    candidates = np.array([j for j in range(ni) if j not in seen])

    scores = model.predict(
        [np.full(len(candidates), uid), candidates], verbose=0
    ).flatten()

    top = candidates[np.argsort(scores)[-5:]]
    hits = len(set(top) & actual)

    P.append(hits / 5)
    R.append(hits / len(actual))

print("\nRecommended items:", top[::-1])
print("Precision@5:", round(np.mean(P), 4))
print("Recall@5:", round(np.mean(R), 4))