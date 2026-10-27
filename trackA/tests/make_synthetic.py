import numpy as np, uuid
rng = np.random.default_rng(0)
centers = rng.normal(size=(4, 384)) * 3
x = np.vstack([c + rng.normal(size=(1500, 384)) for c in centers]).astype("float32")
x /= np.linalg.norm(x, axis=1, keepdims=True)
np.save("emb.npy", x)
np.save("ids.npy", np.array([str(uuid.uuid4()) for _ in range(len(x))]))