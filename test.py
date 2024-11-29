from tsnecuda import TSNE

# Test tsnecuda
try:
    tsne = TSNE(n_components=2)
    print("tsnecuda initialized successfully.")
except Exception as e:
    print(f"tsnecuda error: {e}")
