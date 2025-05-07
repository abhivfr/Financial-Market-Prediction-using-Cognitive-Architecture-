import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("src/monitoring/logs/introspection_log.csv")

# Plot 1: VRAM usage
plt.figure()
plt.plot(df.index, df["vram"])
plt.title("VRAM Usage over Training Steps")
plt.xlabel("Step")
plt.ylabel("VRAM (GB)")
plt.grid(True)
plt.savefig("vram_usage.png")

# Plot 2: Gradient Norms
plt.figure()
plt.plot(df.index, df["gradient_norms"])
plt.title("Gradient Norms over Steps")
plt.xlabel("Step")
plt.ylabel("Gradient Norms")
plt.grid(True)
plt.savefig("grad_norms.png")

# Plot 3: Attention Variance
plt.figure()
plt.plot(df.index, df["attention_variance"])
plt.title("Attention Variance over Steps")
plt.xlabel("Step")
plt.ylabel("Variance")
plt.grid(True)
plt.savefig("attention_variance.png")

print("✅ Plots saved as: vram_usage.png, grad_norms.png, attention_variance.png")
