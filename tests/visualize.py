import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib.animation import FuncAnimation

plt.style.use('dark_background')

class ConsciousnessVisualizer:
    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Consciousness Monitoring', fontsize=16)
        plt.ion()  # Interactive mode

    def update(self, arch, ndict, metrics):
        self._plot_memory_state(ndict, self.axes[0, 0])
        self._plot_attention(arch, self.axes[0, 1])
        self._plot_resources(metrics, self.axes[1, 0])
        self._plot_consciousness_metrics(arch, self.axes[1, 1])
        plt.pause(0.1)

    def _plot_memory_state(self, ndict, ax):
        ax.clear()
        memory = ndict.get_memory()
        memory = torch.cat(memory).detach().cpu()
        sns.heatmap(memory[:10], ax=ax, cmap='viridis')
        ax.set_title('Memory State')

    def _plot_attention(self, arch, ax):
        ax.clear()
        if hasattr(arch.core, 'attention'):
            weights = arch.core.attention.attention_weights.detach().cpu()
            sns.heatmap(weights, ax=ax, cmap='magma')
        ax.set_title('Attention Patterns')

    def _plot_resources(self, metrics, ax):
        ax.clear()
        vram = metrics['vram'][-50:]  # Last 50 points
        ax.plot(vram, label='VRAM Usage (GB)', color='cyan')
        ax.set_title('Resource Usage')
        ax.legend()

    def _plot_consciousness_metrics(self, arch, ax):
        ax.clear()
        if hasattr(arch.core, 'depth_counter'):
            depths = [arch.core.depth_counter]
            ax.bar(['Recursive Depth'], depths, color='magenta')
        ax.set_title('Consciousness Depth')
