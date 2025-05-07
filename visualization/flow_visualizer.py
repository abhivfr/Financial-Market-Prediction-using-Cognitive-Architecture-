#!/usr/bin/env python
# flow_visualizer.py - Information flow visualization for cognitive architecture

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Any, Optional
import json
from datetime import datetime
from collections import defaultdict

class InformationFlowVisualizer:
    """
    Visualizes information flow through cognitive architecture components
    Tracks activation patterns, attention weights, and component interactions
    """
    def __init__(self, model, output_dir="visualizations/flow"):
        """
        Initialize the information flow visualizer for cognitive architecture
        
        Args:
            model: The cognitive architecture model
            output_dir: Directory to save visualizations
        """
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Hooks and activations storage
        self.hooks = []
        self.activations = defaultdict(list)
        self.component_map = {}
        
        # Component relationships
        self.connections = []
        
        # Register hooks for all modules
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks on all relevant model components"""
        # Map component names
        if hasattr(self.model, 'attention'):
            self.component_map['attention'] = self.model.attention
        if hasattr(self.model, 'memory_module'):
            self.component_map['memory'] = self.model.memory_module
        if hasattr(self.model, 'forecast_module'):
            self.component_map['forecast'] = self.model.forecast_module
        if hasattr(self.model, 'regime_detector'):
            self.component_map['regime'] = self.model.regime_detector
        
        # Define module connections
        self.connections = [
            ('input', 'attention'),
            ('input', 'regime'),
            ('attention', 'memory'),
            ('regime', 'memory'),
            ('memory', 'forecast'),
            ('attention', 'forecast')
        ]
        
        # Register hooks for capturing activations
        for name, module in self.component_map.items():
            hook = module.register_forward_hook(self._create_hook_fn(name))
            self.hooks.append(hook)
    
    def _create_hook_fn(self, name):
        """Create a hook function for the specified module"""
        def hook_fn(module, input, output):
            # Capture input and output activations
            if isinstance(input, tuple) and len(input) > 0:
                # Store mean activation magnitude
                if torch.is_tensor(input[0]):
                    self.activations[f"{name}_in"] = input[0].detach().mean().item()
            
            if torch.is_tensor(output):
                self.activations[f"{name}_out"] = output.detach().mean().item()
            elif isinstance(output, tuple) and torch.is_tensor(output[0]):
                self.activations[f"{name}_out"] = output[0].detach().mean().item()
                
        return hook_fn
    
    def capture_model_run(self, inputs, clear_previous=True):
        """
        Capture activations from a model forward pass
        
        Args:
            inputs: Model inputs
            clear_previous: Whether to clear previous activations
        """
        if clear_previous:
            self.activations.clear()
        
        # Add input activation
        if torch.is_tensor(inputs):
            self.activations["input"] = inputs.detach().mean().item()
        
        # Run model forward pass
        with torch.no_grad():
            _ = self.model(inputs)
    
    def create_component_graph(self):
        """Create a networkx graph of component relationships and activations"""
        G = nx.DiGraph()
        
        # Create nodes for all components
        components = set()
        for src, tgt in self.connections:
            components.add(src)
            components.add(tgt)
        
        # Add nodes with activation data
        for component in components:
            activation = 0
            if f"{component}_out" in self.activations:
                activation = self.activations[f"{component}_out"]
            elif component in self.activations:
                activation = self.activations[component]
                
            G.add_node(component, activation=activation)
        
        # Add edges with weights based on activations
        for src, tgt in self.connections:
            src_key = f"{src}_out" if f"{src}_out" in self.activations else src
            tgt_key = f"{tgt}_in" if f"{tgt}_in" in self.activations else tgt
            
            if src_key in self.activations and tgt_key in self.activations:
                # Edge weight is the normalized correlation between src output and tgt input
                src_val = self.activations[src_key]
                tgt_val = self.activations[tgt_key]
                weight = min(abs(src_val * tgt_val) / (abs(src_val) + 1e-10), 1.0)
                
                G.add_edge(src, tgt, weight=weight)
        
        return G
    
    def visualize_flow(self, title="Information Flow in Cognitive Architecture", save_path=None):
        """
        Visualize information flow through the model components
        
        Args:
            title: Plot title
            save_path: Path to save the visualization
        """
        G = self.create_component_graph()
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Position nodes
        pos = {
            'input': (0, 0.5),
            'attention': (0.3, 0.8),
            'regime': (0.3, 0.2),
            'memory': (0.6, 0.5),
            'forecast': (0.9, 0.5)
        }
        
        # Get node sizes based on activation values
        activations = [G.nodes[n].get('activation', 0.1) for n in G.nodes()]
        node_sizes = [2000 * (abs(a) + 0.1) for a in activations]
        
        # Draw nodes with custom labels
        nx.draw_networkx_nodes(G, pos, 
                               node_size=node_sizes,
                               node_color='skyblue', 
                               alpha=0.8)
        
        # Add labels with activation values
        node_labels = {}
        for node in G.nodes():
            act_val = G.nodes[node].get('activation', 0)
            node_labels[node] = f"{node}\n({act_val:.2f})"
        
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
        
        # Draw edges with varying thickness for weight
        for (u, v, d) in G.edges(data=True):
            width = d.get('weight', 0.1) * 5
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width,
                                   alpha=0.7, edge_color='gray',
                                   connectionstyle='arc3,rad=0.1')
        
        plt.title(title)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.savefig(os.path.join(self.output_dir, 'information_flow.png'))
        
        plt.close()
    
    def generate_flow_report(self, inputs_batch, save_dir=None):
        """
        Generate a comprehensive information flow report for a batch of inputs
        
        Args:
            inputs_batch: Batch of model inputs
            save_dir: Directory to save the report
        """
        if save_dir is None:
            save_dir = self.output_dir
            
        os.makedirs(save_dir, exist_ok=True)
        
        flow_data = {
            "component_activations": {},
            "connection_strengths": []
        }
        
        # Process each input in the batch
        for i, inputs in enumerate(inputs_batch):
            self.capture_model_run(inputs)
            
            # Store component activations
            for comp in self.component_map:
                out_key = f"{comp}_out"
                if out_key in self.activations:
                    if comp not in flow_data["component_activations"]:
                        flow_data["component_activations"][comp] = []
                    flow_data["component_activations"][comp].append(self.activations[out_key])
            
            # Store connection strengths
            G = self.create_component_graph()
            for u, v, d in G.edges(data=True):
                flow_data["connection_strengths"].append({
                    "source": u,
                    "target": v,
                    "weight": d.get("weight", 0),
                    "sample_idx": i
                })
                
            # Generate visualization
            save_path = os.path.join(save_dir, f"flow_sample_{i}.png")
            self.visualize_flow(title=f"Information Flow - Sample {i}", save_path=save_path)
        
        # Save flow data
        with open(os.path.join(save_dir, "flow_data.json"), "w") as f:
            json.dump(flow_data, f, indent=2)
            
        # Generate summary visualization
        self._generate_summary_visualization(flow_data, save_dir)
        
        return flow_data
    
    def _generate_summary_visualization(self, flow_data, save_dir):
        """Generate summary visualization of information flow patterns"""
        # Compute average component activations
        avg_activations = {}
        for comp, values in flow_data["component_activations"].items():
            avg_activations[comp] = sum(values) / len(values)
        
        # Compute average connection strengths
        connection_strengths = defaultdict(list)
        for conn in flow_data["connection_strengths"]:
            key = (conn["source"], conn["target"])
            connection_strengths[key].append(conn["weight"])
        
        avg_connections = {}
        for (src, tgt), weights in connection_strengths.items():
            avg_connections[(src, tgt)] = sum(weights) / len(weights)
        
        # Create summary graph
        G = nx.DiGraph()
        
        # Add nodes with average activation
        for comp, act in avg_activations.items():
            G.add_node(comp, activation=act)
        
        # Add edges with average weights
        for (src, tgt), weight in avg_connections.items():
            G.add_edge(src, tgt, weight=weight)
        
        # Create visualization
        plt.figure(figsize=(14, 10))
        
        # Position nodes
        pos = {
            'input': (0, 0.5),
            'attention': (0.3, 0.8),
            'regime': (0.3, 0.2),
            'memory': (0.6, 0.5),
            'forecast': (0.9, 0.5)
        }
        
        # Get node sizes based on activation values
        activations = [G.nodes[n].get('activation', 0.1) for n in G.nodes()]
        node_sizes = [3000 * (abs(a) + 0.1) for a in activations]
        
        # Draw nodes with custom labels
        nx.draw_networkx_nodes(G, pos, 
                               node_size=node_sizes,
                               node_color='skyblue', 
                               alpha=0.8)
        
        # Add labels with activation values
        node_labels = {}
        for node in G.nodes():
            act_val = G.nodes[node].get('activation', 0)
            node_labels[node] = f"{node}\n({act_val:.2f})"
        
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12)
        
        # Draw edges with varying thickness for weight
        for (u, v, d) in G.edges(data=True):
            width = d.get('weight', 0.1) * 8
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width,
                                   alpha=0.7, edge_color='gray',
                                   connectionstyle='arc3,rad=0.1')
            
            # Add edge labels with weights
            edge_label = f"{d.get('weight', 0):.2f}"
            edge_labels = {(u, v): edge_label}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
        
        plt.title("Average Information Flow Across Samples")
        plt.axis('off')
        
        plt.savefig(os.path.join(save_dir, 'average_information_flow.png'))
        plt.close()
