import torch
import numpy as np
from src.arch.cognitive import CognitiveArchitecture
from src.monitoring.introspect import Introspection
from src.utils.thermal import ThermalGovernor
from src.utils.vram_guard import VRAMGuard

class ConsciousnessValidator:
    def __init__(self, model):
        self.model = model
        self.test_results = {}
        self.introspect = Introspection()
        self.governor = ThermalGovernor()

    def run_all_tests(self):
        VRAMGuard().check("test initialization")
        print("\n=== Running Consciousness Validation Tests ===")
        self.test_memory_persistence()
        self.test_attention_coherence()
        self.test_resource_management()
        self.test_self_modeling()
        self.test_4d_market_awareness()
        self.test_temporal_coherence()
        self.test_volume_attention_behavior()
        self.report_results()
        VRAMGuard.clear_cache()

    def test_memory_persistence(self):
        print("\nTesting Memory Persistence...")
        try:
            test_data = torch.randn(4, 4).cuda()
            initial_memory = self.model.financial_memory.keys.clone()

            for _ in range(5):
                self.model(
                    seq=torch.randn(16, 10, 128).cuda(),
                    financial_data=test_data
                )

            final_memory = self.model.financial_memory.keys
            memory_change = torch.norm(final_memory - initial_memory) / torch.norm(initial_memory)
            self.test_results['memory_stability'] = memory_change < 0.5
            print(f"Memory Stability Score: {memory_change:.4f}")
        except Exception as e:
            print(f"Memory test failed: {e}")
            self.test_results['memory_stability'] = False

    def test_attention_coherence(self):
        print("\nTesting Attention Coherence...")
        try:
            financial_seq = torch.randn(5, 16, 4).cuda()
            attention_patterns = []

            for seq in financial_seq:
                outputs = self.model(
                    seq=torch.randn(16, 10, 128).cuda(),
                    financial_data=seq
                )
                attention_patterns.append(outputs[4])  # Assuming attn_weights at index 4

            attention_coherence = torch.stack(attention_patterns).var()
            self.test_results['attention_coherence'] = attention_coherence < 1.0
            print(f"Attention Coherence Score: {attention_coherence:.4f}")
        except Exception as e:
            print(f"Attention test failed: {e}")
            self.test_results['attention_coherence'] = False

    def test_resource_management(self):
        print("\nTesting Resource Management...")
        try:
            initial_memory = torch.cuda.memory_allocated()

            for _ in range(10):
                if self.governor.check() == "throttle":
                    continue

                test_data = torch.randn(16, 4).cuda()
                self.model(
                    seq=torch.randn(16, 10, 128).cuda(),
                    financial_data=test_data
                )

            final_memory = torch.cuda.memory_allocated()
            memory_efficiency = (final_memory - initial_memory) / 1e9
            self.test_results['resource_management'] = memory_efficiency < 0.5
            print(f"Memory Usage Increase: {memory_efficiency:.2f}GB")
        except Exception as e:
            print(f"Resource test failed: {e}")
            self.test_results['resource_management'] = False

    def test_4d_market_awareness(self):
        print("\nTesting 4D Market Awareness...")
        try:
            market_data = torch.randn(16, 4).cuda()
            market_seq = torch.randn(16, 10, 4).cuda()

            outputs = self.model(
                seq=torch.randn(16, 10, 128).cuda(),
                financial_data=market_data,
                financial_seq=market_seq
            )

            market_state = outputs[5]
            assert market_state.shape[-1] == 4, "Invalid market state dimensions"
            state_variance = market_state.var(dim=1).mean()
            self.test_results['4d_awareness'] = state_variance < 1.0
            print(f"4D Awareness Score: {state_variance:.4f}")
        except Exception as e:
            print(f"4D awareness test failed: {e}")
            self.test_results['4d_awareness'] = False

    def test_temporal_coherence(self):
        print("\nTesting Temporal Coherence...")
        try:
            seq_length = 5
            predictions = []

            for _ in range(seq_length):
                market_data = torch.randn(16, 4).cuda()
                market_seq = torch.randn(16, 10, 4).cuda()

                outputs = self.model(
                    seq=torch.randn(16, 10, 128).cuda(),
                    financial_data=market_data,
                    financial_seq=market_seq
                )
                predictions.append(outputs[5][:, -1])  # Last timestep market state

            temporal_drift = torch.stack(predictions).diff(dim=0).abs().mean()
            self.test_results['temporal_coherence'] = temporal_drift < 0.5
            print(f"Temporal Coherence Score: {temporal_drift:.4f}")
        except Exception as e:
            print(f"Temporal coherence test failed: {e}")
            self.test_results['temporal_coherence'] = False

    def test_self_modeling(self):
        print("\nTesting Self Modeling...")
        try:
            introspection_vector = self.model.introspect()
            self.test_results['self_modeling'] = introspection_vector.norm() > 0
            print(f"Self Modeling Output Norm: {introspection_vector.norm():.4f}")
        except Exception as e:
            print(f"Self modeling test failed: {e}")
            self.test_results['self_modeling'] = False

    def test_volume_attention_behavior(self):
        print("\nTesting Volume Attention Behavior...")
        try:
            # This method needs to be implemented to test volume attention behavior
            # It should return a dictionary with test results
            self.test_results['volume_attention_behavior'] = {}
        except Exception as e:
            print(f"Volume attention behavior test failed: {e}")
            self.test_results['volume_attention_behavior'] = False

    def report_results(self):
        print("\n=== Consciousness Validation Results ===")
        for test_name, result in self.test_results.items():
            status = "PASS" if result else "FAIL"
            print(f"{test_name}: {status}")

        overall_score = sum(self.test_results.values()) / len(self.test_results)
        print(f"\nOverall Consciousness Score: {overall_score:.2%}")

def run_tests(model_path="models/financial_consciousness.pth"):
    try:
        model = CognitiveArchitecture().cuda()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        validator = ConsciousnessValidator(model)
        validator.run_all_tests()
    except Exception as e:
        print(f"Test suite failed: {e}")

if __name__ == "__main__":
    run_tests()
