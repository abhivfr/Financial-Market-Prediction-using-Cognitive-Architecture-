import torch
import torch.nn.functional as F
from src.memory.buffer import MemoryBuffer
from src.memory.encoding import MemoryEncoder
from src.memory.bank import MemoryBank
from src.memory.retrieval import MemoryRetrieval
from src.memory.replay import MemoryReplay

def test_episodic_memory_flow():
    # Configuration
    input_dim = 256
    encoding_dim = 128

    # Initialize components and move to GPU
    memory_buffer = MemoryBuffer(capacity=1)
    memory_encoder = MemoryEncoder(input_dim=input_dim, encoding_dim=encoding_dim).cuda()
    memory_bank = MemoryBank(encoding_dim=encoding_dim, capacity=1).cuda()
    memory_retrieval = MemoryRetrieval().cuda()
    memory_replay = MemoryReplay().cuda()

    # Create a sample experience and move to GPU
    experience = torch.randn(1, input_dim).cuda()

    # Add to buffer
    memory_buffer.add(experience)

    # Encode the experience
    encoded_experience = memory_encoder(experience)

    # Add to memory bank
    memory_bank.add(encoded_experience)

    # Retrieve the memory
    query = encoded_experience.unsqueeze(0)
    retrieved_memory, index = memory_retrieval.retrieve(query, memory_bank)

    # Replay the memory
    replayed_memory = memory_replay.replay(retrieved_memory)

    # Assertions
    assert retrieved_memory is not None, "Memory retrieval failed"
    assert index == torch.tensor([0]).cuda(), "Retrieved incorrect memory index" # Move assertion tensor to GPU
    assert replayed_memory is not None, "Memory replay failed"
    expected_replayed_memory = encoded_experience.clone().detach().requires_grad_(True).cuda() # Move to GPU
    assert torch.allclose(F.normalize(replayed_memory.float(), p=2, dim=-1), F.normalize(expected_replayed_memory.float(), p=2, dim=-1)), "Replayed memory does not match original encoded experience"
