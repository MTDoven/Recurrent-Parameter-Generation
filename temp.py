from model import MambaDiffusion, DDPMSampler
import torch

MambaDiffusion.config.update({
    # mamba config
    "num_permutation_state": 1,
    "d_condition": 1,
    "d_model": 8192,
    "d_model_1": 8192,
    "d_model_2": 8192,
    "d_state": 128,
    "d_conv": 4,
    "expand": 2,
    # diffusion config
    "diffusion_batch": 256,
    "layer_channels": [1, 32, 64, 128, 64, 32, 1],
    "dim_per_token": 8192,
    "kernel_size": 7,
    "sample_mode": DDPMSampler,
    "beta": (0.0001, 0.02),
    "T": 1000,
    "forward_once": True,
})
sequence_length = 489
device = "cuda"
embedding = torch.ones((8, sequence_length, 8192)).to(device)
model = MambaDiffusion(sequence_length=sequence_length,
                       positional_embedding=embedding[0, :, :8192]).to(device)
loss = model(output_shape=embedding.shape, x_0=embedding, condition=None,
             permutation_state=torch.tensor([0,]).view((1,)).repeat(8).to(embedding.device))
loss.backward()
print("ok")
input()



