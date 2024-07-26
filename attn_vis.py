import torch
from model.models import EncoderModelWithProjection
import seaborn
import matplotlib.pyplot as plt
import os


data = torch.load("data/eeg_5_95_std.pth")

splits = torch.load("data/block_splits_by_image_all.pth")["splits"][0]["val"]

index = splits[0]

gpu = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
)

eeg = data["dataset"][index]["eeg"][:, 20:460].float().unsqueeze(dim=0).to(gpu)

model:EncoderModelWithProjection = EncoderModelWithProjection.from_pretrained("pretrained/kd-cos")

model.to(device=gpu)

res = model.get_attn_maps(eeg)

for layer in range(12):
    attn_map = res[layer].squeeze(dim=0)

    if not os.path.exists(f"layer-{layer + 1}"):
        os.makedirs(f"layer-{layer + 1}")

    for i in range(len(attn_map)):
        fig, axs = plt.subplots(1, 1, figsize=(12, 12))
        print(attn_map[i])
        seaborn.heatmap(
            attn_map[i].detach().cpu().numpy(),
            square=True,
            vmin=0.0,
            vmax=1.0,
            cbar=False,
            ax=axs,
        )
        fig.savefig(f"layer-{layer + 1}/head-{i + 1}.png")
