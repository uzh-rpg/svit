import matplotlib.pyplot as plt
import torch

def plot_distribution(selector_hist, selector_depth, raw_selector_hist, len_dataset):
    #  histogram of number of used tokens per layer
    fig, axes = plt.subplots(nrows=3, ncols=3)
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        ax.hist(selector_hist[i+3], bins=range(0, 200), edgecolor='blue')
        ax.set_ylim([0, 1600])
        ax.axvline(x=50, linestyle='--', linewidth=1, color='black')
        ax.axvline(x=100, linestyle='--', linewidth=1, color='black')
        ax.axvline(x=150, linestyle='--', linewidth=1, color='black')
        ax.axhline(y=800, linestyle='--', linewidth=1, color='black')
        if i>=6:
            ax.set_xlabel('# tokens')
            ax.set_xticks([0,50,100,150,200])
        else:
            ax.set_xticks([])
        if i%3==0:
            ax.set_ylabel('frequency')
            ax.set_yticks([0,800,1600])
        else:
            ax.set_yticks([])
        ax.set_title(f'Block {i + 4}')
    plt.tight_layout()
    plt.show()

    grouped_hist = []
    grouped_hist.append(selector_hist[3]+selector_hist[4] + selector_hist[5])
    grouped_hist.append(selector_hist[6] + selector_hist[7] + selector_hist[8])
    grouped_hist.append(selector_hist[9] + selector_hist[10] + selector_hist[11])
    plt.gca().set_aspect(0.037)
    plt.hist(grouped_hist[0], bins=range(0, 200), histtype='step', edgecolor='blue', label='layer 4,5,6')
    plt.hist(grouped_hist[1], bins=range(0, 200), histtype='step', edgecolor='red', label='layer 7,8,9')
    plt.hist(grouped_hist[2], bins=range(0, 200), histtype='step', edgecolor='green', label='layer 10,11,12')
    plt.xlabel('# tokens')
    plt.ylabel('# samples in validation set')
    plt.ylim(0, 3500)
    plt.legend(loc='upper left')
    plt.show()




    #  depth map of number of used layers per patch location
    selector_depth[0] = selector_depth[0] / len_dataset
    selector_depth[0] = selector_depth[0] + 3
    heat_map = selector_depth[0].reshape(14, 14)
    plt.imshow(heat_map, cmap='hot', vmin=5, vmax=10)
    plt.colorbar()
    plt.show()

    # ratio of dropped forever / used in the next layer/ used in later layers
    raw_selectors = [torch.cat(x, dim=0).cpu() if len(x) != 0 else None for x in raw_selector_hist]  # [None, None, None, (50000, 196), ..., (50000, 196)]
    for i in range(len(raw_selectors)):
        if raw_selectors[i] is None:
            raw_selectors[i] = torch.ones_like(raw_selectors[-3])  # [(50000, 196), ..., (50000, 196)]
    raw_selectors = torch.cat([x.unsqueeze(0) for x in raw_selectors], dim=0)  # (12, 50000, 196)
    use_next = torch.zeros_like(raw_selectors)  # (12, 50000, 196)
    use_later = torch.zeros_like(raw_selectors)  # (12, 50000, 196)
    for i in range(len(use_next) - 1):
        use_next[i] = (1 - raw_selectors[i]) * raw_selectors[i + 1]
    for i in range(len(use_later) -1):
        use_later[i] = torch.logical_and(~raw_selectors[i].bool(), raw_selectors[i+1:].sum(0)>=1).float()
    drop = torch.logical_and(~raw_selectors.bool(), ~use_later.bool()).float()
    ratio_select = raw_selectors.view(raw_selectors.shape[0], -1).mean(-1)
    ratio_use_next = use_next.view(use_next.shape[0], -1).mean(-1)
    ratio_use_later = use_later.view(use_later.shape[0], -1).mean(-1)
    ratio_drop = drop.view(drop.shape[0], -1).mean(-1)
    X = ['4', '5', '6', '7', '8', '9', '10', '11', '12']
    X_axis = torch.arange(len(X))
    width = 0.25
    Black = [0, 0, 0, 1]
    Amber = [1, 0.75, 0, 1]
    Aqua = [0, 1, 1, 1]
    plt.bar(X_axis - width, ratio_drop[3:], width, label='tokens skipped & not used later', color=Black)
    plt.bar(X_axis, ratio_use_later[3:], width, label='tokens skipped & used later', color=Aqua)
    plt.bar(X_axis + width, ratio_use_next[3:], width, label='tokens skipped & used in the next layer', color=Amber)
    plt.xticks(X_axis, X)
    plt.ylim(0, 0.7)
    plt.legend()
    plt.xlabel('Layers')
    plt.ylabel('Ratio of tokens')
    plt.show()
    return