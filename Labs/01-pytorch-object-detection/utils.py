import torch
from collections import defaultdict
from tqdm import tqdm

def extract_save_features(loader: torch.utils.data.DataLoader,
                        model: torch.nn.Module,
                        device: torch.device,
                        filename: str):
    all_features = []
    all_targets = defaultdict(list)

    with torch.no_grad():
        model.eval()
        for (inputs, targets) in tqdm(loader):

            inputs = inputs.to(device=device)

            # Compute the forward propagation through the body
            # just to extract the features
            all_features.append(model(inputs))

            for k, v in targets.items():
                all_targets[k].append(v)

        for k, v in all_targets.items():
            all_targets[k] = torch.squeeze(torch.cat(v, 0))
        all_features = torch.squeeze(torch.cat(all_features , 0))
    print(all_features.shape)
    print("The features that are saved are {} features maps of size {} x {}, with {} channels".format(all_features.shape[0], all_features.shape[2], all_features.shape[3], all_features.shape[1]))
    for k, v in all_targets.items():
        print("The entry {} have shape {}".format(k, v.shape))

    torch.save(dict([("features", all_features)] + list(all_targets.items())), filename)


