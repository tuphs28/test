import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from experiments.pilleater_experiments.create_probing_dataset_old import ProbingDataset, ProbingDatasetCleaned
from typing import Optional
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import time

class ConvProbe(nn.Module):
    def __init__(self, in_channels: int, out_dim: int, kernel_size: int, padding: int = 0, nl: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_dim, kernel_size=kernel_size, padding=padding, bias=False)
        self.out_dim = out_dim
        self.loss_fnc = nn.CrossEntropyLoss()
    def forward(self, input: torch.tensor, targets: Optional[torch.tensor] = None):
        out = self.conv(input)
        if targets is not None:
            assert out.shape[0] == targets.shape[0]
            out = out.view(out.shape[0], self.out_dim, 169)
            targets = targets.view(out.shape[0], 169)
            loss = self.loss_fnc(out, targets)
        else:
            loss = None
        return out, loss
    
class LinProbe(nn.Module):
    def __init__(self, in_channels: int, out_dim: int, nl: bool = False):
        super().__init__()
        self.ff = nn.Linear(in_features=in_channels*64, out_features=out_dim*64, bias=False)
        #self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_dim, kernel_size=kernel_size, padding=padding, bias=False)
        self.out_dim = out_dim
        self.loss_fnc = nn.CrossEntropyLoss()
    def forward(self, input: torch.tensor, targets: Optional[torch.tensor] = None):
        input = input.view(input.shape[0], -1)
        #out = self.conv(input)
        out = self.ff(input)
        if targets is not None:
            assert out.shape[0] == targets.shape[0]
            out = out.view(out.shape[0], self.out_dim, 64)
            targets = targets.view(out.shape[0], 64)
            loss = self.loss_fnc(out, targets)
        else:
            loss = None
        return out, loss
    
if __name__ == "__main__":
    import pandas as pd
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="run convprobe patching exps")
    parser.add_argument("--feature", type=str, default="agent_loc_future_trajectory_120")
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--kernel", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="250m")
    parser.add_argument("--test_mode", type=str, default="test")
    parser.add_argument("--convprobe_off", action="store_false", default=True, help="use linprobe rather than convprobe")
    parser.add_argument("--debug", action="store_false", default=True, help="debug")
    args = parser.parse_args()

    channels = list(range(32))
    batch_size = 2
    num_epochs = args.num_epochs
    wd = args.weight_decay
    kernel = args.kernel
    testmode = args.test_mode
    conv = args.convprobe_off
    model_name = args.model_name 
    debug = args.debug
    assert kernel in [1,3,5]
    probe_args = {}
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): 
        print("test")
        device = torch.device("cuda")
    else: 
        device = torch.device("cpu") 
    
    features = [args.feature]
    #features = ["tar_next_current_0"]
    layers = [("layer0", 32), ("layer1", 96), ("layer2", 160), ("x", 0)][:]
    #layers = [("layer2", 160)]
    for feature in features:
        print(f"=================================== FEATURE: {feature} =========================================")

        probe_args["feature"] = feature        
        probe_args["positive_feature"] = feature
        results = {}

        train_dataset_c = torch.load(f"./data/train_data_full_{model_name}.pt")
        if testmode == "test":
            print("!!!!")
            test_dataset_c = torch.load(f"./data/test_data_full_{model_name}.pt")
        else:
            test_dataset_c = torch.load(f"./data/val_data_random_{model_name}.pt")
        cleaned_train_data, cleaned_test_data, cleaned_val_data = [], [], []
        for trans in train_dataset_c.data:
            if type(trans[probe_args["feature"]]) == int:
                if trans[probe_args["feature"]] != -1:
                    cleaned_train_data.append(trans)
            else:
                cleaned_train_data.append(trans)
        for trans in test_dataset_c.data:
            if type(trans[probe_args["feature"]]) == int:
                if trans[probe_args["feature"]] != -1:
                    cleaned_test_data.append(trans)
            else:
                cleaned_test_data.append(trans)
        train_dataset_c.data = cleaned_train_data
        test_dataset_c.data = cleaned_test_data
        out_dim = 1 + int(max([c[feature].max().item() for c in train_dataset_c.data]))
        print(out_dim)
        for seed in [0]: # CHANGE BACK
            print(f"=============== Seed: {seed} ================")
            torch.manual_seed(seed)
            for mode in ["hidden_states"]:

                cleaned_train_data = [(trans[mode].cpu(), trans["board_state"], trans[probe_args["feature"]], trans[probe_args["positive_feature"]]) for trans in train_dataset_c.data]
                cleaned_test_data = [(trans[mode].cpu(), trans["board_state"], trans[probe_args["feature"]], trans[probe_args["positive_feature"]]) for trans in test_dataset_c.data]
                train_dataset = ProbingDatasetCleaned(cleaned_train_data)
                test_dataset = ProbingDatasetCleaned(cleaned_test_data)
                
                for layer_name, layer_idx in layers:
                    if mode == "bl_hidden_states" and layer_name == "x":
                        break
                    print(f"========= {layer_name} =========")
                    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,persistent_workers=True)
                    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,persistent_workers=True)

                    if conv:
                        if debug:
                            print("convprobe")
                        probe = ConvProbe(in_channels=14 if layer_name=="x" else 32, out_dim=out_dim, kernel_size=kernel, padding=(0 if kernel==1 else (1 if kernel == 3 else 2)))
                    else:
                        if debug:
                            print("linprobe")
                        probe = LinProbe(in_channels=14 if layer_name=="x" else 32, out_dim=out_dim)
                    probe.to(device)
                    optimiser = torch.optim.AdamW(params=probe.parameters(), lr=1e-3, weight_decay=wd)
               
                    for epoch in range(1, num_epochs+1):
                        start_time = time.time()

                        precisions = [0 for _ in range(out_dim)]
                        recalls = [0 for _ in range(out_dim)]
                        fones = [0 for _ in range(out_dim)]
                        conf_mat = [[0 for i in range(out_dim)] for j in range(out_dim)]

                        for hiddens, states, targets, _ in train_loader:
                            #print(hiddens.device, targets.device, _.device, convprobe.conv.weight.device)
                            hiddens = states.to(torch.float).to(device) if layer_name=="x" else hiddens[:,-1,[layer_idx+c for c in channels],:,:].to(device)
                            targets = targets.to(torch.long).to(device)
                            optimiser.zero_grad()
                            logits, loss = probe(hiddens, targets)
                            loss.backward()
                            optimiser.step()
                        full_acc = 0
                        positive_acc = 0
                        prop_pos_cor = 0
                        if epoch % 1 == 0:
                            with torch.no_grad():
                                labs, preds = [], []
                                for hiddens, states, targets, positive_targets in test_loader:
                                    hiddens = states.to(torch.float).to(device) if layer_name=="x" else hiddens[:,-1,[layer_idx+c for c in channels],:,:].to(device)
                                    targets = targets.to(torch.long).to(device)
                                    logits, loss = probe(hiddens, targets)
                                    full_acc += (torch.sum(logits.argmax(dim=1)==targets.view(-1,169)).item())
                                    preds += logits.argmax(dim=1).view(-1).tolist()
                                    labs += positive_targets.view(-1).tolist()
                                    #if probe_args["positive_feature"] is not None:
                                        #for i in range(positive_targets.shape[0]):
                                            #for j in range(out_dim):
                                                #for k in range(out_dim):
                                                #conf_mat[j][k] += torch.sum((logits[[i],:,:].argmax(dim=1)==k)[positive_targets[[i],:,:].view(-1,64)==j]).item()
                                if out_dim == 2:
                                    prec, rec, f1, sup = precision_recall_fscore_support(labs, preds, average='binary', pos_label=1, zero_division=1, labels=[0,1])
                                else:
                                    prec, rec, f1, sup = precision_recall_fscore_support(labs, preds, average='macro', zero_division=1, labels=list(range(out_dim)))
                                
                                precisions, recalls, fones, _ = precision_recall_fscore_support(labs, preds, average=None, zero_division=1, labels=list(range(out_dim)))

                                print(f"---- Epoch {epoch} -----")
                                print("Full acc:", full_acc/(len(test_dataset.data)*169))
                                print("F1:", f1)
                                print("Time:", time.time()-start_time)
                                #if probe_args["positive_feature"] is not None:
                                    #for j in range(out_dim):
                                        #print(f"-- Out Dim {j} --")
                                        #recalls[j] = conf_mat[j][j] / sum(conf_mat[j]) if sum(conf_mat[j]) > 0 else 1 # I CHANGED THIS TO 1
                                        #precisions[j] = conf_mat[j][j] / sum([conf_mat[k][j] for k in range(out_dim)]) if  sum([conf_mat[k][j] for k in range(out_dim)]) > 0 else 1 # I CHANGED THIS TO 1
                                        #fones[j] = 0 if precisions[j]+recalls[j]==0 else (2*precisions[j]*recalls[j]) / (precisions[j] + recalls[j])

                                        #print("Precision:", precisions[j])
                                        #print("Recall:", recalls[j])
                                        #print("F1: ", fones[j])

                    if out_dim != 1:
                        results_dict = {"Acc": full_acc/(len(test_dataset.data)*169)}
                        for j in range(out_dim):
                            results_dict[f"Precision_{j}"] = precisions[j]
                            results_dict[f"Recall_{j}"] = recalls[j]
                            results_dict[f"F1_{j}"] = fones[j]
                        results_dict["Avg_F1"] = f1
                        results[f"{layer_name}_{mode}"] = results_dict

                    if not os.path.exists(f"./convresults/models/{feature}") and mode=="hidden_states":
                        os.mkdir(f"./convresults/models/{feature}")
                    torch.save(probe.state_dict(), f"./convresults/models/{feature}/{model_name}{'full' if not conv else ''}_{layer_name}_kernel{kernel}_wd{wd}_seed{seed}.pt")

            results_df = pd.DataFrame(results)
            results_df.to_csv(f"./convresults/{model_name}{'full' if not conv else ''}_{feature}_kernel{kernel}_wd{wd}_seed{seed}.csv")
