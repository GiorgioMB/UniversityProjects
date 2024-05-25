import pandas as pd
import numpy as np
import torch
import os
import duckdb
from rdkit import Chem, RDLogger
from bioservices import UniProt
from rdkit.Chem import AllChem
import dask.dataframe as dd
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader
import pickle
import gc
import dask
from torch_geometric.nn import AttentiveFP, GIN, Linear, NeuralFingerprint, global_add_pool
from protein_bert.proteinbert import load_pretrained_model
from protein_bert.proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
import torch.optim as optim
import psutil
from typing import List, Tuple, Union
import warnings
import joblib

warnings.filterwarnings("ignore")
device = torch.device("cpu")
class LowRankBilinearPooling(torch.nn.Module):
    """
    A Low-Rank approximation of the Bilinear pooling operation, as described in https://arxiv.org/abs/1610.04325
    Inputs:
    - in_channels1: Number of input channels for the first input tensor
    - in_channels2: Number of input channels for the second input tensor
    - hidden_dim: Number of hidden dimensions to project the input tensors to
    - out_channels: Number of output channels
    - nonlinearity: Non-linear function to apply to the projected input tensors
    - sum_pool: Whether to sum the output channels or not

    forward() inputs:
    - x1: Input tensor 1
    - x2: Input tensor 2
    
    forward() outputs:
    - lrbp: Low-Rank Bilinear Pooling output tensor (Dimensions: [batch_size, 1] if sum_pool is True, [batch_size, out_channels] if sum_pool is False)
    """
    def __init__(self, in_channels1, in_channels2, hidden_dim, out_channels, nonlinearity = torch.nn.ReLU, sum_pool = False):
        super().__init__()
        self.nonlinearity = nonlinearity()
        self.sum_pool = sum_pool
        self.proj1 = Linear(in_channels1, hidden_dim, bias = False, weight_initializer='kaiming_uniform')
        self.proj2 = Linear(in_channels2, hidden_dim, bias = False, weight_initializer='kaiming_uniform')
        self.proj = Linear(hidden_dim, out_channels, weight_initializer = 'kaiming_uniform')
    
    def forward(self, x1, x2):
        x1_ = self.nonlinearity(self.proj1(x1))
        x2_ = self.nonlinearity(self.proj2(x2))
        lrbp = self.proj(x1_.unsqueeze(-2) * x2_.unsqueeze(1))
        return lrbp.sum(dim = (1, 2)) if self.sum_pool else lrbp.squeeze(1)
    

class MultiModelGNNBind(torch.nn.Module):
    """
    Stacked Meta-Model combining Molecular Fingerprinting and Graph Isomorphism Network for binding affinity prediction
    Inputs:
    - num_node_features: Number of node features for the input graphs
    - num_edge_features: Number of edge features for the input graphs
    - num_heads: Number of attention heads for the TransformerConv layers
    - dropout: Dropout ratio for the TransformerConv layers

    forward() inputs:
    - data: PyTorch Geometric Data object for the whole molecule
    - global_protein_features: Tensor of global protein features
    - local_protein_features: Tensor of local protein features

    forward() output: Tensor of predicted binding affinities in the range [0, 1]
    """
    def __init__(self, num_node_features, num_edge_features, num_heads=8, dropout=0.2):
        super(MultiModelGNNBind, self).__init__()
        ## FP Models
        self.graph_attfp = AttentiveFP(in_channels = num_node_features, edge_dim = num_edge_features, hidden_channels = 512, out_channels = 256, num_layers = 3, num_timesteps = 5, dropout = dropout)
        self.graph_neufp = NeuralFingerprint(in_channels = num_node_features, hidden_channels = 512, out_channels = 256, num_layers = 3)  

        ##GIN Model
        self.graph_gin = GIN(in_channels = num_node_features, hidden_channels = 512, num_layers = 3, out_channels = 256, dropout = dropout, jk = 'lstm')
        self.pool = global_add_pool
        ##Protein Encoder
        self.protein_encoder = LowRankBilinearPooling(in_channels1 = 1562, in_channels2 = 15599, hidden_dim = 1000, out_channels = 256 * 3)

        ## Dense layers for combined features
        self.fc1 = LowRankBilinearPooling(in_channels1 = 256 * 3, in_channels2 = 256 * 3, hidden_dim = 2448, out_channels = 2048)
        self.fc2 = Linear(2048, 1024, weight_initializer='kaiming_uniform')
        self.fc3 = Linear(1024, 256, weight_initializer='kaiming_uniform')
        self.fc4 = Linear(256, 64, weight_initializer='kaiming_uniform') 
        self.fc5 = Linear(64, 1, weight_initializer='glorot')
        self.debug_print = True

    def forward(self, data: Data, global_protein_features: torch.Tensor, local_protein_features: torch.Tensor) -> torch.Tensor:        
        ## Molecule layers processing
        x_attfp = self.graph_attfp(data.x, data.edge_index, data.edge_attr, data.batch)
        x_neufp = self.graph_neufp(data.x, data.edge_index, data.batch)
        x_gin = self.graph_gin(data.x, data.edge_index, data.edge_attr, data.batch)
        x_gin = self.pool(x_gin, data.batch)
        ## Encode protein features
        encoded_protein = self.protein_encoder(x1 = global_protein_features, x2 = local_protein_features)
        encoded_protein = F.relu(encoded_protein)
        ##
        if self.debug_print:
            print("x_attfp shape:", x_attfp.shape)
            print("x_neufp shape:", x_neufp.shape)
            print("x_gin shape:", x_gin.shape)
            print("encoded_protein shape:", encoded_protein.shape)
            self.debug_print = False
        ## Concatenate all graph features with protein features
        combined = torch.cat((x_attfp, x_neufp, x_gin), dim = 1)
        ## Pass through dense layers
        combined = F.relu(self.fc1(combined, encoded_protein))
        combined = F.relu(self.fc2(combined))
        combined = F.relu(self.fc3(combined))
        combined = F.relu(self.fc4(combined))
        return torch.sigmoid(self.fc5(combined))


class MoleculeGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset class for the Molecule-Graph dataset
    Inputs:
    - dataframe: Pandas DataFrame containing the dataset
    - device: Device to move the data to

    __getitem__() output:
    - graph: PyTorch Geometric Data object for the molecule
    - local: Tensor of local protein features
    - global: Tensor of global protein features
    - target: Tensor of binding affinity
    """
    def __init__(self, dataframe, device, transform=None, pre_transform=None):
        super(MoleculeGraphDataset, self).__init__(None, transform, pre_transform)
        self.dataframe = dataframe
        self.unique_proteins = self.dataframe['protein_name'].unique()
        self.device = device ##As we have an old GPU, this is more or less useless
        self.init_protein_representation()

    def init_protein_representation(self):
        """
        Initialize protein representations for all unique proteins in the dataset as a dictionary of matrices
        """
        self.prot_dict = {}
        seq_len = 1400
        for protein in self.unique_proteins:
            if protein == 'sEH':
                protein_to_encode = uniprot.retrieve("P34913")['sequence']['value'][1:556]
            elif protein == 'HSA':
                protein_to_encode = uniprot.retrieve("P02768")['sequence']['value'][24:610]
            elif protein == 'BRD4':
                protein_to_encode = uniprot.retrieve("O60885")['sequence']['value'][43:461]
            encoded_protein = input_encoder.encode_X([protein_to_encode], seq_len)
            local_representations, global_representations = encoder.predict(encoded_protein, batch_size=128)
            local_representations = torch.tensor(local_representations[0], device=self.device)
            global_representations = torch.tensor(global_representations[0], device=self.device)
            local_representations = torch.mean(local_representations, dim=0)
            self.prot_dict[protein] = global_representations, local_representations

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple[Data, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[idx]
        graphs = []
        graphs.append(pickle.loads(row['molecule_smiles']))        
        graphs = [graph.to(self.device) for graph in graphs]
        local = torch.tensor(self.prot_dict[row['protein_name']][1], device = self.device)
        global_ = torch.tensor(self.prot_dict[row['protein_name']][0], device = self.device)
        target = torch.tensor([row['binds']], dtype=torch.float, device=self.device)
        return (*graphs, local, global_, target) ##Return the graphs, one-hot encoded protein name and the target

def collate_fn(batch):
    transposed = list(zip(*batch))
    graphs = [Batch.from_data_list(graph_list) for graph_list in transposed[0]] ## Create a batch from the graphs
    local_vector = torch.stack(transposed[1], dim=0) ## Stack the local vectors
    global_vector = torch.stack(transposed[2], dim=0) ## Stack the global vectors
    targets = torch.stack(transposed[3], dim=0) ## Stack the targets
    return (*graphs, local_vector, global_vector, targets)


def get_atom_features(mol: Chem.Mol) -> List[List[Union[int, float]]]:
    features = []
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_ring_sizes = [len(ring) for ring in atom_rings if atom_idx in ring]  ## Sizes of rings this atom is part of
        mean_ring_size = np.mean(atom_ring_sizes) if atom_ring_sizes else 0
        formal_charge = atom.GetFormalCharge() if atom.GetFormalCharge() else -1  ## If no formal charge, set to -1
        features.append([
            atom.GetAtomicNum(),                         ## Atomic number                                     (1)
            atom.GetDegree(),                            ## Degree, number of bonded atoms                    (2)
            atom.GetTotalValence(),                      ## Valence                                           (3)
            formal_charge,                               ## Formal charge                                     (4)
            atom.GetHybridization().real,                ## Hybridization state, converted to a real number   (5)
            atom.GetIsAromatic(),                        ## Aromaticity, boolean converted to int             (6)
            atom.GetTotalNumHs(),                        ## Total number of attached hydrogen atoms           (7)
            atom.IsInRing(),                             ## Is the atom in a ring? Boolean converted to int   (8)
            atom.GetMass(),                              ## Atomic mass                                       (9)
            atom.GetChiralTag().real,                    ## Chirality, converted to a real number             (10)
            atom.GetExplicitValence(),                   ## Explicit valence                                  (11)
            atom.GetImplicitValence(),                   ## Implicit valence                                  (12)
            atom.GetNumRadicalElectrons(),               ## Number of radical electrons                       (13)
            mean_ring_size,                              ## Sizes of rings the atom is a part of              (14)
        ])
    return features


def get_bond_features(mol: Chem.Mol) -> List[List[Union[int, float]]]:
    bond_features = []
    topological_matrix = Chem.GetDistanceMatrix(mol) ## Retrieve only once
    for bond in mol.GetBonds():
        is_in_ring = bond.IsInRing()
        ring_size = 0
        if is_in_ring:
            ring_info = mol.GetRingInfo()
            for ring in ring_info.BondRings():
                if bond.GetIdx() in ring:
                    ring_size = len(ring)
                    break
        start_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        topological_distance = topological_matrix[start_atom_idx][end_atom_idx]
        bond_features.append([
            bond.GetBondTypeAsDouble(),      ## Bond type (single, double, etc.) as double      (1)
            bond.GetIsConjugated(),          ## Conjugation, boolean converted to int           (2)
            bond.GetIsAromatic(),            ## Aromaticity, boolean converted to int           (3)
            bond.GetStereo(),                ## Stereochemistry of the bond                     (4)
            is_in_ring,                      ## Is the bond in a ring? Boolean converted to int (5)
            start_atom_idx,                  ## Index of the start atom of the bond             (6)
            end_atom_idx,                    ## Index of the end atom of the bond               (7)
            ring_size,                       ## Size of the ring that the bond is a part of     (8)
            topological_distance,            ## Topological distance between bonded atoms       (9)
            bond.IsInRing() and mol.GetAtomWithIdx(start_atom_idx).GetDegree() > 2 and mol.GetAtomWithIdx(end_atom_idx).GetDegree() > 2  ## Is rotatable
        ])
    return bond_features

def smiles_to_graph(smiles: str) -> Union[Data, None]:
    mol = Chem.MolFromSmiles(smiles)
    me = Chem.MolFromSmiles('C')
    dy = Chem.MolFromSmiles('[Dy]')
    mol = AllChem.ReplaceSubstructs(mol, dy, me)[0]
    if not mol:
        raise ValueError("Smile provided couldn't be parsed") ##To be sure RDKit can parse it
    mol = Chem.AddHs(mol)
    atom_features = get_atom_features(mol)
    bond_features = get_bond_features(mol)
    ## Create atom feature matrix
    features = torch.tensor(atom_features, dtype=torch.float)

    ## Edge index and edge feature matrix construction
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.append((start, end))
        edge_indices.append((end, start))  ## Since graph is undirected
        for _ in range(2):  ## Add the same bond features for both directions
            edge_attrs.append(bond_features[bond.GetIdx()])

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    data = Data(x = features, edge_index = edge_index, edge_attr = edge_attr)
    return pickle.dumps(data)

def process_and_replace_smiles_columns(df: dd.DataFrame) -> dd.DataFrame:
    column_order = [
        'buildingblock1_smiles', 'buildingblock2_smiles', 'buildingblock3_smiles',
        'molecule_smiles', 'protein_name', 'binds'
    ]
    for column in ['molecule_smiles']:
        df[f'{column}_graph'] = df[column].apply(smiles_to_graph)
        df = df.drop(columns=[column])
        df = df.rename(columns={f'{column}_graph': column})
    df = df.reindex(columns=column_order)
    if df.columns.tolist() != column_order:
        print("Uh oh! You are a silly goose!")
        raise ValueError("Column order is incorrect, expected:", column_order, "but got:", df.columns.tolist())
    return df


def train_model(model: torch.nn.Module, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                criterion: nn.BCELoss, 
                optimizer: optim.Optimizer, 
                num_epochs: int = 10):
    print("Beginning training")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        counter = 0
        for data, local_vector, global_vector, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(data, local_vector, global_vector)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * 100
            counter += 1
            if counter % 100 == 0:
                mem = psutil.virtual_memory()
                print(f'Remaining RAM: {mem.available / 1024**3:.2f} GB')


        epoch_loss = running_loss / len(train_loader.dataset)
        print("Intermediate output: training loss", epoch_loss)
        
        ## Clear memory after each training epoch
        del outputs, loss
        gc.collect()
        torch.cuda.empty_cache()  

        ## Validation phase
        val_loss = 0.0
        all_outputs = []
        all_targets = []
        model.eval()
        with torch.no_grad():
            for data, local_vector, global_vector, targets in val_loader:
                outputs = model(data, local_vector, global_vector)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * 100
                all_outputs.append(outputs)
                all_targets.append(targets)

            val_loss /= len(val_loader.dataset)
            all_outputs = torch.cat(all_outputs)
            all_targets = torch.cat(all_targets)

            if isinstance(all_outputs, torch.Tensor):
                all_outputs = all_outputs.cpu().numpy()
            if isinstance(all_targets, torch.Tensor):
                all_targets = all_targets.cpu().numpy()
            
            mAP = average_precision_score(all_targets, all_outputs, average='micro')

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, mAP: {mAP:.4f}')

        ## Clear memory after validation phase
        del all_outputs, all_targets, outputs, loss
        gc.collect()
        torch.cuda.empty_cache()
        torch.save(model.state_dict(), 'model.pth')
        print('Saved model at epoch: ', epoch+1)
    print('Training complete')

############################################################################################################
if __name__ == '__main__':
    dask.config.set({"dataframe.convert-string": False})
    cpu_count = os.cpu_count()
    print("Number of available CPU cores for this task:", cpu_count)
    model = MultiModelGNNBind(num_node_features=14, num_edge_features=10).to(device)
    print("Model has been moved to device:", device)
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Model overview:")
    print(model)

    train_path = 'train.parquet'
    ##Check that the file exists
    if not os.path.exists(train_path):
        print("File not found")
        raise FileNotFoundError
    ##Note: for now we're only using 20Million 0s
    con = duckdb.connect()
    df = con.query(f"""(SELECT *
                            FROM parquet_scan('{train_path}')
                            WHERE binds = 0
                            ORDER BY random()
                            LIMIT 20000000)
                            UNION ALL
                            (SELECT *
                            FROM parquet_scan('{train_path}')
                            WHERE binds = 1
                            ORDER BY random()
                            LIMIT 10000000)""").df()
    con.close()
    df.drop(columns = 'id', inplace = True)
    to_assert = [
        'buildingblock1_smiles', 'buildingblock2_smiles', 'buildingblock3_smiles',
        'molecule_smiles', 'protein_name', 'binds'
    ]
    ##Check that the order of the columns is correct
    assert df.columns.tolist() == to_assert
    print("Data Loaded, columns are correct")
    print("The columns are:", df.columns)
    print(df[df['binds'] == 0].shape, df[df['binds'] == 1].shape) 
    pretrained_model_generator, input_encoder = load_pretrained_model()
    encoder = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(1400))
    print("Protein model has been loaded")
    ddf = dd.from_pandas(df, npartitions = 10000)
    print("Dask DataFrame has been created")
    dtypes = {
            'buildingblock1_smiles': 'object',
            'buildingblock2_smiles': 'object',
            'buildingblock3_smiles': 'object',
            'molecule_smiles': 'object',
            'protein_name': 'str',
            'binds': 'int'
        }
    sample_df = pd.DataFrame(columns=to_assert)
    for col, type_ in dtypes.items():
        sample_df[col] = sample_df[col].astype(type_)
    print(sample_df.dtypes)
    print("Sample DataFrame has been created")
    processed_ddf = ddf.map_partitions(process_and_replace_smiles_columns, meta = sample_df)
    print("Parallel processing has been started")
    final_df = processed_ddf.compute(scheduler='processes') 
    ##Check the final DataFrame is a Pandas DataFrame
    assert isinstance(final_df, pd.DataFrame)
    print("Graphs have been processed")
    train_df, val_df = train_test_split(final_df, test_size = 0.1, stratify=final_df['binds'], random_state = 42)
    ##Better splitting could be done, but as of now, this is fine
    print("Data has been split")
    uniprot = UniProt()
    print("UniProt Link has been established")


    train_dataset = MoleculeGraphDataset(train_df, device)
    val_dataset = MoleculeGraphDataset(val_df, device)
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, collate_fn=collate_fn)
    print("Data has been loaded into PyTorch DataLoaders")
    criterion = nn.BCELoss() ##Could also be nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, val_loader, criterion, optimizer)
    print("Train.py execution finished")
