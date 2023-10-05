import modin.pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.stats import zscore


def calculate_physicochemical_descriptors(df, smiles_column='SMILES', zscore_norm=True):
    # Initialize empty list to store the descriptors
    descriptors_list = []
    
    # Loop through each SMILES string in the DataFrame
    for smiles in df[smiles_column]:
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is not None:
            # Calculate descriptors using RDKit's CalcMolDescriptors
            descriptors = Chem.Descriptors.CalcMolDescriptors(mol)
            descriptors_list.append(descriptors)
        else:
            # Append None if the SMILES string is invalid
            descriptors_list.append([None]*len(Descriptors._descList))
    
    # Convert the list of descriptors into a DataFrame
    descriptors_df = pd.DataFrame(descriptors_list, columns=[desc_name for desc_name, _ in Descriptors._descList])

    # drop columns with no variability
    single_value_columns = descriptors_df.columns[descriptors_df.nunique() <= 1]
    descriptors_df = descriptors_df.drop(columns=single_value_columns)

    # zscore normalize 
    if zscore_norm:
        descriptors_df = descriptors_df.apply(zscore)
    
    # Concatenate the original DataFrame with the descriptors DataFrame
    result_df = pd.concat([df, descriptors_df], axis=1)
    
    return result_df