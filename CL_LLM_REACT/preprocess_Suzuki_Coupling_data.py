import pandas as pd 
from rdkit import Chem
from collections import defaultdict
# data
# data

reactant_1_smiles = {
    '6-chloroquinoline': 'C1=C(Cl)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC', 
    '6-Bromoquinoline': 'C1=C(Br)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC', 
    '6-triflatequinoline': 'C1C2C(=NC=CC=2)C=CC=1OS(C(F)(F)F)(=O)=O.CCC1=CC(=CC=C1)CC',
    '6-Iodoquinoline': 'C1=C(I)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC', 
    '6-quinoline-boronic acid hydrochloride': 'C1C(B(O)O)=CC=C2N=CC=CC=12.Cl.O',
    'Potassium quinoline-6-trifluoroborate': '[B-](C1=CC2=C(C=C1)N=CC=C2)(F)(F)F.[K+].O',
    '6-Quinolineboronic acid pinacol ester': 'B1(OC(C(O1)(C)C)(C)C)C2=CC3=C(C=C2)N=CC=C3.O'
}

reactant_2_smiles = {
    '2a, Boronic Acid': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1B(O)O', 
    '2b, Boronic Ester': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1B4OC(C)(C)C(C)(C)O4', 
    '2c, Trifluoroborate': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1[B-](F)(F)F.[K+]',
    '2d, Bromide': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1Br' 
}

catalyst_smiles = {
    'Pd(OAc)2': 'CC(=O)O~CC(=O)O~[Pd]'
}

ligand_smiles = {
    'P(tBu)3': 'CC(C)(C)P(C(C)(C)C)C(C)(C)C', 
    'P(Ph)3 ': 'c3c(P(c1ccccc1)c2ccccc2)cccc3', 
    'AmPhos': 'CC(C)(C)P(C1=CC=C(C=C1)N(C)C)C(C)(C)C', 
    'P(Cy)3': 'C1(CCCCC1)P(C2CCCCC2)C3CCCCC3', 
    'P(o-Tol)3': 'CC1=CC=CC=C1P(C2=CC=CC=C2C)C3=CC=CC=C3C',
    'CataCXium A': 'CCCCP(C12CC3CC(C1)CC(C3)C2)C45CC6CC(C4)CC(C6)C5', 
    'SPhos': 'COc1cccc(c1c2ccccc2P(C3CCCCC3)C4CCCCC4)OC', 
    'dtbpf': 'CC(C)(C)P(C1=CC=C[CH]1)C(C)(C)C.CC(C)(C)P(C1=CC=C[CH]1)C(C)(C)C.[Fe]', 
    'XPhos': 'P(c2ccccc2c1c(cc(cc1C(C)C)C(C)C)C(C)C)(C3CCCCC3)C4CCCCC4', 
    'dppf': 'C1=CC=C(C=C1)P([C-]2C=CC=C2)C3=CC=CC=C3.C1=CC=C(C=C1)P([C-]2C=CC=C2)C3=CC=CC=C3.[Fe+2]', 
    'Xantphos': 'O6c1c(cccc1P(c2ccccc2)c3ccccc3)C(c7cccc(P(c4ccccc4)c5ccccc5)c67)(C)C',
    'None': ''
}

reagent_1_smiles = {
    'NaOH': '[OH-].[Na+]', 
    'NaHCO3': '[Na+].OC([O-])=O', 
    'CsF': '[F-].[Cs+]', 
    'K3PO4': '[K+].[K+].[K+].[O-]P([O-])([O-])=O', 
    'KOH': '[K+].[OH-]', 
    'LiOtBu': '[Li+].[O-]C(C)(C)C', 
    'Et3N': 'CCN(CC)CC', 
    'None': ''
}

solvent_1_smiles = {
    'MeCN': 'CC#N.O', 
    'THF': 'C1CCOC1.O', 
    'DMF': 'CN(C)C=O.O', 
    'MeOH': 'CO.O', 
    'MeOH/H2O_V2 9:1': 'CO.O', 
    'THF_V2': 'C1CCOC1.O'
}

def make_reaction_smiles(row):
    precursors = f" {reactant_1_smiles[row['Reactant_1_Name']]}.{reactant_2_smiles[row['Reactant_2_Name']]}.{catalyst_smiles[row['Catalyst_1_Short_Hand']]}.{ligand_smiles[row['Ligand_Short_Hand']]}.{reagent_1_smiles[row['Reagent_1_Short_Hand']]}.{solvent_1_smiles[row['Solvent_1_Short_Hand']]} "
    product = 'C1=C(C2=C(C)C=CC3N(C4OCCCC4)N=CC2=3)C=CC2=NC=CC=C12'
#     print(precursors, product)
    can_precursors = Chem.MolToSmiles(Chem.MolFromSmiles(precursors.replace('...', '.').replace('..', '.').replace(' .', '').replace('. ', '').replace(' ', '')))
    can_product = Chem.MolToSmiles(Chem.MolFromSmiles(product))
    
    return f"{can_precursors}>>{can_product}"
    

def make_reaction(pathtoxl) :
    df = pd.read_excel(pathtoxl)
    df = df.fillna('None')
    df['rxn']= [make_reaction_smiles(row) for i, row in df.iterrows()]
    df['y'] = df['Product_Yield_PCT_Area_UV']/ 100.
    reactions_df = df[['rxn', 'y']] 
    return reactions_df


def create_task_aware_reaction_df(pathtoxl):
    df = pd.read_excel(pathtoxl)
    df = df.fillna('None')
    reaction_smiles_list, task_label_list = [], [] 

    for i, row in df.iterrows():
        precursors = f" {reactant_1_smiles[row['Reactant_1_Name']]}.{reactant_2_smiles[row['Reactant_2_Name']]}.{catalyst_smiles[row['Catalyst_1_Short_Hand']]}.{ligand_smiles[row['Ligand_Short_Hand']]}.{reagent_1_smiles[row['Reagent_1_Short_Hand']]}.{solvent_1_smiles[row['Solvent_1_Short_Hand']]} "
        product = 'C1=C(C2=C(C)C=CC3N(C4OCCCC4)N=CC2=3)C=CC2=NC=CC=C12'
#       print(precursors, product)
        can_precursors = Chem.MolToSmiles(Chem.MolFromSmiles(precursors.replace('...', '.').replace('..', '.').replace(' .', '').replace('. ', '').replace(' ', '')))
        can_product = Chem.MolToSmiles(Chem.MolFromSmiles(product))

        # Generate the reaction SMILES and task!
        reaction_smiles = f"{can_precursors}>>{can_product}"
        task_label = f"{reactant_1_smiles[row['Reactant_1_Name']]}.{reactant_2_smiles[row['Reactant_2_Name']]}"
        reaction_smiles_list.append(reaction_smiles)
        task_label_list.append(task_label)
    df['rxn']= reaction_smiles_list
    df['task_label']= task_label_list
    df['y'] = df['Product_Yield_PCT_Area_UV']/ 100.
    task_aware_reactions_df = df[['task_label', 'rxn', 'y']] 
    
    return task_aware_reactions_df


def task_aware_splits(df):
    """
    Takes the preprocessed pandas dataframe as input and created a task aware split as defined in Nature Machine Intelligence, 4 (2022),1185–1197.
    """

    task_groups = defaultdict(list)

    for i, row in df.iterrows():
        task_groups[row['task_label']].append({"rxn":row["rxn"],"y":row['y']})

    return task_groups
