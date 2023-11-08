#%%
##Full graph (Companies as nodes)
import pandas as pd
import networkx as nx
from itertools import combinations
import numpy as np

df = pd.read_csv('filter_seed.csv')

company_investors = {}
for index, row in df.iterrows():
    company = row['Organization Name']
    # Combine 'Investor Names' and 'Lead Investors' into a single set of unique investors
    investors = set()
    if pd.notnull(row['Investor Names']):
        investors.update(row['Investor Names'].split(','))
    if pd.notnull(row['Lead Investors']):
        investors.update(row['Lead Investors'].split(','))
    
    if company not in company_investors:
        company_investors[company] = set()
    company_investors[company].update(investors)

G = nx.MultiGraph()
for company in company_investors.keys():
    G.add_node(company)

for company1, company2 in combinations(company_investors.keys(), 2):
    shared_investors = len(company_investors[company1].intersection(company_investors[company2]))
    if shared_investors > 0:
        for _ in range(shared_investors):
            G.add_edge(company1, company2)

companies = list(G.nodes)
adjacency_matrix = np.zeros((len(companies), len(companies)), dtype=int)

# Populate the adjacency matrix
for company1, company2, data in G.edges(data=True):
    i, j = companies.index(company1), companies.index(company2)
    adjacency_matrix[i, j] += 1
    adjacency_matrix[j, i] += 1  # Since the graph is undirected

# Adjacency matrix converted to a DataFrame for easier viewing
adjacency_matrix_df = pd.DataFrame(adjacency_matrix, index=companies, columns=companies)

# Print the maximum of the matrix
print(adjacency_matrix_df.max().max())
nx.write_graphml(G, "graph-investor-edges-finaldate.graphml")
adjacency_matrix_df.to_csv("adjacency_matrix-investor-edges-finaldate.csv")


#%%
##Month by month graph (Companies as nodes)
import pandas as pd
import networkx as nx
from itertools import combinations
import numpy as np
from datetime import datetime
import csv

df = pd.read_csv('filter_seed.csv')
df['Announced Date'] = pd.to_datetime(df['Announced Date'], infer_datetime_format = True)
df.sort_values('Announced Date', inplace=True)

company_investors = {}
layered_company_graphs = {} 
# Function to update the graph for the current month with company nodes
def update_company_graph_for_month(current_month, G, company_investors):
    for company in company_investors.keys():
        G.add_node(company)
        
    for company1, company2 in combinations(company_investors.keys(), 2):
        shared_investors = len(company_investors[company1].intersection(company_investors[company2]))
        if shared_investors > 0:
            G.add_edge(company1, company2, start_month=current_month)

# Process the dataframe by month and create graphs
for (year, month), group in df.groupby([df['Announced Date'].dt.year, df['Announced Date'].dt.month]):
    current_month = datetime(year, month, 1)
    if not layered_company_graphs:
        G = nx.MultiGraph()
    else:
        G = layered_company_graphs[max(layered_company_graphs.keys())].copy()
    
    for index, row in group.iterrows():
        company = row['Organization Name']
        investors = set()
        if pd.notnull(row['Investor Names']):
            investors.update(row['Investor Names'].split(','))
        if pd.notnull(row['Lead Investors']):
            investors.update(row['Lead Investors'].split(','))
        if company not in company_investors:
            company_investors[company] = set()
        company_investors[company].update(investors)
    
    update_company_graph_for_month(current_month.strftime('%Y-%m'), G, company_investors)
    layered_company_graphs[current_month] = G

# Save the graphs with company nodes to GraphML files
for month, graph in layered_company_graphs.items():
    filename = f"company_network_{month.strftime('%Y-%m')}.graphml"
    nx.write_graphml(graph, filename)

# Write the mapping of months to company graph filenames to a CSV
with open('company_layer_mapping.csv', 'w', newline='') as csvfile:
    fieldnames = ['month', 'filename']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for month in layered_company_graphs.keys():
        writer.writerow({'month': month.strftime('%Y-%m'), 'filename': f"company_network_{month.strftime('%Y-%m')}.graphml"})

#%%
#%%
## Full graph (Investors as nodes)
import pandas as pd
import networkx as nx
from itertools import combinations
import numpy as np

df = pd.read_csv('filter_seed.csv')

investor_companies = {}
for index, row in df.iterrows():
    # Combine 'Investor Names' and 'Lead Investors' into a single set of unique investors
    investors = set()
    if pd.notnull(row['Investor Names']):
        investors.update(row['Investor Names'].split(','))
    if pd.notnull(row['Lead Investors']):
        investors.update(row['Lead Investors'].split(','))

    for investor in investors:
        if investor not in investor_companies:
            investor_companies[investor] = set()
        investor_companies[investor].add(row['Organization Name'])

G = nx.Graph()
for investor in investor_companies.keys():
    G.add_node(investor)

for investor1, investor2 in combinations(investor_companies.keys(), 2):
    shared_companies = investor_companies[investor1].intersection(investor_companies[investor2])
    if shared_companies:
        G.add_edge(investor1, investor2, companies=list(shared_companies))

nx.write_graphml(G, "investor_graph.graphml")
#%%
## Month by month graph (Investors as nodes)
import pandas as pd
import networkx as nx
from itertools import combinations
import numpy as np
from datetime import datetime
import csv

df = pd.read_csv('filter_seed.csv')
df['Announced Date'] = pd.to_datetime(df['Announced Date'], format='%m-%d-%Y')
df.sort_values('Announced Date', inplace=True)
investor_companies = {}
layered_graphs = {}

# Function to update the graph for the current month
def update_investor_graph_for_month(current_month, G, investor_companies):
    for investor1, investor2 in combinations(investor_companies.keys(), 2):
        shared_companies = investor_companies[investor1].intersection(investor_companies[investor2])
        if shared_companies:
            G.add_edge(investor1, investor2, start_month=current_month, companies=list(shared_companies))

# Process the dataframe by month
for (year, month), group in df.groupby([df['Announced Date'].dt.year, df['Announced Date'].dt.month]):
    current_month = datetime(year, month, 1)
    if not layered_graphs:
        G = nx.Graph()
    else:
        G = layered_graphs[max(layered_graphs.keys())].copy()    
    for index, row in group.iterrows():
        investors = set()
        if pd.notnull(row['Investor Names']):
            investors.update(row['Investor Names'].split(','))
        if pd.notnull(row['Lead Investors']):
            investors.update(row['Lead Investors'].split(','))
        for investor in investors:
            if investor not in investor_companies:
                investor_companies[investor] = set()
            investor_companies[investor].add(row['Organization Name'])
    update_investor_graph_for_month(current_month.strftime('%Y-%m'), G, investor_companies)
    layered_graphs[current_month] = G

for month, graph in layered_graphs.items():
    formatted_month = month.strftime('%Y-%m')
    filename = f"investor_network_{formatted_month}.graphml"
    nx.write_graphml(graph, filename)

with open('investor_layer_mapping.csv', 'w', newline='') as csvfile:
    fieldnames = ['month', 'filename']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for month in layered_graphs.keys():
        formatted_month = month.strftime('%Y-%m')
        filename = f"investor_network_{formatted_month}.graphml"
        writer.writerow({'month': formatted_month, 'filename': filename})
