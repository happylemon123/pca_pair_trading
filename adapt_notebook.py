import json
import os

notebook_path = 'pca_strategy_real.ipynb'

if not os.path.exists(notebook_path):
    print(f"Error: {notebook_path} not found.")
    exit(1)

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Define the new cells
markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 1. Load Real Data\n",
        "We load historical adjusted close prices for Technology Sector stocks (S&P 500 components) downloaded using `yfinance`.\n",
        "The data is stored in `data/stock_prices.csv`."
    ]
}

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Load Data\n",
        "import pandas as pd\n",
        "import os\n",
        "\n",
        "data_path = 'data/stock_prices.csv'\n",
        "if os.path.exists(data_path):\n",
        "    prices_df = pd.read_csv(data_path, index_col=0, parse_dates=True)\n",
        "    print(f'Loaded data shape: {prices_df.shape}')\n",
        "else:\n",
        "    raise FileNotFoundError(f'{data_path} not found. Please run data_loader.py first.')\n",
        "\n",
        "# Calculate daily returns\n",
        "returns_df = prices_df.pct_change().dropna()\n",
        "print(f'Returns shape: {returns_df.shape}')\n",
        "\n",
        "# Visualize prices\n",
        "prices_df.plot(legend=False, title='Stock Prices (Normalized)', figsize=(12, 6))\n",
        "plt.show()"
    ]
}

# Find and replace the simulation cells
new_cells = []
simulation_found = False

for cell in nb['cells']:
    # Detect the "Data Simulation" markdown cell
    if cell['cell_type'] == 'markdown' and "## 1. Data Simulation" in "".join(cell['source']):
        new_cells.append(markdown_cell)
        simulation_found = True
        continue
    
    # Detect the simulation code cell (heuristic: looks for np.random.seed)
    if cell['cell_type'] == 'code' and "np.random.seed" in "".join(cell['source']):
        new_cells.append(code_cell)
        continue
        
    new_cells.append(cell)

if simulation_found:
    nb['cells'] = new_cells
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully updated {notebook_path} to use real data.")
else:
    print("Could not find simulation section to replace.")
