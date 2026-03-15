.
├── doc/                        # Folder untuk laporan PDF [cite: 171]
│   └── Laporan_Tugas_Besar_1.pdf
├── src/                        # Folder untuk source code [cite: 170]
│   ├── engine/                 # Core logic untuk bonus Automatic Differentiation
│   │   └── autodiff.py         # Implementasi Autodiff (Bonus 40%) 
│   ├── neuron/
│   │   ├── base.py             # Base class untuk layer
│   │   ├── layer.py           # Feedforward layer [cite: 19-20]
│   │   ├── activations.py      # Sigmoid, ReLU, Tanh, Softmax + 2 Bonus [cite: 21-31, 137]
│   │   └── normalization.py    # Implementasi RMSNorm (Bonus 10%) [cite: 142]
│   ├── models/
│   │   └── ffnn.py             # Class utama FFNN (Forward, Backward, Save/Load) [cite: 52-63]
│   ├── optim/
│   │   ├── gradient_descent.py # Standard GD [cite: 72-76]
│   │   ├── adam.py             # Adam Optimizer (Bonus 40%) [cite: 149]
│   │   ├── loss.py             # MSE, BCE, CCE [cite: 34-35]
│   │   └── initializers.py     # Xavier, He, Zero, Uniform, Normal [cite: 40-51, 138-141]
│   ├── utils/
│   │   ├── visualization.py    # Plot distribusi bobot & gradien [cite: 54-58]
│   │   └── preprocessing.py    # Pengolahan dataset Global Student Placement [cite: 125-130]
│   ├── main.ipynb              # File .ipynb untuk pengujian & eksperimen [cite: 89]
│   └── train.py                # Script untuk training loop & verbose mode [cite: 77-86]
├── data/                       # Tempat menyimpan dataset .csv
├── README.md                   # Deskripsi repo, cara run, & pembagian tugas [cite: 195]
└── requirements.txt            # Daftar library (NumPy, Matplotlib, dll.)