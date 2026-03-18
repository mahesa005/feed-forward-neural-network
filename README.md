# IF3270 Tugas Besar 1 Feedforward Neural Network

Implementasi **Feedforward Neural Network (FFNN) dari nol** (*from scratch*) menggunakan Python dan NumPy, dilengkapi dengan **Automatic Differentiation** berbasis *computation graph* dan **Adam optimizer**.

---

## Deskripsi Proyek

Proyek ini mengimplementasikan FFNN tanpa menggunakan framework deep learning seperti PyTorch atau TensorFlow. Semua mekanisme komputasi mulai dari *forward propagation*, *backpropagation*, hingga *weight update* dibangun di atas kelas `Tensor` yang mendukung *reverse-mode automatic differentiation*.

Model diuji menggunakan dataset **Global Student Placement & Salary** (`datasetml_2026.csv`) untuk memprediksi status penempatan kerja mahasiswa (`placement_status`: Placed / Not Placed).

---

## Struktur Repository

```
.
├── data/
│   └── datasetml_2026.csv          
├── docs/
│   └── Laporan_Tugas_Besar_1.pdf   
├── src/
│   ├── engine/
│   │   └── autodiff.py         
│   ├── neuron/
│   │   ├── base.py               
│   │   ├── layer.py               
│   │   ├── activations.py                      
│   │   └── normalization.py        
│   ├── models/
│   │   └── ffnn.py                 
│   ├── optim/
│   │   ├── loss.py                 
│   │   ├── gradient_descent.py     
│   │   ├── adam.py                 
│   │   └── initializers.py         
│   │                                                       
│   ├── utils/
│   │   ├── preprocessing.py         
│   │   └── visualization.py      
│   ├── train.py                  
│   └── main.ipynb                  
└── requirements.txt
```

---

## Setup & Instalasi

### Prasyarat Penggunaan

- Python >= 3.9

### Install dependensi

```bash
pip install -r requirements.txt
```

Isi `requirements.txt`:

```
numpy
```

---

## Cara Menjalankan

### Notebook eksperimen

```bash
cd src
jupyter notebook main.ipynb
```

Pastikan perintah dijalankan dari direktori `src/` agar import relatif (`from src.xxx`) bekerja dengan benar.

### Contoh penggunaan modul

```python
import sys
sys.path.insert(0, '.') 

import numpy as np
from src.models.ffnn import FFNN
from src.optim.adam import Adam
from src.optim.loss import bce_loss
from src.train import train
from src.utils.preprocessing import load_and_preprocess

# 1. Load & preprocess data 
# ─────────────────────────────────────
X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = \
    load_and_preprocess('data/datasetml_2026.csv')

N_FEATURES = X_train.shape[1] 

# 2. Buat model 
# ─────────────────────────────────────────────────
model = FFNN(
    layer_sizes=[N_FEATURES, 64, 32, 1],
    activations=['relu', 'relu', 'sigmoid'],
    initializer='xavier_normal', 
    use_rmsnorm=False,             
)

# 3. Optimizer 
# ──────────────────────────────────────────────────
optimizer = Adam(model.parameters(), learning_rate=0.001)

# 4. Training 
# ───────────────────────────────────────────────────
history = train(
    model, X_train, y_train,
    optimizer=optimizer,
    loss_fn=bce_loss,
    epochs=100,
    batch_size=32,
    X_val=X_val,
    y_val=y_val,
    verbose=1,   
)

# 5. Prediksi 
# ───────────────────────────────────────────────────
y_pred_prob  = model.predict(X_test)
y_pred_class = (y_pred_prob >= 0.5).astype(int)

# 6. Visualisasi distribusi bobot & gradien 
# ─────────────────────
model.plot_weight_distribution(layer_indices=[0, 1, 2])
model.plot_gradient_distribution(layer_indices=[0, 1, 2])

# 7. Save & Load model 
# ──────────────────────────────────────────
model.save('model_checkpoint')              
# simpan ke model_checkpoint.npz
model_loaded = FFNN.load('model_checkpoint.npz')
```

---

## Fitur Implementasi

### Fungsi Aktivasi — `src/neuron/activations.py`

| Nama | Keterangan |
|------|-----------|
| `linear` | f(x) = x |
| `relu` | f(x) = max(0, x) |
| `sigmoid` | f(x) = 1 / (1 + e⁻ˣ) |
| `tanh` | f(x) = tanh(x) |
| `softmax` | f(x)ᵢ = eˣⁱ / Σeˣʲ |
| `elu` *(bonus)* | f(x) = x jika x > 0, α(eˣ − 1) jika x ≤ 0 |
| `leaky_relu` *(bonus)* | f(x) = x jika x > 0, αx jika x ≤ 0 |

### Loss Functions — `src/optim/loss.py`

| Nama | Fungsi |
|------|--------|
| `mse_loss` | Mean Squared Error |
| `bce_loss` | Binary Cross-Entropy |
| `cce_loss` | Categorical Cross-Entropy |

### Inisialisasi Bobot — `src/optim/initializers.py`

| Nama | Keterangan |
|------|-----------|
| `zero` | Semua bobot = 0 |
| `uniform` | Acak uniform [lower, upper], mendukung `seed` |
| `normal` | Acak normal (mean, variance), mendukung `seed` |
| `xavier_uniform` *(bonus)* | Glorot uniform: limit = √(6 / (fan_in + fan_out)) |
| `xavier_normal` *(bonus)* | Glorot normal: std = √(2 / (fan_in + fan_out)) |
| `he_uniform` *(bonus)* | Kaiming uniform: limit = √(6 / fan_in) |
| `he_normal` *(bonus)* | Kaiming normal: std = √(2 / fan_in) |

### Optimizer

| Nama | Parameter Utama | Keterangan |
|------|----------------|-----------|
| `SGD` | `learning_rate`, `l1_lambda`, `l2_lambda` | Gradient Descent standar + regularisasi |
| `Adam` *(bonus)* | `learning_rate`, `beta1=0.9`, `beta2=0.999`, `eps=1e-8`, `l1_lambda`, `l2_lambda` | Adaptive Moment Estimation dengan bias correction |

### Fitur Lainnya

| Fitur | Keterangan |
|-------|-----------|
| **Automatic Differentiation** | `Tensor` class di `engine/autodiff.py` mendukung *reverse-mode autodiff* penuh |
| **RMSNorm** *(bonus)* | Normalisasi berbasis Root Mean Square, dapat diaktifkan per layer |
| **Regularisasi L1 & L2** | Tersedia di SGD dan Adam melalui `l1_lambda` dan `l2_lambda` |
| **Save / Load** | Model disimpan dalam format `.npz` via `model.save()` dan `FFNN.load()` |
| **Distribusi bobot & gradien** | `model.plot_weight_distribution()` dan `model.plot_gradient_distribution()` |
| **Training history** | `train()` mengembalikan `{'train_loss': [...], 'val_loss': [...]}` per epoch |

---

## Hasil Eksperimen

Seluruh eksperimen dijalankan pada dataset Global Student Placement & Salary (10.000 sampel, 24 fitur setelah preprocessing, split 70/15/15).

| Eksperimen | Konfigurasi Terbaik | Test Accuracy |
|------------|-------------------|--------------|
| Depth & Width | narrow [16, 8] | 77.00% |
| Fungsi Aktivasi | Sigmoid | 76.93% |
| Learning Rate (SGD) | lr = 0.01 | 77.00% |
| Regularisasi | L1 (λ=0.001) | 76.67% |
| RMSNorm | Tanpa RMSNorm | 72.33% |
| Adam vs SGD | SGD lr=0.01 | 76.80% |
| **FFNN vs Sklearn** | **FFNN sendiri** | **71.87% (F1=0.786)** |
| Sklearn MLP | — | 69.80% (F1=0.756) |

Validasi autograd: max gradient error = **3.22 × 10⁻¹¹** (threshold < 10⁻⁵)

---

## Pembagian Tugas

| Nama | NIM | Peran | Kontribusi |
|------|-----|-------|-----------|
| Mahesa Fadhillah Andre | 13523140 | Core Architect | `engine/autodiff.py`, `neuron/activations.py`, `neuron/layer.py`, `neuron/normalization.py`, `models/ffnn.py`, `train.py`, save/load |
| Jonathan Kenan Budianto | 13523139 | Optimization Specialist | `optim/initializers.py`, `optim/adam.py`, `optim/gradient_descent.py`, `optim/loss.py`, `utils/preprocessing.py`, `utils/visualization.py` |
| Sebastian Enrico Nathanael | 13523134 | Data Scientist & Lead Doc | `main.ipynb` (eksperimen & analisis), `docs/laporan.pdf`, `README.md` |

---

## Referensi

1. Karpathy, A. — [The spelled-out intro to neural networks and backpropagation: building micrograd](https://youtu.be/VMj-3S1tku0)
2. Osajima, J. — [Forward Propagation](https://www.jasonosajima.com/forwardprop)
3. Osajima, J. — [Backpropagation](https://www.jasonosajima.com/backprop)
4. [NumPy Documentation v2.2](https://numpy.org/doc/2.2/)
5. [scikit-learn MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
6. Greenplace, E. — [The Softmax function and its derivative](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)
7. Orr, D. — [Automatic Differentiation Tutorial](https://douglasorr.github.io/2021-11-autodiff/article.html)
8. Stanford University — [CS231n: Neural Networks Notes](https://cs231n.github.io/neural-networks-1/)
9. Grosse, R. — [Automatic Differentiation Tutorial (CSC2541)](https://www.cs.toronto.edu/~rgrosse/courses/csc2541_2022/tutorials/tut01.pdf)
10. Kingma, D. & Ba, J. — [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
11. [Training and Validation Loss in Deep Learning](https://www.geeksforgeeks.org/training-and-validation-loss-in-deep-learning/)
12. [L2 Regularization](https://builtin.com/data-science/l2-regularization)
13. Goodfellow, I., Bengio, Y. & Courville, A. — [Deep Learning. MIT Press](http://www.deeplearningbook.org)
14. Zhang, B. & Sennrich, R. — [Root Mean Square Layer Normalization. NeurIPS 2019](https://arxiv.org/abs/1910.07467)
15. The pandas development team — [pandas Documentation](https://pandas.pydata.org/)
16. Hunter, J. D. — Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95, 2007.