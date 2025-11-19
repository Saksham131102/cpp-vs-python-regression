# **🚀 Compute Benchmarks**
## **Comparing C++ vs Python for Multivariate Linear Regression**
This repository benchmarks the performance of **multivariate linear regression** implemented in **both C++ and Python**, using:
- **Native C++**
- **C++ with Eigen**
- **Native Python**
- **Python with NumPy**

The results highlight how low-level optimizations and high-performance libraries impact computational speed—especially for linear algebra–heavy machine learning tasks.

## **📊 Benchmark Results**
| Implementation        | Time Taken |
|----------------------|------------|
| **Python (native)**  | `~4.9761090 s` |
| **C++ (native)**     | `~1.8601100 s` |
| **Python + NumPy**   | `~0.5322770 s` |
| **C++ + Eigen**      | `~0.0531356 s` |

## **🔍 Key Insight**
Even though Python is slower in its pure form, its libraries like **NumPy** are heavily optimized in **C/C++**, which is why they drastically outperform native Python.

On the other hand, when C++ is paired with an optimized math library like **Eigen**, it delivers **an order of magnitude faster performance** than NumPy.

## **🧠 Why Python Libraries Are So Fast**
Most major Python ML and scientific libraries (NumPy, Pandas, Scikit-Learn, PyTorch, TensorFlow) are written in **C, C++, or Fortran** under the hood.
Python acts mainly as a **wrapper**, making them easy to use while still achieving near-C++ performance.

## **⚙️ Compilation (for C++)**
### **Native C++**
```
g++ -std=c++17 native_regression.cpp -o run1
```

### **C++ + Eigen**
```
g++ -std=c++17 -O3 -march=native -ffast-math eigen_regression.cpp -o run2
```
- ```-O3``` enables aggressive optimizations like inlining, loop unrolling, and vectorization.
- ```-march=native``` lets the compiler use all CPU-specific instruction sets (like AVX/AVX2/AVX-512) which greatly speeds up math-heavy operations.
- ```-ffast-math``` allows the compiler to ignore strict IEEE floating-point rules for faster operations

## **🧪 Run Benchmarks**

### **Python**
```
python3 native_regression.py
python3 numpy_regression.py
```

### **C++**
```
./run1
./run2
```

## **🏁 Conclusion**

This experiment clearly shows the power of **low-level optimizations** and why Python can compete with C++—thanks to its C/C++-powered libraries.
But when both languages use optimized libraries, **C++ still comes out on top in raw speed**.