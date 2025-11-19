#include<bits/stdc++.h>
using namespace std;

double calculate_cost(double &w, double &b, vector<double> &x, vector<double> &y) {
  double error_sum = 0.0;
  int m = x.size();
  for(int i=0;i<m;i++) {
    double y_hat = w * x[i] + b;
    error_sum += (y_hat - y[i]) * (y_hat - y[i]);
  }
  double cost  = (error_sum) / (2 * m);
  return cost;
}

void compute_gradient(vector<double> &x, vector<double> &y, double &w, double & b, double &alpha) {
  int m = x.size();
  double dw = 0.0;
  double db = 0.0;

  for(int i=0;i<m;i++) {
    double y_hat = w * x[i] + b;
    dw = dw + (y_hat - y[i]) * x[i];
    db = db + (y_hat - y[i]);
  }

  dw = dw / m;
  db = db / m;

  w = w - alpha * dw;
  b = b - alpha * db;
}

int main() {
  vector<double> x = {1,2,3,4,5};
  vector<double> y = {3,5,7,9,11};

  double w = 0.0;
  double b = 0.0;
  double alpha = 0.01;
  double iterations = 1000;

  double cost = calculate_cost(w, b, x, y);

  for(int i=0;i<iterations;i++) {
    compute_gradient(x, y, w, b, alpha); // will update 'w' and 'b'
    cout << "updated w & b: " << w << " | " << b << "\n";
    cout << "cost: " << calculate_cost(w, b, x, y) << "\n";
  }

  cout << "\n\nFinal parameters\n";
  cout << "w: " << w << "\n";
  cout << "b: " << b << "\n";
  cout << "Final cost: " << calculate_cost(w, b, x, y) << "\n";
  return 0;
}