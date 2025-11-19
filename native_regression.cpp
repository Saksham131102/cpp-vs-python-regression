#include<bits/stdc++.h>
using namespace std;

class Normalizer {
public:
  vector<double> mean;
  vector<double> sd;

  // computing mean and sd
  // here const means modification to X is not allowed
  void fit(const vector<vector<double>> &X) {
    int n = X.size();
    int m = X[0].size();
    
    mean.assign(m, 0.0);
    sd.assign(m, 0.0);

    // Calculating Mean for each feature
    for(int i=0;i<X.size();i++) {
      for(int j=0;j<X[0].size();j++) {
        mean[j] += X[i][j];
      }
    }

    for(int i=0;i<mean.size();i++) {
      mean[i] = mean[i] / n;
    }

    // Calculating standard deviation for each feature
    for(int i=0;i<X.size();i++) {
      for(int j=0;j<X[0].size();j++) {
        sd[j] += (X[i][j] - mean[j]) * (X[i][j] - mean[j]);
      }
    }

    for(int i=0;i<sd.size();i++) {
      sd[i] = sd[i] / n;
      sd[i] = sqrt(sd[i]);
    }
  }

   // const here means this function can't update the data members of this class.
  vector<vector<double>> transform(vector<vector<double>> X) const {
    vector<vector<double>> X_norm(X.size(), vector<double> (X[0].size()));
    for(int i=0;i<X.size();i++) {
      for(int j=0;j<X[0].size();j++) {
        X_norm[i][j] = (X[i][j] - mean[j]) / sd[j];
      }
    }

    return X_norm;
  }
};

double compute_cost(const vector<double> &W, const double &b, const vector<vector<double>> &X, const vector<double> &y) {
  double cost = 0.0;

  size_t n = X.size();
  if(n == 0)
    return 0.0;
  
  size_t m = X[0].size();

  if(W.size() != m || y.size() != n) {
    throw invalid_argument("Dimensions of X, W or y do not match");
  }

  for(int i=0;i<n;i++) {
    // double y_hat = w * x[i] + b
    double y_hat = b;
    for(int j=0;j<m;j++) {
      y_hat += (W[j] * X[i][j]);
    }
    cost += (y_hat - y[i]) * (y_hat - y[i]);
  }

  return cost / (2.0*n);
}

void compute_gradient(vector<double> &W, double &b, const vector<vector<double>> &X, const vector<double> &y, const double alpha) {
  size_t n = X.size();
  size_t m = X[0].size();

  vector<double> dw(m, 0.0);
  double db = 0.0;

  for(int i=0;i<n;i++) {
    double y_hat = b;
    for(int j=0;j<m;j++) {
      y_hat += (W[j] * X[i][j]);
    }
    for(int j=0;j<m;j++) {
      dw[j] += (y_hat - y[i]) * X[i][j];
    }
    db += (y_hat - y[i]);
  }

  for(int j=0;j<m;j++) {
    dw[j] = dw[j] / n;
  }
  db = db / n;

  for(int j=0;j<m;j++) {
    W[j] = W[j] - alpha * dw[j];
  }
  b = b - alpha * db;
}

int main() {

  auto start = chrono::high_resolution_clock::now(); // start timer

  vector<vector<double>> X = {
    {-122.23,37.88,41.0,880.0,129.0,322.0,126.0,8.3252},
    {-122.22,37.86,21.0,7099.0,1106.0,2401.0,1138.0,8.3014},
    {-122.24,37.85,52.0,1467.0,190.0,496.0,177.0,7.2574},
    {-122.25,37.85,52.0,1274.0,235.0,558.0,219.0,5.6431},
    {-122.25,37.85,52.0,1627.0,280.0,565.0,259.0,3.8462},
    {-122.25,37.85,52.0,919.0,213.0,413.0,193.0,4.0368},
    {-122.25,37.84,52.0,2535.0,489.0,1094.0,514.0,3.6591},
    {-122.25,37.84,52.0,3104.0,687.0,1157.0,647.0,3.12},
    {-122.26,37.84,42.0,2555.0,665.0,1206.0,595.0,2.0804},
    {-122.25,37.84,52.0,3549.0,707.0,1551.0,714.0,3.6912},
    {-122.26,37.85,52.0,2202.0,434.0,910.0,402.0,3.2031},
    {-122.26,37.85,52.0,3503.0,752.0,1504.0,734.0,3.2705},
    {-122.26,37.85,52.0,2491.0,474.0,1098.0,468.0,3.075},
    {-122.26,37.84,52.0,696.0,191.0,345.0,174.0,2.6736},
    {-122.26,37.85,52.0,2643.0,626.0,1212.0,620.0,1.9167},
    {-122.26,37.85,50.0,1120.0,283.0,697.0,264.0,2.125},
    {-122.27,37.85,52.0,1966.0,347.0,793.0,331.0,2.775},
    {-122.27,37.85,52.0,1228.0,293.0,648.0,303.0,2.1202},
    {-122.26,37.84,50.0,2239.0,455.0,990.0,419.0,1.9911},
    {-122.27,37.84,52.0,1503.0,298.0,690.0,275.0,2.6033},
    {-122.27,37.85,40.0,751.0,184.0,409.0,166.0,1.3578},
    {-122.27,37.85,42.0,1639.0,367.0,929.0,366.0,1.7135},
    {-122.27,37.84,52.0,2436.0,541.0,1015.0,478.0,1.725},
    {-122.27,37.84,52.0,1688.0,337.0,853.0,325.0,2.1806},
    {-122.27,37.84,52.0,2224.0,437.0,1006.0,422.0,2.6},
    {-122.28,37.85,41.0,535.0,123.0,317.0,119.0,2.4038},
    {-122.28,37.85,49.0,1130.0,244.0,607.0,239.0,2.4597},
    {-122.28,37.85,52.0,1898.0,421.0,1102.0,397.0,1.808},
    {-122.28,37.84,50.0,2082.0,492.0,1131.0,473.0,1.6424},
    {-122.28,37.84,52.0,729.0,160.0,395.0,155.0,1.6875},
    {-122.28,37.84,49.0,1916.0,447.0,863.0,378.0,1.9274},
    {-122.28,37.84,52.0,2153.0,481.0,1168.0,441.0,1.9615},
    {-122.27,37.84,48.0,1922.0,409.0,1026.0,335.0,1.7969},
    {-122.27,37.83,49.0,1655.0,366.0,754.0,329.0,1.375},
    {-122.27,37.83,51.0,2665.0,574.0,1258.0,536.0,2.7303},
    {-122.27,37.83,49.0,1215.0,282.0,570.0,264.0,1.4861},
    {-122.27,37.83,48.0,1798.0,432.0,987.0,374.0,1.0972},
    {-122.28,37.83,52.0,1511.0,390.0,901.0,403.0,1.4103},
    {-122.26,37.83,52.0,1470.0,330.0,689.0,309.0,3.48},
    {-122.26,37.83,52.0,2432.0,715.0,1377.0,696.0,2.5898},
    {-122.26,37.83,52.0,1665.0,419.0,946.0,395.0,2.0978},
    {-122.26,37.83,51.0,936.0,311.0,517.0,249.0,1.2852},
    {-122.26,37.84,49.0,713.0,202.0,462.0,189.0,1.025},
    {-122.26,37.84,52.0,950.0,202.0,467.0,198.0,3.9643},
    {-122.26,37.83,52.0,1443.0,311.0,660.0,292.0,3.0125},
    {-122.26,37.83,52.0,1656.0,420.0,718.0,382.0,2.6768},
    {-122.26,37.83,50.0,1125.0,322.0,616.0,304.0,2.026},
    {-122.27,37.82,43.0,1007.0,312.0,558.0,253.0,1.7348},
    {-122.26,37.82,40.0,624.0,195.0,423.0,160.0,0.9506},
    {-122.27,37.82,40.0,946.0,375.0,700.0,352.0,1.775}
  };

  vector<double> y = {
    452600.0,358500.0,352100.0,341300.0,342200.0,269700.0,
    299200.0,241400.0,226700.0,261100.0,281500.0,241800.0,
    213500.0,191300.0,159200.0,140000.0,152500.0,155500.0,
    158700.0,162900.0,147500.0,159800.0,113900.0,99700.0,
    132600.0,107500.0,93800.0,105500.0,108900.0,132000.0,
    122300.0,115200.0,110400.0,104900.0,109700.0,97200.0,
    104500.0,103900.0,191400.0,176000.0,155400.0,150000.0,
    118800.0,188800.0,184400.0,182300.0,142500.0,137500.0,
    187500.0,112500.0
  };

  // Normalize the data
  cout << "Normalizing the data!!!\n";
  Normalizer normalizer;
  normalizer.fit(X);
  vector<vector<double>> X_norm = normalizer.transform(X);
  cout << "Normalization completed!!!\n";

  int m = X[0].size();

  // Parameters
  vector<double> W(m, 0.0);
  double b = 0.0;

  // Hyper Parameters
  double alpha = 0.01;
  double iterations = 100000;

  cout << "Initial value of W's and b\n";
  for(int i=0;i<m;i++) {
    cout << "w_" << (i+1) << ": " << W[i] << (i < m-1 ? " | " : "");
  }
  cout << "\nb: " << b << "\n";

  cout << "Initial Cost: " << compute_cost(W, b, X_norm, y);

  // Model Learning
  for(int i=0;i<iterations;i++) {
    // compute gradient descent
    compute_gradient(W, b, X_norm, y, alpha);
    // for(int j=0;j<m;j++) {
    //   cout << "w_" << (j+1) << ": " << W[j] << (j < m-1 ? " | " : "");
    // }
    // cout << "\nb: " << b << "\n";

    // cout << "Cost: " << compute_cost(W, b, X_norm, y);
  }

  cout << "\n\nFinal parameters\n";
  for(int j=0;j<m;j++) {
    cout << "w_" << (j+1) << ": " << W[j] << (j < m-1 ? " | " : "");
  }
  cout << "\nb: " << b << "\n";

  cout << "Cost: " << compute_cost(W, b, X_norm, y);

  

  // Doing prediction

  cout << "\n\nDoing Prediction\n\n";
  vector<double> x_test = {-122.22,37.8,52.0,2183.0,465.0,1129.0,460.0,3.2632};
  vector<double> x_test_norm(x_test.size());

  // normalizing x_test
  for(int j=0;j<x_test.size();j++) {
    x_test_norm[j] = (x_test[j] - normalizer.mean[j]) / normalizer.sd[j];
  }

  double y_pred = b;
  for(int j=0;j<x_test_norm.size();j++) {
    y_pred += (W[j] * x_test_norm[j]);
  }

  cout << "Predicted Price: " << y_pred << "\n";

  auto end = chrono::high_resolution_clock::now(); // end timer 
  chrono::duration<double> duration = end - start;

  cout << "Time taken by Native code: " << duration.count() << "seconds\n";

  return 0;
}