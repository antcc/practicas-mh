#include <vector>
#include <string>

using namespace std;

// ------------------------- Global constants ------------------------------------

const int K = 5;  //< Number of folds

// ------------------------- Data structures ------------------------------------

// Represents a data point
struct Example {
  vector<double> traits;
  string category;
  int n;
};

// ------------------------------ Functions -----------------------------------------

// Trim string from both sides
inline std::string trim(const std::string &s);
// Parse CSV file
void read_csv(string filename, vector<Example>& result);

// K-fold cross validation
vector<vector<Example>> make_partitions(const vector<Example>& data);

// Distance squared between two data points
// @cond e1.n == e2.n
double distance_sq(const Example& e1, const Example& e2);
// Distance squared between two data points (considering weights)
// @cond e1.n == e2.n
double distance_sq_weights(const Example& e1, const Example& e2, const vector<double>& w);

// Set all values of a vector<double> to 0.0
void clear_vector(vector<double>& w);
