/**
 * Metaheurísticas.
 *
 * Problema: APC
 *
 * Práctica 1: algoritmo greedy y búsqueda local.
 *
 * Antonio Coín Castro.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cctype>
#include <string>
#include <algorithm>
#include <unordered_map>

using namespace std;

const int K = 5;

// ------------------------ Helper functions -------------------------------------------------

inline std::string trim(const std::string &s)
{
   auto wsfront=std::find_if_not(s.begin(),s.end(),[](int c){return std::isspace(c);});
   auto wsback=std::find_if_not(s.rbegin(),s.rend(),[](int c){return std::isspace(c);}).base();
   return (wsback<=wsfront ? std::string() : std::string(wsfront,wsback));
}

// ------------------------- Data structures and functions ------------------------------------

struct Example {
  vector<double> traits;
  string category;
  int n;
};

void read_csv(string filename, vector<Example>& result) {
  ifstream file(filename);
  string line;

  while (getline(file, line)) {
    Example e;
    istringstream s(line);
    string field;

    while (getline(s, field, ';')) {
      if (s.peek() != '\n' && s.peek() != EOF)
        e.traits.push_back(stod(field));
    }

    e.category = trim(field);
    e.n = e.traits.size();
    result.push_back(e);
  }
}

// K-fold cross validation
vector<vector<Example>> make_partitions(const vector<Example>& data) {
  unordered_map<string, int> category_count;
  vector<vector<Example>> partitions;

  for (int i = 0; i < K; i++)
    partitions.push_back(vector<Example>());

  for (auto& ex : data) {
    int count = category_count[ex.category];
    partitions[count].push_back(ex);

    category_count[ex.category] = (category_count[ex.category] + 1) % K;
  }

  return partitions;
}

// @cond e1.n == e2.n
double distance_sq(const Example& e1, const Example& e2) {
  double distance = 0.0;
  for (int i = 0; i < e1.n; i++)
    distance += (e2.traits[i] - e1.traits[i]) * (e2.traits[i] - e1.traits[i]);
  return distance;
}

// @cond e1.n == e2.n
double distance_sq_weights(const Example& e1, const Example& e2, const vector<double>& w) {
  double distance = 0.0;
  for (int i = 0; i < e1.n; i++)
    distance += w[i] * (e2.traits[i] - e1.traits[i]) * (e2.traits[i] - e1.traits[i]);
  return distance;
}

// @cond e.n == training[i].n
string classifier_1nn(const Example& e, const vector<Example>& training) {
  string category = training[0].category;
  double dmin = distance_sq(e, training[0]);

  for (int i = 1; i < e.n; i++) {
    double dist = distance_sq(e, training[i]);

    if (dist < dmin) {
      category = training[i].category;
      dmin = dist;
    }
  }
  return category;
}

// @cond e.n == training[i].n
string classifier_1nn_weights(const Example& e, const vector<Example>& training,
                              const vector<double>& w) {
  string category = training[0].category;
  double dmin = distance_sq(e, training[0]);

  for (int i = 1; i < e.n; i++) {
    double dist = distance_sq_weights(e, training[i], w);

    if (dist < dmin) {
      category = training[i].category;
      dmin = dist;
    }
  }
  return category;
}

void nearest_examples(const vector<Example>& training, const Example& e,
                        int self, Example& n_friend, Example& n_enemy) {
  double dmin_friend = numeric_limits<double>::max();
  double dmin_enemy = numeric_limits<double>::max();
  double dist;

  for (int i = 1; i < self; i++) {
    dist = distance_sq(training[i], e);
    if (training[i].category != e.category && dist < dmin_enemy)
      dmin_enemy = dist;
    else if (training[i].category == e.category && dist < dmin_friend)
      dmin_friend = dist;
  }

  for (int i = self + 1; i < e.n; i++) {
    dist = distance_sq(training[i], e);
    if (training[i].category != e.category && dist < dmin_enemy)
      dmin_enemy = dist;
    else if (training[i].category == e.category && dist < dmin_friend)
      dmin_friend = dist;
  }
}

// @cond w[i] = 0 && w.size() == training[i].n
void relief(const vector<Example>& training, vector<double>& w) {
  for (int i = 0; i < training.size(); i++) {
    Example n_friend, n_enemy;

    // Find nearest friend and enemy
    nearest_examples(training, training[i], i, n_friend, n_enemy);

    // Update W and calculate maximum
    double max = 0.0;
    for (int j = 0; j < w.size(); j++) {
      w[j] = w[j] + abs(training[i].traits[j] - n_enemy.traits[j])
             - abs(training[i].traits[j] - n_friend.traits[j]);

      if (w[j] > max)
        max = w[j];
    }

    // Normalize
    for (int j = 0; j < w.size(); j++) {
      if (w[j] < 0) w[j] = 0;
      else w[j] = w[j] / max;
    }
  }
}

// ------------------------- Main function ------------------------------------

int main() {
  vector<Example> ionosphere;
  read_csv("../../data/colposcopy_normalizados.csv", ionosphere);

  auto partitions = make_partitions(ionosphere);

  // Use every possible partition as test
  for (int i = 0; i < K; i++) {

  }
}
