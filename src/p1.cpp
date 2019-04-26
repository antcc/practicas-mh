/**
 * Metaheurísticas.
 *
 * Problema: APC
 * Práctica 1: algoritmo greedy y búsqueda local.
 *
 * Antonio Coín Castro.
 */

#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <functional>
#include <algorithm>
#include <random>
#include "util.h"
#include "timer.h"

using namespace std;

#define DEBUG 0
#define TABLE 1

// ----------------------- Constants and global variables -------------------------------

// Measures importance of classification and reduction rates.
const float alpha = 0.5;

// Standard deviation for normal distribution
const float sigma = 0.3;

// Maximum number of iterations for local search method
const int MAX_ITER = 15000;

// Upper bound for neighbour generation in local search method
const int MAX_NEIGHBOUR_PER_TRAIT = 20;

// Seed for randomness
int seed = 2019;

// Random engine generator
default_random_engine gen;

// Number of algorithms
constexpr int NUM_ALGORITHMS = 2;

// Names of algorithms
const string algorithms_names[NUM_ALGORITHMS] = {
  "RELIEF",
  "BÚSQUEDA LOCAL"
};

// ------------------------------ Functions -----------------------------------------

/*************************************************************************************/
/* CLASSIFIER
/*************************************************************************************/

// 1-nearest neighbour
// @cond e.n == training[i].n
string classifier_1nn(const Example& e, const vector<Example>& training) {
  int selected = 0;
  double dmin = distance_sq(e, training[0]);

  for (int i = 1; i < training.size(); i++) {
    double dist = distance_sq(e, training[i]);

    if (dist < dmin) {
      selected = i;
      dmin = dist;
    }
  }
  return training[selected].category;
}

// 1-nearest neighbour with weights (using leave-one-out strategy)
// @param self Position of example @e in vector @training, or -1 if it's not in it.
// @cond e.n == training[i].n
string classifier_1nn_weights(const Example& e, const vector<Example>& training,
                              int self, const vector<double>& w) {
  int selected = 0;
  double dmin = numeric_limits<double>::max();

  for (int i = 0; i < self; i++) {
    double dist = distance_sq_weights(e, training[i], w);

    if (dist < dmin) {
      selected = i;
      dmin = dist;
    }
  }

  for (int i = self + 1; i < training.size(); i++) {
    double dist = distance_sq_weights(e, training[i], w);

    if (dist < dmin) {
      selected = i;
      dmin = dist;
    }
  }
  return training[selected].category;
}

/*************************************************************************************/
/* RELIEF
/*************************************************************************************/

// Calculate nearest example of same class and nearest example of different class
// @cond Each example must have at least one enemy and one friend (apart from itself)
// @cond training[i].n == e.n
void nearest_examples(const vector<Example>& training, const Example& e,
                      int self, int& n_friend, int& n_enemy) {
  double dist;
  double dmin_friend = numeric_limits<double>::max();
  double dmin_enemy = numeric_limits<double>::max();

  for (int i = 0; i < self; i++) {
    dist = distance_sq(training[i], e);

    if (training[i].category != e.category && dist < dmin_enemy) {
      n_enemy = i;
      dmin_enemy = dist;
    }

    else if (training[i].category == e.category && dist < dmin_friend) {
      n_friend = i;
      dmin_friend = dist;
    }
  }

  // Skip ourselves
  for (int i = self + 1; i < training.size(); i++) {
    dist = distance_sq(training[i], e);

    if (training[i].category != e.category && dist < dmin_enemy) {
      n_enemy = i;
      dmin_enemy = dist;
    }

    else if (training[i].category == e.category && dist < dmin_friend) {
      n_friend = i;
      dmin_friend = dist;
    }
  }
}

// Greedy algorithm to compute weights
// @cond w.size() == training[i].n
// Does not return anything.
int relief(const vector<Example>& training, vector<double>& w) {
  // Set w[i] = 0
  init_vector(w);

  for (int i = 0; i < training.size(); i++) {
    int n_friend = 0;
    int n_enemy = 0;

    // Find nearest friend and enemy
    nearest_examples(training, training[i], i, n_friend, n_enemy);

    // Update W
    for (int j = 0; j < w.size(); j++) {
      w[j] = w[j] + fabs(training[i].traits[j] - training[n_enemy].traits[j])
             - fabs(training[i].traits[j] - training[n_friend].traits[j]);
    }
  }

  // Normalize weights
  double max = *max_element(w.begin(), w.end());
  for (int j = 0; j < w.size(); j++) {
    if (w[j] < 0) w[j] = 0.0;
    else w[j] = (double) w[j] / max;
  }

  return 0;
}

/*************************************************************************************/
/* STATISTICS
/*************************************************************************************/

// Return classification rate in test dataset
// @cond classified.size() == test.size()
float class_rate(const vector<string>& classified, const vector<Example>& test) {
  int correct = 0;

  for (int i = 0; i < classified.size(); i++)
    if (classified[i] == test[i].category)
      correct++;

  return 100.0 * correct / classified.size();
}

// Return reduction rate of traits
float red_rate(const vector<double>& w) {
  int discarded = 0;

  for (auto weight : w)
    if (weight < 0.2)
      discarded++;

  return 100.0 * discarded / w.size();
}

// Return objective function
float objective(float class_rate, float red_rate) {
  return alpha * class_rate + (1.0 - alpha) * red_rate;
}

/*************************************************************************************/
/* LOCAL SEARCH
/*************************************************************************************/

// Best-first local search method to compute weights
// w.size() == training[i].n
// @return Number of mutations that have improven the solution
int local_search(const vector<Example>& training, vector<double>& w) {
  normal_distribution<double> normal(0.0, sigma);
  uniform_real_distribution<double> uniform_real(0.0, 1.0);
  const int n = w.size();
  vector<string> classified;
  vector<int> index;
  double best_objective;
  int iter = 0;
  int neighbour = 0;
  bool improvement = false;
  int mut = 0;

  // Initialize index vector and solution
  for (int i = 0; i < n; i++) {
    index.push_back(i);
    w[i] = uniform_real(gen);
  }

  shuffle(index.begin(), index.end(), gen);

  // Evaluate initial solution
  for (int i = 0; i < training.size(); i++)
    classified.push_back(classifier_1nn_weights(training[i], training, i, w));

  best_objective = objective(class_rate(classified, training), red_rate(w));
  classified.clear();

  // Best-first search
  while (iter < MAX_ITER && neighbour < n * MAX_NEIGHBOUR_PER_TRAIT) {
    // Select component to mutate
    int comp = index[iter % n];

    // Mutate w
    vector<double> w_mut = w;
    w_mut[comp] += normal(gen);

    // Truncate weights
    if (w_mut[comp] > 1) w_mut[comp] = 1;
    else if (w_mut[comp] < 0) w_mut[comp] = 0;

    // Acceptance criterion
    for (int i = 0; i < training.size(); i++)
      classified.push_back(classifier_1nn_weights(training[i], training, i, w_mut));

    double current_objective = objective(class_rate(classified, training), red_rate(w_mut));
    iter++;

    if (current_objective > best_objective) {
      mut++;
      neighbour = 0;
      w = w_mut;
      best_objective = current_objective;
      improvement = true;
    }

    else {
      neighbour++;
    }

    classified.clear();

    // Update index vector if needed
    if (iter % n == 0 || improvement) {
      shuffle(index.begin(), index.end(), gen);
      improvement = false;
    }
  }

#if DEBUG == 2
  cout << "Iteraciones: " << iter << endl << endl;
#endif

  return mut;
}

/*************************************************************************************/
/* RUN ALGORITHMS
/*************************************************************************************/

// Print results
void print_results(bool global, float class_rate, float red_rate,
                   float objective, float time) {
  string type = global ? "global" : "parcial";
  cout << "Tasa de clasificación " << type << ": " << class_rate << "%" << endl;
  cout << "Tasa de reducción " << type << ": " << red_rate << "%" << endl;
  cout << "Agregado " << type << ": " << objective << endl;
  cout << "Tiempo empleado " << type << ": " << time << " ms" << endl << endl;
}

// Print result in LaTeX table format
void print_results_table(int partition, float class_rate, float red_rate,
                         float objective, float time) {
  cout << fixed << setprecision(2)
       << (partition == 0 ? "" : to_string(partition)) << " & " << class_rate << " & "
       << red_rate << " & " << objective << " & " << time << endl;
}

// Run every algorithm for a particular dataset and print results
void run_p1(const string& filename) {
  cout << "----------------------------------------------------------" << endl;
  cout << "CONJUNTO DE DATOS: " << filename << endl;
  cout << "----------------------------------------------------------" << endl << endl;

  // Read dataset from file
  vector<Example> dataset;
  read_csv(filename, dataset);

  // Make partitions to train/test
  shuffle(dataset.begin(), dataset.end(), gen);
  auto partitions = make_partitions(dataset);

  // Accumulated statistical values
  float class_rate_acum[NUM_ALGORITHMS + 1] = {0.0};
  float red_rate_acum[NUM_ALGORITHMS + 1] = {0.0};
  float objective_acum[NUM_ALGORITHMS + 1] = {0.0};
  double time_acum[NUM_ALGORITHMS + 1] = {0.0};

  // Weight vector
  vector<double> w;
  w.resize(partitions[0][0].n);

  // List of every algorithm
  function<int(const vector<Example>&, vector<double>&)> algorithms[NUM_ALGORITHMS] = {
    relief,
    local_search
  };

  // Run standard 1-nn
  // Run every algorithm to compute weights
    cout << "---------" << endl;
    cout << "1-NN" << endl;
    cout << "---------" << endl << endl;

    // Use every possible partition as test
    for (int i = 0; i < K; i++) {

#if TABLE == 0
      cout << "----- Ejecución " << i + 1 << " -----" << endl << endl;
#endif

      vector<Example> training;
      vector<Example> test = partitions[i];
      vector<string> classified;

      // Create test/training partitions
      for (int j = 0; j < K; j++)
        if (j != i)
          training.insert(training.end(), partitions[j].begin(), partitions[j].end());

      // Run algorithm and collect data
      start_timers();
      for (auto e : test)
        classified.push_back(classifier_1nn(e, training));
      double time_w = elapsed_time();

      // Update results
      float class_rate_w = class_rate(classified, test);
      float red_rate_w = 0.0;
      float objective_w = objective(class_rate_w, red_rate_w);

      class_rate_acum[0] += class_rate_w;
      red_rate_acum[0] += red_rate_w;
      objective_acum[0] += objective_w;
      time_acum[0] += time_w;

#if DEBUG == 3
      cout << "Solución:\n[";
      for (auto weight : w)
        cout << weight << ", ";
      cout << "]" << endl << endl;
#endif

      // Print partial results

#if TABLE == 0
      print_results(false, class_rate_w, red_rate_w, objective_w, time_w);
#elif TABLE == 1
      print_results_table(i + 1, class_rate_w, red_rate_w, objective_w, time_w);
#endif
    }

  // Run every algorithm to compute weights
  for (int p = 0; p < NUM_ALGORITHMS; p++) {
    cout << "---------" << endl;
    cout << algorithms_names[p] << endl;
    cout << "---------" << endl << endl;

    // Use every possible partition as test
    for (int i = 0; i < K; i++) {

#if TABLE == 0
      cout << "----- Ejecución " << i + 1 << " -----" << endl << endl;
#endif

      vector<Example> training;
      vector<Example> test = partitions[i];
      vector<string> classified;

      // Create test/training partitions
      for (int j = 0; j < K; j++)
        if (j != i)
          training.insert(training.end(), partitions[j].begin(), partitions[j].end());

      // Run algorithm and collect data
      start_timers();
      algorithms[p](training, w);  // Call algorithm
      for (auto e : test)
        classified.push_back(classifier_1nn_weights(e, training, -1, w));
      double time_w = elapsed_time();

      // Update results
      float class_rate_w = class_rate(classified, test);
      float red_rate_w = red_rate(w);
      float objective_w = objective(class_rate_w, red_rate_w);

      class_rate_acum[p+1] += class_rate_w;
      red_rate_acum[p+1] += red_rate_w;
      objective_acum[p+1] += objective_w;
      time_acum[p+1] += time_w;

#if DEBUG == 3
      cout << "Solución:\n[";
      for (auto weight : w)
        cout << weight << ", ";
      cout << "]" << endl << endl;
#endif

      // Print partial results

#if TABLE == 0
      print_results(false, class_rate_w, red_rate_w, objective_w, time_w);
#elif TABLE == 1
      print_results_table(i + 1, class_rate_w, red_rate_w, objective_w, time_w);
#endif

      // Clear classification vector
      classified.clear();
    }
  }

  // Print global (averaged) results
  cout << "------------------------------------------" << endl << endl;
  for (int p = 0;  p < NUM_ALGORITHMS + 1; p++) {
    cout << "----- Resultados globales " << (p == 0 ? "1-NN" : algorithms_names[p-1])
         << " -----" << endl << endl;

      // Print partial results

#if TABLE == 0
      print_results(true, class_rate_acum[p] / K, red_rate_acum[p] / K,
                    objective_acum[p] / K, time_acum[p] / K);
#elif TABLE == 1
      print_results_table(0, class_rate_acum[p] / K, red_rate_acum[p] / K,
                          objective_acum[p] / K, time_acum[p] / K);
#endif

  }
}

// ------------------------- Main function ------------------------------------

int main(int argc, char * argv[]) {
  if (argc > 1) {
    seed = stoi(argv[1]);

    gen = default_random_engine(seed);

    for (int i = 2; i < argc; i++)
      run_p1(argv[i]);
  }

  else {
    gen = default_random_engine(seed);

    // Dataset 1: colposcopy
    run_p1("data/colposcopy_normalizados.csv");

    // Dataset 2: ionosphere
    run_p1("data/ionosphere_normalizados.csv");

    // Dataset 3: texture
    run_p1("data/texture_normalizados.csv");
  }
}
