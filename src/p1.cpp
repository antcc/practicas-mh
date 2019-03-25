/**
 * Metaheurísticas.
 *
 * Problema: APC
 * Práctica 1: algoritmo greedy y búsqueda local.
 *
 * Antonio Coín Castro.
 */

#include <iostream>
#include <limits>
#include <cmath>
#include <algorithm>
#include <random>
#include "util.h"
#include "timer.h"

using namespace std;

#define DEBUG 0

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
const int SEED = 20;

// Random engine generator
default_random_engine gen(SEED);

// ------------------------------ Functions -----------------------------------------

/*************************************************************************************/
/* CLASSIFIER
/*************************************************************************************/

// 1-nearest neighbour
// @cond e.n == training[i].n
string classifier_1nn(const Example& e, const vector<Example>& training) {
  string category = training[0].category;
  double dmin = distance_sq(e, training[0]);

  for (int i = 1; i < training.size(); i++) {
    double dist = distance_sq(e, training[i]);

    if (dist < dmin) {
      category = training[i].category;
      dmin = dist;
    }
  }
  return category;
}

// 1-nearest neighbour with weights (using leave-one-out strategy)
// @param self Position of example @e in vector @training, or -1 if it's not in it.
// @cond e.n == training[i].n
string classifier_1nn_weights(const Example& e, const vector<Example>& training,
                              int self, const vector<double>& w) {
  string category;
  double dmin = numeric_limits<double>::max();

  for (int i = 0; i < self; i++) {
    double dist = distance_sq_weights(e, training[i], w);

    if (dist < dmin) {
      category = training[i].category;
      dmin = dist;
    }
  }

  for (int i = self + 1; i < training.size(); i++) {
    double dist = distance_sq_weights(e, training[i], w);

    if (dist < dmin) {
      category = training[i].category;
      dmin = dist;
    }
  }

  return category;
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
void relief(const vector<Example>& training, vector<double>& w) {
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
    if (w_mut[comp] < 0) w_mut[comp] = 0;

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

#if DEBUG == 1
  cout << "Iteraciones: " << iter << endl << endl;
#endif

  return mut;
}

/*************************************************************************************/
/* RUN ALGORITHMS
/*************************************************************************************/

// Run every algorithm for a particular dataset and print results
void run(const string& filename) {
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
  float class_rate_acum[3] = {0.0, 0.0, 0.0};
  float red_rate_acum[3] = {0.0, 0.0, 0.0};
  float objective_acum[3] = {0.0, 0.0, 0.0};
  double time_acum[3] = {0.0, 0.0, 0.0};

  // Weight vector
  vector<double> w;
  w.resize(partitions[0][0].n);

  // Use every possible partition as test
  for (int i = 0; i < K; i++) {
    cout << "----- Ejecución " << i + 1 << " -----" << endl << endl;

    vector<Example> training;
    vector<Example> test = partitions[i];
    vector<string> classified;

    for (int j = 0; j < i; j++)
      training.insert(training.end(), partitions[j].begin(), partitions[j].end());
    for (int j = i + 1; j < K; j++)
      training.insert(training.end(), partitions[j].begin(), partitions[j].end());

    // Algorithm 1: 1-NN
    cout << "---------" << endl;
    cout << "1-NN " << endl;
    cout << "---------" << endl;

    start_timers();
    for (auto e : test)
      classified.push_back(classifier_1nn(e, training));
    double time_w = elapsed_time();

    // Update results
    float class_rate_w = class_rate(classified, test);
    float red_rate_w = 0;
    float objective_w = objective(class_rate_w, red_rate_w);

    class_rate_acum[0] += class_rate_w;
    red_rate_acum[0] += red_rate_w;
    objective_acum[0] += objective_w;
    time_acum[0] += time_w;

    // Print partial results
    cout << "Tasa de clasificación parcial: " << class_rate_w << "%" << endl;
    cout << "Tasa de reducción parcial: " << red_rate_w << "%" << endl;
    cout << "Agregado parcial: " << objective_w << endl;
    cout << "Tiempo empleado parcial: " << time_w << " ms" << endl << endl;

    classified.clear();

    /******************************************************************************/

    // Algorithm 2: RELIEF
    cout << "---------" << endl;
    cout << "RELIEF " << endl;
    cout << "---------" << endl;

    start_timers();
    relief(training, w);
    for (auto e : test)
      classified.push_back(classifier_1nn_weights(e, training, -1, w));
    time_w = elapsed_time();

    // Update results
    class_rate_w = class_rate(classified, test);
    red_rate_w = red_rate(w);
    objective_w = objective(class_rate_w, red_rate_w);

    class_rate_acum[1] += class_rate_w;
    red_rate_acum[1] += red_rate_w;
    objective_acum[1] += objective_w;
    time_acum[1] += time_w;

#if DEBUG == 1
      cout << "Vector de pesos (RELIEF):\n";
      for (auto weight : w)
        cout << weight << ", ";
      cout << endl << endl;
#endif

    // Print partial results
    cout << "Tasa de clasificación parcial: " << class_rate_w << "%" << endl;
    cout << "Tasa de reducción parcial: " << red_rate_w << "%" << endl;
    cout << "Agregado parcial: " << objective_w << endl;
    cout << "Tiempo empleado parcial: " << time_w << " ms" << endl << endl;

    classified.clear();

    /******************************************************************************/

    // Algorithm 3: LOCAL SEARCH
    cout << "---------" << endl;
    cout << "BÚSQUEDA LOCAL " << endl;
    cout << "---------" << endl;

    start_timers();
    int mut = local_search(training, w);
    for (auto e : test)
      classified.push_back(classifier_1nn_weights(e, training, -1, w));
    time_w = elapsed_time();

    // Update results
    class_rate_w = class_rate(classified, test);
    red_rate_w = red_rate(w);
    objective_w = objective(class_rate_w, red_rate_w);

    class_rate_acum[2] += class_rate_w;
    red_rate_acum[2] += red_rate_w;
    objective_acum[2] += objective_w;
    time_acum[2] += time_w;

    #if DEBUG == 1
      cout << "Vector de pesos (BÚSQUEDA LOCAL):\n";
      for (auto weight : w)
        cout << weight << ", ";
      cout << endl << endl;
    #endif

    // Print partial results
    cout << "Tasa de clasificación parcial: " << class_rate_w << "%" << endl;
    cout << "Tasa de reducción parcial: " << red_rate_w << "%" << endl;
    cout << "Agregado parcial: " << objective_w << endl;
    cout << "Tiempo empleado parcial: " << time_w << " ms" << endl;
    cout << "Mutaciones: " << mut << endl << endl;

    classified.clear();
  }

  // Print global (averaged) results
  cout << "----- Resultados globales 1-NN -----" << endl;
  cout << "Tasa de clasificación global: " << class_rate_acum[0] / K << "%" << endl;
  cout << "Tasa de reducción global: " << red_rate_acum[0] / K << "%" << endl;
  cout << "Agregado global: " << objective_acum[0] / K << endl;
  cout << "Tiempo empleado global: " << time_acum[0] / K << " ms" << endl << endl;

  cout << "----- Resultados globales RELIEF -----" << endl;
  cout << "Tasa de clasificación global: " << class_rate_acum[1] / K << "%" << endl;
  cout << "Tasa de reducción global: " << red_rate_acum[1] / K << "%" << endl;
  cout << "Agregado global: " << objective_acum[1] / K << endl;
  cout << "Tiempo empleado global: " << time_acum[1] / K << " ms" << endl << endl;

  cout << "----- Resultados globales BÚSQUEDA LOCAL -----" << endl;
  cout << "Tasa de clasificación global: " << class_rate_acum[2] / K << "%" << endl;
  cout << "Tasa de reducción global: " << red_rate_acum[2] / K << "%" << endl;
  cout << "Agregado global: " << objective_acum[2] / K << endl;
  cout << "Tiempo empleado global: " << time_acum[2] / K << " ms" << endl << endl;
}

// ------------------------- Main function ------------------------------------

int main() {

  // Dataset 1: ionosphere
  run("data/ionosphere_normalizados.csv");

  // Dataset 2: colposcopy
  run("data/colposcopy_normalizados.csv");

  // Dataset 3: texture
  run("data/texture_normalizados.csv");
}
