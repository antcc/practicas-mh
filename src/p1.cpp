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
#include "random.h"
#include "util.h"
#include "timer.h"

using namespace std;

#define DEBUG 1

// ------------------------- Constants ------------------------------------

const float alpha = 0.5;

// ------------------------- Functions ------------------------------------

// 1-nearest neighbour
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

// 1-nearest neighbour with weights
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

// Calculate nearest example of same class and nearest example of different class
void nearest_examples(const vector<Example>& training, const Example& e,
                      int self, Example& n_friend, Example& n_enemy) {
  double dmin_friend = numeric_limits<double>::max();
  double dmin_enemy = numeric_limits<double>::max();
  double dist;

  for (int i = 1; i < self; i++) {
    dist = distance_sq(training[i], e);

    if (training[i].category != e.category && dist < dmin_enemy) {
      n_enemy = training[i];
      dmin_enemy = dist;
    }

    else if (training[i].category == e.category && dist < dmin_friend) {
      n_friend = training[i];
      dmin_friend = dist;
    }
  }

  for (int i = self + 1; i < e.n; i++) {
    dist = distance_sq(training[i], e);

    if (training[i].category != e.category && dist < dmin_enemy) {
      n_enemy = training[i];
      dmin_enemy = dist;
    }

    else if (training[i].category == e.category && dist < dmin_friend) {
      n_friend = training[i];
      dmin_friend = dist;
    }
  }
}

// Greedy algorithm to compute weights
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
    cerr << endl << endl;

    // Normalize
    for (int j = 0; j < w.size(); j++) {
      if (w[j] < 0) w[j] = 0;
      else w[j] = w[j] / max;
    }
  }
}

float class_rate(const vector<string>& classified, const vector<Example>& test) {
  int correct = 0;

  for (int i = 0; i < classified.size(); i++)
    if (classified[i] == test[i].category)
      correct++;

  return (float) correct / classified.size();
}

float red_rate(const vector<double>& w, int n) {
  if (w.empty())
    return 0.0;

  int discarded = 0;

  for (auto weight : w)
    if (weight < 0.2)
      discarded++;

  return (float) discarded / n;
}

float objective(float class_rate, float red_rate) {
  return alpha * class_rate + (1 - alpha) * red_rate;
}

void run(const string& filename) {
  cout << "----------------------------------------------------------" << endl;
  cout << "CONJUNTO DE DATOS: " << filename << endl;
  cout << "----------------------------------------------------------" << endl << endl;

  // Read dataset from file
  vector<Example> dataset;
  read_csv(filename, dataset);

  // Make partitions to train/test
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
    int n = test[0].n;

    clear_vector(w);

    for (int j = 0; j < i; j++)
      training.insert(training.end(), partitions[j].begin(), partitions[j].end());
    for (int j = i + 1; j < K; j++)
      training.insert(training.end(), partitions[j].begin(), partitions[j].end());

    // Algorithm 1: 1-NN
    cout << "---------" << endl;
    cout << "1-NN " << endl;
    cout << "---------" << endl << endl;

    start_timers();
    for (auto e : test)
      classified.push_back(classifier_1nn(e, training));
    double time_w = elapsed_time();

    // Update results
    float class_rate_w = class_rate(classified, test);
    float red_rate_w = red_rate(w, n);
    float objective_w = objective(class_rate_w, red_rate_w);

    class_rate_acum[0] += class_rate_w;
    red_rate_acum[0] += red_rate_w;
    objective_acum[0] += objective_w;
    time_acum[0] += time_w;

    // Print partial results
    cout << "Tasa de clasificación parcial: " << class_rate_w << endl;
    cout << "Tasa de reducción parcial: " << red_rate_w << endl;
    cout << "Agregado parcial: " << objective_w << endl;
    cout << "Tiempo empleado parcial: " << time_w << endl << endl;

    /******************************************************************************/

    // Algorithm 2: RELIEF
    cout << "---------" << endl;
    cout << "RELIEF " << endl;
    cout << "---------" << endl << endl;

    start_timers();
    relief(training, w);
    for (auto e : test)
      classified.push_back(classifier_1nn_weights(e, training, w));
    time_w = elapsed_time();

    // Update results
    class_rate_w = class_rate(classified, test);
    red_rate_w = red_rate(w, n);
    objective_w = objective(class_rate_w, red_rate_w);

    class_rate_acum[1] += class_rate_w;
    red_rate_acum[1] += red_rate_w;
    objective_acum[1] += objective_w;
    time_acum[1] += time_w;

    // Print partial results
    if (DEBUG) {
      cout << "Vector de pesos:\n";
      for (auto weight : w)
        cout << weight << ", ";
      cout << endl << endl;
    }

    cout << "Tasa de clasificación parcial: " << class_rate_w << endl;
    cout << "Tasa de reducción parcial: " << red_rate_w << endl;
    cout << "Agregado parcial: " << objective_w << endl;
    cout << "Tiempo empleado parcial: " << time_w << endl << endl;

    clear_vector(w);

  }

  // Print global (averaged) results
  cout << "----- Resultados globales 1-NN -----" << endl;
  cout << "Tasa de clasificación global: " << class_rate_acum[0] / K << endl;
  cout << "Tasa de reducción global: " << red_rate_acum[0] / K << endl;
  cout << "Agregado global: " << objective_acum[0] / K << endl;
  cout << "Tiempo empleado global: " << time_acum[0] / K << endl << endl;

  cout << "----- Resultados globales RELIEF -----" << endl;
  cout << "Tasa de clasificación global: " << class_rate_acum[1] / K << endl;
  cout << "Tasa de reducción global: " << red_rate_acum[1] / K << endl;
  cout << "Agregado global: " << objective_acum[1] / K << endl;
  cout << "Tiempo empleado global: " << time_acum[1] / K << endl << endl;
}

// ------------------------- Main function ------------------------------------

int main() {
  // Dataset 1: texture
  run("data/texture_normalizados.csv");
}
