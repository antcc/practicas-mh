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

using namespace std;

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

    // Normalize
    for (int j = 0; j < w.size(); j++) {
      if (w[j] < 0) w[j] = 0;
      else w[j] = w[j] / max;
    }
  }
}

// ------------------------- Main function ------------------------------------

int main() {
  // Load 3 datasets
  vector<Example> ionosphere;
  read_csv("../../data/colposcopy_normalizados.csv", ionosphere);

  auto partitions = make_partitions(ionosphere);

  // Use every possible partition as test
  for (int i = 0; i < K; i++) {
    
  }
}
