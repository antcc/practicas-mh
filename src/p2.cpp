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
#include <functional>
#include <random>
#include <set>
#include "util.h"
#include "timer.h"

using namespace std;

#define DEBUG 0

// ----------------------- Constants and global variables -------------------------------

// Measures importance of classification and reduction rates.
const float alpha = 0.5;

// Parameter of BLX cross operator
const float alpha_blx = 0.3;

// Standard deviation for normal distribution
const float sigma = 0.3;

// Maximum number of iterations
const int MAX_ITER = 15000;

// Upper bound for neighbour generation in low-intensity local search method
const int MAX_NEIGHBOUR_PER_TRAIT = 2;

// Seed for randomness
int seed = 20;

// Random engine generator
default_random_engine gen;

// Number of algorithms
constexpr int NUM_ALGORITHMS = 7;

// Names of algorithms
const string algorithms_names[NUM_ALGORITHMS] = {
  "AGG-BLX",
  "AGG-CA",
  "AGE-BLX",
  "AGE-CA",
  "AM-(10, 1.0)",
  "AM-(10, 0.1)",
  "AM-(10, 0.1 mej)"
};

// ------------------------------ Data structures -----------------------------------------

// Chromosome
struct Chromosome {
  vector<double> w;  // Weight vector that represents the chromosome
  float fitness;  // Value of the objective function for w
};

// Population of chromosomes with custom comparator
struct ChromosomeComp {
  bool operator()(const Chromosome& lhs, const Chromosome& rhs) {
    return lhs.fitness < rhs.fitness;
  }
};

typedef multiset<Chromosome, ChromosomeComp> Population;

// ------------------------------ Functions -----------------------------------------

/*************************************************************************************/
/* CLASSIFIER
/*************************************************************************************/

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

// Return the value of the objective function
float objective(float class_rate, float red_rate) {
  return alpha * class_rate + (1.0 - alpha) * red_rate;
}

// Evaluate a solution
float evaluate(const vector<Example>& training, const vector<double> w) {
  vector<string> classified;

  for (int i = 0; i < training.size(); i++)
    classified.push_back(classifier_1nn_weights(training[i], training, i, w));

  return objective(class_rate(classified, training), red_rate(w));

}

/*************************************************************************************/
/* LOW-INTENSITY LOCAL SEARCH
/*************************************************************************************/

// Low-intensity local search method to compute weights
// w.size() == training[i].n
// @return Number of mutations that have improven the solution
int low_intensity_local_search(const vector<Example>& training, vector<double> w) {
  normal_distribution<double> normal(0.0, sigma);
  const int n = w.size();
  vector<int> index;
  double best_objective;
  int neighbour = 0;
  bool improvement = false;
  int mut = 0;
  int j = 0;

  // Initialize index vector
  for (int i = 0; i < n; i++)
    index.push_back(i);
  shuffle(index.begin(), index.end(), gen);

  // Evaluate initial solution
  best_objective = evaluate(training, w);

  // Best-first search
  while (neighbour < n * MAX_NEIGHBOUR_PER_TRAIT) {
    // Select component to mutate
    int comp = j++;

    // Mutate w
    vector<double> w_mut = w;
    w_mut[comp] += normal(gen);

    // Truncate weights
    if (w_mut[comp] > 1) w_mut[comp] = 1;
    else if (w_mut[comp] < 0) w_mut[comp] = 0;

    // Acceptance criterion
    double current_objective = evaluate(training, w_mut);

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

    // Update index vector if needed
    if (j == n || improvement) {
      shuffle(index.begin(), index.end(), gen);
      improvement = false;
      j = 0;
    }
  }

  return mut;
}

/*************************************************************************************/
/* GENETIC OPERATORS
/*************************************************************************************/

// Blx cross operator
pair<Chromosome, Chromosome> blx_cross(const Chromosome& c1, const Chromosome& c2) {

}

// Arithmetic cross operator
Chromosome arithmetic_cross(const Chromosome& c1, const Chromosome& c2) {

}

// Mutation operator
void mutate(Chromosome& c) {

}

// Initialize population
void init_population(Population& population) {

}

/*************************************************************************************/
/* GENERATIONAL GENETIC ALGORITHM (AGG)
/*************************************************************************************/

void agg_blx(const vector<Example> training, vector<double>& w) {
  Population population;
}

void agg_ca(const vector<Example> training, vector<double>& w) {

}

/*************************************************************************************/
/* STEADY STATE GENETIC ALGORITHM (AGE)
/*************************************************************************************/

void age_blx(const vector<Example> training, vector<double>& w) {

}

void age_ca(const vector<Example> training, vector<double>& w) {

}

/*************************************************************************************/
/* MEMETIC ALGORITHM (AM)
/*************************************************************************************/

void am_1(const vector<Example> training, vector<double>& w) {

}

void am_2(const vector<Example> training, vector<double>& w) {

}

void am_3(const vector<Example> training, vector<double>& w) {

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
  cout << "Agregado " << type << ": " << objective / K << endl;
  cout << "Tiempo empleado " << type << ": " << time << " ms" << endl << endl;
}

// Run every algorithm for a particular dataset and print results
void run_p2(const string& filename) {
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
  float class_rate_acum[NUM_ALGORITHMS] = {0.0};
  float red_rate_acum[NUM_ALGORITHMS] = {0.0};
  float objective_acum[NUM_ALGORITHMS] = {0.0};
  double time_acum[NUM_ALGORITHMS] = {0.0};

  // Weight vector
  vector<double> w;
  w.resize(partitions[0][0].n);

  // List of every algorithm
  function<void(const vector<Example>&, vector<double>&)> algorithms[NUM_ALGORITHMS] = {
    agg_blx,
    agg_ca,
    age_blx,
    age_ca,
    am_1,
    am_2,
    am_3
  };

  // Run every algorithm
  for (int p = 0; p < 1; p++) {  // FIXME: bucle completo hasta NUM_ALGORITHMS
    cout << "---------" << endl;
    cout << algorithms_names[p] << endl;
    cout << "---------" << endl << endl;

    // Use every possible partition as test
    for (int i = 0; i < K; i++) {
      cout << "----- Ejecución " << i + 1 << " -----" << endl << endl;

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

      class_rate_acum[p] += class_rate_w;
      red_rate_acum[p] += red_rate_w;
      objective_acum[p] += objective_w;
      time_acum[p] += time_w;

#if DEBUG == 1
      cout << "Vector de pesos:\n[";
      for (auto weight : w)
        cout << weight << ", ";
      cout << "]" << endl << endl;
#endif

      // Print partial results
      print_results(false, class_rate_w, red_rate_w, objective_w, time_w);

      // Clear classification vector
      classified.clear();
    }
  }

  // Print global (averaged) results
  cout << "------------------------------------------" << endl << endl;
  for (int p = 0;  p < 1; p++) { // FIXME: bucle completo hasta NUM_ALGORITHMS
    cout << "----- Resultados globales " << algorithms_names[p] << " -----" << endl << endl;
    print_results(true, class_rate_acum[p] / K, red_rate_acum[p] / K,
                  objective_acum[p] / K, time_acum[p] / K);
  }
}

// ------------------------- Main function ------------------------------------

int main(int argc, char * argv[]) {
  if (argc > 1) {
    seed = stoi(argv[1]);

    gen = default_random_engine(seed);

    for (int i = 2; i < argc; i++)
      run_p2(argv[i]);
  }

  else {
    gen = default_random_engine(seed);

    // Dataset 1: colposcopy
    run_p2("data/colposcopy_normalizados.csv");

    // Dataset 2: ionosphere
    //run_p2("data/ionosphere_normalizados.csv");

    // Dataset 3: texture
    //run_p2("data/texture_normalizados.csv");
  }
}
