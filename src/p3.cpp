/**
 * Metaheurísticas.
 *
 * Problema: APC
 * Práctica 3: enfriamiento simulado, búsqueda local reiterada
 * y evolución diferencial.
 *
 * Antonio Coín Castro.
 */

#include <iostream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <random>
#include <set>
#include "util.h"
#include "timer.h"

using namespace std;

#define DEBUG 0
#define TABLE 1

// ----------------------- Constants and global variables -------------------------------

// -- Simulated annealing --

// Parameters for calculating initial temperature
const float phi = 0.3;
const float mu = 0.3;

// Final temperature
float final_temp = 1e-3;

// Maximum number of neighbours generated per trait
const int MAX_NEIGHBOUR_PER_TRAIT = 10;

// Maximum number of successful generations per neighbour
const float MAX_SUCCESS_PER_NEIGHBOUR = 0.1;

// -- General --

// Measures importance of classification and reduction rates.
const float alpha = 0.5;

// Standard deviation for normal distribution
const float sigma = 0.3;

// Maximum number of iterations
const int MAX_ITER = 15000;

// Number of algorithms
constexpr int NUM_ALGORITHMS = 4;

// Names of algorithms
const string algorithms_names[NUM_ALGORITHMS] = {
  "ES",
  "ILS",
  "DE/rand/1",
  "DE/current-to-best/1"
};

// Seed for randomness
long seed = 2019;

// Random engine generator
default_random_engine generator;

// ---------------------------- Data strutures --------------------------------------

struct Solution {
  vector<double> w;
  float fitness;
};

// ------------------------------ Functions -----------------------------------------

/***********************************************************************************/
/* CLASSIFIER
/***********************************************************************************/

// 1-nearest neighbour with weights (using leave-one-out strategy)
// @param self Position of example @e in vector @training, or -1 if it's not in it.
// @cond e.n == training[i].n
string classifier_1nn_weights(const Example& e, const vector<Example>& training,
                              int self, const vector<double>& w) {
  int selected = 0;
  double dmin = numeric_limits<double>::max();

  for (int i = 0; i < training.size(); i++) {
    if (i != self) {
      double dist = distance_sq_weights(e, training[i], w);

      if (dist < dmin) {
        selected = i;
        dmin = dist;
      }
    }
  }
  return training[selected].category;
}

/***********************************************************************************/
/* STATISTICS
/***********************************************************************************/

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
float evaluate(const vector<Example>& training, const vector<double>& w) {
  vector<string> classified;

  for (int i = 0; i < training.size(); i++)
    classified.push_back(classifier_1nn_weights(training[i], training, i, w));

  return objective(class_rate(classified, training), red_rate(w));
}

/***********************************************************************************/
/* COMMON OPERATORS
/***********************************************************************************/

// Initialize solution with size n
Solution init_solution(const vector<Example> training, int n) {
  Solution sol;
  uniform_real_distribution<double> random_real(0.0, 1.0);

  sol.w.resize(n);
  for (int i = 0; i < n; i++)
    sol.w[i] = random_real(generator);
  sol.fitness = evaluate(training, sol.w);

  return sol;
}

// Mutate a component of a weight vector
void mutate(vector<double>& w, int comp) {
  normal_distribution<double> normal(0.0, sigma);
  w[comp] += normal(generator);

  if (w[comp] < 0.0) w[comp] = 0.0;
  if (w[comp] > 1.0) w[comp] = 1.0;
}

/***********************************************************************************/
/* SIMULATED ANNEALING
/***********************************************************************************/

void simulated_annealing(const vector<Example>& training, vector<double>& w) {
  Solution sol, best_sol;
  float temp;
  float initial_temp;
  int iter;
  int successful;
  int neighbour;
  int n = w.size();
  uniform_int_distribution<int> random_int(0, n - 1);
  uniform_real_distribution<double> random_real(0.0, 1.0);

  // 1. Initialize solution and temperature
  sol = init_solution(training, n);
  best_sol = sol;
  iter++;
  initial_temp = (mu * best_sol.fitness) / (- 1.0 * log(phi));
  temp = initial_temp;

  if (final_temp >= temp)
    final_temp = temp / 100.0;

#if DEBUG >= 1
  cerr << endl << "Temperatura final: " << final_temp << endl << endl;
#endif

  const int MAX_NEIGHBOUR = MAX_NEIGHBOUR_PER_TRAIT * n;
  const int MAX_SUCCESS = MAX_SUCCESS_PER_NEIGHBOUR * MAX_NEIGHBOUR;
  const int M = MAX_ITER / MAX_NEIGHBOUR;
  const float beta = (float) (initial_temp - final_temp) / (M * initial_temp * final_temp);

  // 2. Outer loop
  successful = MAX_SUCCESS;
  iter = 0;
  while(iter < MAX_ITER && successful != 0) {
    neighbour = 0;
    successful = 0;

#if DEBUG >= 1
    cerr << "----------- Temperatura: " << temp << " ---------------" << endl;
    cerr << "Iteraciones: " << iter << endl << endl;
#endif

    // 3. Inner loop (cooling)
    while(iter < MAX_ITER && neighbour < MAX_NEIGHBOUR && successful < MAX_SUCCESS) {
      // 4. Mutate random component
      int comp = random_int(generator);
      Solution sol_mut = sol;
      mutate(sol_mut.w, comp);
      sol_mut.fitness = evaluate(training, sol_mut.w);
      iter++;
      neighbour++;

      // 5. Acceptance criterion
      float diff = sol.fitness - sol_mut.fitness;  // We are maximizing the fitness

      if (diff < 0 || random_real(generator) <= exp(-1.0 * diff / temp)) {
        successful++;
        sol = sol_mut;
        if (sol.fitness > best_sol.fitness)
          best_sol = sol;
      }

#if DEBUG >= 2
      cerr << "Vecinos: " << neighbour << endl;
      cerr << "Éxitos: " << successful << endl;
      cerr << "Mejor fitness actual: " << best_sol.fitness << endl << endl;
#endif

    }
    // 6. Cool-down (Cauchy scheme)
    temp = temp / (1.0 + beta * temp);
  }

#if DEBUG >= 1
  cerr << "Iteraciones: " << iter << endl << endl;
#endif

  w = best_sol.w;

}

/***********************************************************************************/
/* ITERATED LOCAL SEARCH
/***********************************************************************************/

void ils(const vector<Example>& training, vector<double>& w) {

}

/***********************************************************************************/
/* DIFFERENTIAL EVOLUTION
/***********************************************************************************/

void de_rand(const vector<Example>& training, vector<double>& w) {

}

void de_current_to_best(const vector<Example>& training, vector<double>& w) {

}

/***********************************************************************************/
/* RUN ALGORITHMS
/***********************************************************************************/

// Print results
void print_results(bool global, float class_rate, float red_rate,
                   float objective, float time) {
  string type = global ? "global" : "parcial";
  cout << "Tasa de clasificación " << type << ": " << class_rate << "%" << endl;
  cout << "Tasa de reducción " << type << ": " << red_rate << "%" << endl;
  cout << "Agregado " << type << ": " << objective << endl;
  cout << "Tiempo empleado " << type << ": " << time << " s" << endl << endl;
}

// Print result in LaTeX table format
void print_results_table(int partition, float class_rate, float red_rate,
                         float objective, float time) {
  cout << fixed << setprecision(2)
       << (partition == 0 ? "" : to_string(partition)) << " & " << class_rate << " & "
       << red_rate << " & " << objective << " & " << time << endl;
}

// Run every algorithm for a particular dataset and print results
void run_p3(const string& filename) {

#if TABLE < 2
  cout << "----------------------------------------------------------" << endl;
  cout << "CONJUNTO DE DATOS: " << filename << endl;
  cout << "----------------------------------------------------------" << endl << endl;
#endif

  // Read dataset from file
  vector<Example> dataset;
  read_csv(filename, dataset);

  // Make partitions to train/test
  shuffle(dataset.begin(), dataset.end(), generator);
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
    simulated_annealing,
    ils,
    de_rand,
    de_current_to_best
  };

  // Run every algorithm
  for (int p = 0; p < 1; p++) {

#if TABLE < 2
    cout << "---------" << endl;
    cout << algorithms_names[p] << endl;
    cout << "---------" << endl << endl;
#endif

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
      double time_w = elapsed_time() / 1000.0;

      // Update results
      float class_rate_w = class_rate(classified, test);
      float red_rate_w = red_rate(w);
      float objective_w = objective(class_rate_w, red_rate_w);

      class_rate_acum[p] += class_rate_w;
      red_rate_acum[p] += red_rate_w;
      objective_acum[p] += objective_w;
      time_acum[p] += time_w;

#if DEBUG >= 3
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

#if TABLE < 2
  cout << "------------------------------------------" << endl << endl;
#endif

  for (int p = 0;  p < 1; p++) {

#if TABLE < 2
    cout << "----- Resultados globales " << algorithms_names[p] << " -----" << endl << endl;
#endif

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

    for (int i = 2; i < argc; i++) {
      generator = default_random_engine(seed);
      run_p3(argv[i]);
    }
  }

  else {
    generator = default_random_engine(seed);

    // Dataset 1: colposcopy
    run_p3("data/colposcopy_normalizados.csv");

    generator = default_random_engine(seed);

    // Dataset 2: ionosphere
    run_p3("data/ionosphere_normalizados.csv");

    generator = default_random_engine(seed);

    // Dataset 3: texture
    run_p3("data/texture_normalizados.csv");
  }
}
