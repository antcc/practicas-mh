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
int seed = 20;

// Random engine generator
default_random_engine gen;


/*************************************************************************************/
/* RUN ALGORITHMS
/*************************************************************************************/

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

    for (int j = 0; j < K; j++)
      if (j != i)
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
    run_p2("data/colposcopy_normalizados.csv");

    // Dataset 2: ionosphere
    run_p2("data/ionosphere_normalizados.csv");

    // Dataset 3: texture
    run_p2("data/texture_normalizados.csv");
  }
}
