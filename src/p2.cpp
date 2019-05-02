/**
 * Metaheurísticas.
 *
 * Problema: APC
 * Práctica 2: algoritmos genéticos y meméticos.
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

// Size of (intermediate) population for generational genetic algorithms.
const int SIZE_AGG = 30;

// Size of (intermediate) population for steady state genetic algorithms.
const int SIZE_AGE = 2;

// Size of population for memetic algorithms. Must be an even positive integer
const int SIZE_AM = 10;

// Frequency for applying local search in memetic algorithms
const int FREQ_BL = 10;

// Cross probability
const float pc = 0.7;

// Mutation probability
const float pm = 0.001;

// Local search probaility for memetic algorithms
const float pls = 0.1;

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

// Seed for randomness
int seed = 2019;

// Random engine generator
default_random_engine generator;

// ------------------------------ Data structures -----------------------------------------

// Chromosome
struct Chromosome {
  vector<double> w;  // Weight vector that represents the chromosome
  float fitness;     // Value of the objective function for w
};

// Custom comparator for chromosomes
struct ChromosomeComp {
  bool operator()(const Chromosome& lhs, const Chromosome& rhs) {
    return lhs.fitness < rhs.fitness;
  }
};

// Population
typedef multiset<Chromosome, ChromosomeComp> Population;

// Intermediate population (non-evaluated)
typedef vector<Chromosome> IntermediatePopulation;

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
// @return Number of evaluations of the objective function
int low_intensity_local_search(const vector<Example>& training, Chromosome& c) {
  normal_distribution<double> normal(0.0, sigma);
  const int n = c.w.size();
  vector<int> index;
  double best_objective;
  int iter = 0;

  // Initialize index vector
  for (int i = 0; i < n; i++)
    index.push_back(i);
  shuffle(index.begin(), index.end(), generator);

  // Evaluate initial solution
  best_objective = c.fitness;

  // Best-first search
  while (iter < n * MAX_NEIGHBOUR_PER_TRAIT) {
    // Select component to mutate
    int comp = index[iter % n];

    // Mutate w
    Chromosome c_mut = c;
    c_mut.w[comp] += normal(generator);

    // Truncate weights
    if (c_mut.w[comp] > 1) c_mut.w[comp] = 1;
    else if (c_mut.w[comp] < 0) c_mut.w[comp] = 0;

    // Acceptance criterion
    c_mut.fitness = evaluate(training, c_mut.w);
    iter++;

    if (c_mut.fitness > best_objective) {
      c = c_mut;
      best_objective = c_mut.fitness;
    }

    // Update index vector if needed
    if (iter % n == 0)
      shuffle(index.begin(), index.end(), generator);
  }

  return iter;
}

/*************************************************************************************/
/* GENETIC OPERATORS
/*************************************************************************************/

// Initialize population
void init_population(Population& pop, int num_chromosomes, int size,
                     const vector<Example>& training) {
  uniform_real_distribution<double> random_real(0.0, 1.0);

  for (int i = 0; i < num_chromosomes; i++) {
    Chromosome c;
    c.w.resize(size);

    for (int j = 0; j < size; j++)
      c.w[j] = random_real(generator);

    c.fitness = evaluate(training, c.w);
    pop.insert(c);
  }
}

// Selection operation
// Binary tournament with replacement
// @return The selected chromosome
Chromosome selection(const Population& pop) {
  uniform_int_distribution<int> random_int(0, pop.size() - 1);

  // Get iterators to two random chromosomes
  auto p1 = pop.begin();
  auto p2 = pop.begin();
  advance(p1, random_int(generator));
  advance(p2, random_int(generator));

  return p1->fitness > p2->fitness ? *p1 : *p2;
}

// Blx cross operator
// Generates two descendants for every two parents
// @cond c1.w.size() == c2.w.size()
pair<Chromosome, Chromosome> blx_cross(const Chromosome& c1, const Chromosome& c2) {
  Chromosome h1, h2;

  h1.w.resize(c1.w.size());
  h2.w.resize(c1.w.size());

  for (int i = 0; i < c1.w.size(); i++ ) {
    float cmin = min(c1.w[i], c2.w[i]);
    float cmax = max(c1.w[i], c2.w[i]);
    float diff = cmax - cmin;

    uniform_real_distribution<float>
      random_real(cmin - diff * alpha_blx, cmax + diff * alpha_blx);

    // NOTE: if c1.w[i] == c2.w[i], then h1.w[i] = h2.w[i] = c1.w[i]

    h1.w[i] = random_real(generator);
    h2.w[i] = random_real(generator);

    // Truncate
    if (h1.w[i] < 0) h1.w[i] = 0.0;
    if (h1.w[i] > 1) h1.w[i] = 1.0;
    if (h2.w[i] < 0) h2.w[i] = 0.0;
    if (h2.w[i] > 1) h2.w[i] = 1.0;
  }

  h1.fitness = -1.0;
  h2.fitness = -1.0;

  return make_pair(h1, h2);
}

// Arithmetic cross operator
// Generates one descendant for every two parents
// @cond c1.w.size() == c2,w.size()
Chromosome arithmetic_cross(const Chromosome& c1, const Chromosome& c2) {
  Chromosome h;

  h.w.resize(c1.w.size());

  for (int i = 0; i < c1.w.size(); i++)
    h.w[i] = (c1.w[i] + c2.w[i]) / 2.0;

  h.fitness = -1.0;

  return h;
}

// Mutation operator
// Mutates a given gene of a chromosome
// @cond 0 <= comp <= c.w.size()
void mutate(Chromosome& c, int comp) {
  normal_distribution<double> normal(0.0, sigma);
  c.w[comp] += normal(generator);
  c.fitness = -1.0;

  // Truncate
  if (c.w[comp] < 0) c.w[comp] = 0.0;
  if (c.w[comp] > 1) c.w[comp] = 1.0;
}

// Return expected number of mutations
// Uses custom "rounding" method
int expected_mutations(int total_genes) {
  float expected_mut = pm * total_genes;

  // We want diversity
  if (expected_mut <= 1.0)
    return 1;

  float remainder = modf(expected_mut, &expected_mut);
  uniform_real_distribution<double> random_real(0.0, 1.0);
  double u = random_real(generator);
  if (u <= remainder)
    expected_mut++;

  return expected_mut;
}

/***********************************************************************************/
/* GENERATIONAL GENETIC ALGORITHM (AGG)
/***********************************************************************************/

// AGG using BLX cross operator
// @return Total generations
int agg_blx(const vector<Example> training, vector<double>& w) {
  Population pop;
  Population::reverse_iterator best_parent;  // Elitism
  int iter = 0;
  int age = 1;
  int total_genes = w.size() * SIZE_AGG;
  int num_cross = pc * (SIZE_AGG / 2);  // Expected crosses
  uniform_int_distribution<int> random_int(0, total_genes - 1);

#if DEBUG >= 1
  cout << "[AGG-BLX] Genes por cromosoma: " << w.size() << endl;
#endif

  // 1. Build and evaluate initial population
  init_population(pop, SIZE_AGG, w.size(), training);
  iter += SIZE_AGG;

  while (iter < MAX_ITER) {
    IntermediatePopulation pop_temp;
    Population new_pop;

    best_parent = pop.rbegin();  // Save best parent for elitism

#if DEBUG >= 1
      cout << "[AGG-BLX] Mejor fitness actual: "
           << best_parent->fitness << endl;
#endif

    // 2. Select intermediate population (already evaluated)
    pop_temp.resize(SIZE_AGG);
    for (int i = 0; i < SIZE_AGG; i++) {
      pop_temp[i] = selection(pop);
    }

    // 3. Recombine intermediate population
    for (int i = 0; i < 2 * num_cross; i += 2) {
      auto offspring = blx_cross(pop_temp[i], pop_temp[i+1]);
      pop_temp[i] = offspring.first;
      pop_temp[i+1] = offspring.second;
    }

    // 4. Mutate intermediate population
    set<int> mutated;
    int num_mut = expected_mutations(total_genes);
    for (int i = 0; i < num_mut; i++) {
      int comp;

      // Select (coded) component to mutate, without repetition
      while(mutated.size() == i) {
        comp = random_int(generator);
        mutated.insert(comp);
      }

      int selected = comp / w.size();
      int gene = comp % w.size();

      mutate(pop_temp[selected], gene);

#if DEBUG >= 2
      cout << "[AGG-BLX] Mutación " << i + 1 << ":"
           << " gen " << comp % w.size() << " del cromosoma "
           << (comp / w.size()) << endl;
#endif

    }

    // 5. Evaluate, replace original population and apply elitism
    for (int i = 0; i < SIZE_AGG; i++) {
      if (pop_temp[i].fitness == -1.0) {
        pop_temp[i].fitness = evaluate(training, pop_temp[i].w);
        iter++;
      }
      new_pop.insert(pop_temp[i]);
    }

    auto current_best = new_pop.rbegin();

#if DEBUG >= 1
      cout << "[AGG-BLX] Mejor fitness intermedio: "
           << current_best->fitness << endl;
#endif

    if (current_best->fitness < best_parent->fitness) {

#if DEBUG >= 1
      cout << "[AGG-BLX] Reemplazo elitista" << endl;
#endif

      // Replace worst chromosome of intermediate population
      new_pop.erase(new_pop.begin());
      new_pop.insert(*best_parent);
    }

    // 6. Replace previous population entirely (new generation)
    pop = new_pop;
    age++;

#if DEBUG >= 1
    cout << "[AGG-BLX] Número de iteraciones: " << iter << " -----------" << endl;
#endif

#if TABLE == 2
    cout << iter << " " << pop.rbegin()->fitness << endl;
#endif

  }

  // Choose best chromosome as solution
  w = pop.rbegin()->w;

#if DEBUG >= 1
  cout << "[AGG-BLX] Fitness solución: " << pop.rbegin()->fitness << endl;
#endif

  return age;
}

// AGG using arithmetic cross operator
// @return Total generations
int agg_ca(const vector<Example> training, vector<double>& w) {
  Population pop;
  Population::reverse_iterator best_parent;  // Elitism
  int iter = 0;
  int age = 1;
  int total_genes = w.size() * SIZE_AGG;
  int num_cross = pc * (SIZE_AGG / 2);  // Expected crosses
  uniform_int_distribution<int> random_int(0, total_genes - 1);

#if DEBUG >= 1
  cout << "[AGG-CA] Genes por cromosoma: " << w.size() << endl;
#endif

  // 1. Build and evaluate initial population
  init_population(pop, SIZE_AGG, w.size(), training);
  iter += SIZE_AGG;

  while (iter < MAX_ITER) {
    IntermediatePopulation pop_temp;
    Population new_pop;

    best_parent = pop.rbegin();  // Save best parent for elitism

#if DEBUG >= 1
      cout << "[AGG-CA] Mejor fitness actual: "
           << best_parent->fitness << endl;
#endif

    // 2. Select intermediate population (already evaluated)
    // NOTE: we select 2 * SIZE_AGG because we only get one descendant for
    // every two parents, even though we might not use every selected chromosome.
    pop_temp.resize(2 * SIZE_AGG);
    for (int i = 0; i < 2 * SIZE_AGG; i++) {
      pop_temp[i] = selection(pop);
    }

    // 3. Recombine intermediate population
    for (int i = 0; i < 2 * num_cross; i++) {
      pop_temp[i] = arithmetic_cross(pop_temp[i], pop_temp[2 * SIZE_AGG - i - 1]);
    }

    // 4. Mutate intermediate population
    set<int> mutated;
    int num_mut = expected_mutations(total_genes);  // Expected mutations
    for (int i = 0; i < num_mut; i++) {
      int comp;

      // Select (coded) component to mutate, without repetition
      while(mutated.size() == i) {
        comp = random_int(generator);
        mutated.insert(comp);
      }

      int selected = comp / w.size();
      int gene = comp % w.size();

      mutate(pop_temp[selected], gene);

#if DEBUG >= 2
      cout << "[AGG-CA] Mutación " << i + 1 << ":"
           << " gen " << comp % w.size() << " del cromosoma "
           << (comp / w.size()) << endl;
#endif

    }

    // 5. Evaluate, replace original population and apply elitism
    for (int i = 0; i < SIZE_AGG; i++) {
      if (pop_temp[i].fitness == -1.0) {
        pop_temp[i].fitness = evaluate(training, pop_temp[i].w);
        iter++;
      }
      new_pop.insert(pop_temp[i]);
    }

    auto current_best = new_pop.rbegin();

#if DEBUG >= 1
      cout << "[AGG-CA] Mejor fitness intermedio: "
           << current_best->fitness << endl;
#endif

    if (current_best->fitness < best_parent->fitness) {

#if DEBUG >= 1
      cout << "[AGG-CA] Reemplazo elitista" << endl;
#endif

      // Replace worst chromosome of intermediate population
      new_pop.erase(new_pop.begin());
      new_pop.insert(*best_parent);
    }

    // 6. Replace previous population entirely (new generation)
    pop = new_pop;
    age++;

#if DEBUG >= 1
    cout << "[AGG-CA] Número de iteraciones: " << iter << " -----------" << endl;
#endif

#if TABLE == 2
    cout << iter << " " << pop.rbegin()->fitness << endl;
#endif

  }

  // Choose best chromosome as solution
  w = pop.rbegin()->w;

#if DEBUG >= 1
  cout << "[AGG-CA] Fitness solución: " << pop.rbegin()->fitness << endl;
#endif

  return age;
}

/*************************************************************************************/
/* STEADY STATE GENETIC ALGORITHM (AGE)
/*************************************************************************************/

// AGE using BLX cross operator
// @return Total generations
int age_blx(const vector<Example> training, vector<double>& w) {
  Population pop;
  int iter = 0;
  int age = 1;
  int num_genes = w.size();
  int num_cross = 1.0 * SIZE_AGE / 2;  // Expected crosses (pc = 1)
  float pmut = pm * SIZE_AGE * num_genes;
  uniform_int_distribution<int> random_int(0, num_genes - 1);

#if DEBUG >= 1
  cout << "[AGE-BLX] Genes por cromosoma: " << w.size() << endl;
#endif

  // 1. Build and evaluate initial population
  init_population(pop, SIZE_AGG, w.size(), training);
  iter += SIZE_AGG;

  while (iter < MAX_ITER) {
    IntermediatePopulation pop_temp;
    Population new_pop;

    // 2. Select intermediate population (already evaluated)
    pop_temp.resize(SIZE_AGE);
    for (int i = 0; i < SIZE_AGE; i++) {
      pop_temp[i] = selection(pop);
    }

    // 3. Recombine intermediate population
    for (int i = 0; i < 2 * num_cross; i += 2) {
      auto offspring = blx_cross(pop_temp[i], pop_temp[i+1]);
      pop_temp[i] = offspring.first;
      pop_temp[i+1] = offspring.second;
    }

    // 4. Mutate intermediate population
    uniform_real_distribution<double> random_real(0.0, 1.0);
    for (int i = 0; i < SIZE_AGE; i++) {
      if (random_real(generator) <= pmut) {
        int gene = random_int(generator);
        mutate(pop_temp[i], gene);

#if DEBUG >= 2
        cout << "[AGE-CA] Mutación " << i + 1 << ":"
             << " gen " << gene << " del cromosoma "
             << i << endl;
#endif

      }
    }

    // 5. Evaluate intermediate population
    for (int i = 0; i < SIZE_AGE; i++) {
      pop_temp[i].fitness = evaluate(training, pop_temp[i].w);
      iter++;
      new_pop.insert(pop_temp[i]);
    }

    // 6. Change previous population
    auto worst = pop.begin();
    auto second_worst = ++pop.begin();
    auto current_best = new_pop.rbegin();
    auto current_second_best = ++new_pop.rbegin();

#if DEBUG >= 2
    cout << "[AGE-BLX] Fitness mejor hijo: " << current_best->fitness << endl;
    cout << "[AGE-BLX] Fitness segundo mejor hijo " << current_second_best->fitness << endl;
    cout << "[AGE-BLX] Peor fitness población anterior: " << worst->fitness << endl;
    cout << "[AGE-BLX] Segundo peor fitness población anterior: "
         << second_worst->fitness << endl;
#endif

    // NOTE: This replacement scheme is only valid when SIZE_AGE == 2

    // Case 1: both descendants survive
    if (current_second_best->fitness > second_worst->fitness) {

#if DEBUG >= 1
      cout << "[AGE-BLX] Sobreviven los dos hijos" << endl;
#endif

      pop.erase(second_worst);
      pop.erase(pop.begin());
      pop.insert(*current_second_best);
      pop.insert(*current_best);
    }

    // Case 2: only the best descendant survives
    else if (current_best->fitness > worst->fitness) {

#if DEBUG >= 1
      cout << "[AGE-BLX] Sobrevive solo el mejor hijo" << endl;
#endif

      pop.erase(worst);
      pop.insert(*current_best);
    }

    // 7. New generation
    age++;

#if DEBUG >= 1
    cout << "[AGE-BLX] Número de iteraciones: " << iter << " -----------" << endl;
#endif

#if TABLE == 2
    cout << iter << " " << pop.rbegin()->fitness << endl;
#endif

  }

  // Choose best chromosome as solution
  w = pop.rbegin()->w;

#if DEBUG >= 1
  cout << "[AGE-BLX] Fitness solución: " << pop.rbegin()->fitness << endl;
#endif

  return age;
}

// AGE using arithmetic cross operator
// Return Total generations
int age_ca(const vector<Example> training, vector<double>& w) {
  Population pop;
  int iter = 0;
  int age = 1;
  int num_genes = w.size();
  int num_cross = 1.0 * SIZE_AGE / 2;  // Expected crosses (pc = 1)
  float pmut = pm * SIZE_AGE * num_genes;
  uniform_int_distribution<int> random_int(0, num_genes - 1);

#if DEBUG >= 1
  cout << "[AGE-CA] Genes por cromosoma: " << w.size() << endl;
#endif

  // 1. Build and evaluate initial population
  init_population(pop, SIZE_AGG, w.size(), training);
  iter += SIZE_AGG;

  while (iter < MAX_ITER) {
    IntermediatePopulation pop_temp;
    Population new_pop;

    // 2. Select intermediate population (already evaluated)
    // NOTE: we select 2 * SIZE_AGE because we only get one descendant for
    // every two parents, even though we might not use every selected chromosome.
    pop_temp.resize(2 * SIZE_AGE);
    for (int i = 0; i < 2 * SIZE_AGE; i++) {
      pop_temp[i] = selection(pop);
    }

    // 3. Recombine intermediate population
    for (int i = 0; i < 2 * num_cross; i++) {
      pop_temp[i] = arithmetic_cross(pop_temp[i], pop_temp[2 * SIZE_AGE - i - 1]);
    }

    // 4. Mutate intermediate population
    uniform_real_distribution<double> random_real(0.0, 1.0);
    for (int i = 0; i < SIZE_AGE; i++) {
      if (random_real(generator) <= pmut) {
        int gene = random_int(generator);
        mutate(pop_temp[i], gene);

#if DEBUG >= 2
        cout << "[AGE-CA] Mutación " << i + 1 << ":"
             << " gen " << gene << " del cromosoma "
             << i << endl;
#endif
      }
    }

    // 5. Evaluate intermediate population
    for (int i = 0; i < SIZE_AGE; i++) {
      pop_temp[i].fitness = evaluate(training, pop_temp[i].w);
      iter++;
      new_pop.insert(pop_temp[i]);
    }

    // 6. Change previous population
    auto worst = pop.begin();
    auto second_worst = ++pop.begin();
    auto current_best = new_pop.rbegin();
    auto current_second_best = ++new_pop.rbegin();

#if DEBUG >= 2
    cout << "[AGE-CA] Fitness mejor hijo: " << current_best->fitness << endl;
    cout << "[AGE-CA] Fitness segundo mejor hijo " << current_second_best->fitness << endl;
    cout << "[AGE-CA] Peor fitness población anterior: " << worst->fitness << endl;
    cout << "[AGE-CA] Segundo peor fitness población anterior: "
         << second_worst->fitness << endl;
#endif

    // NOTE: This replacement scheme is only valid when SIZE_AGE == 2

    // Case 1: both descendants survive
    if (current_second_best->fitness > second_worst->fitness) {

#if DEBUG >= 1
      cout << "[AGE-CA] Sobreviven los dos hijos" << endl;
#endif

      pop.erase(second_worst);
      pop.erase(pop.begin());
      pop.insert(*current_second_best);
      pop.insert(*current_best);
    }

    // Case 2: only the best descendant survives
    else if (current_best->fitness > worst->fitness) {

#if DEBUG >= 1
      cout << "[AGE-CA] Sobrevive solo el mejor hijo" << endl;
#endif

      pop.erase(worst);
      pop.insert(*current_best);
    }

    // 7. New generation
    age++;

#if DEBUG >= 1
    cout << "[AGE-CA] Número de iteraciones: " << iter << " -----------" << endl;
#endif

#if TABLE == 2
    cout << iter << " " << pop.rbegin()->fitness << endl;
#endif

  }

  // Choose best chromosome as solution
  w = pop.rbegin()->w;

#if DEBUG >= 1
  cout << "[AGE-CA] Fitness solución: " << pop.rbegin()->fitness << endl;
#endif

  return age;
}

/*************************************************************************************/
/* MEMETIC ALGORITHMS (AM)
/*************************************************************************************/

// AM-(10, 1.0)
// Apply local search to all chromosomes every 10 generations
int am_1(const vector<Example> training, vector<double>& w) {
  Population pop;
  Population::reverse_iterator best_parent;  // Elitism
  int iter = 0;
  int age = 1;
  int total_genes = w.size() * SIZE_AM;
  int num_cross = pc * (SIZE_AM / 2);  // Expected crosses
  uniform_int_distribution<int> random_int(0, total_genes - 1);

#if DEBUG >= 1
  cout << "[AM-(10, 1.0)] Genes por cromosoma: " << w.size() << endl;
#endif

  // 1. Build and evaluate initial population
  init_population(pop, SIZE_AM, w.size(), training);
  iter += SIZE_AM;

  while (iter < MAX_ITER) {
    IntermediatePopulation pop_temp;
    Population new_pop;

    best_parent = pop.rbegin();  // Save best parent for elitism

#if DEBUG >= 2
      cout << "[AM-(10, 1.0)] Mejor fitness actual: "
           << best_parent->fitness << endl;
#endif

    // 2. Select intermediate population (already evaluated)
    pop_temp.resize(SIZE_AM);
    for (int i = 0; i < SIZE_AM; i++) {
      pop_temp[i] = selection(pop);
    }

    // 3. Recombine intermediate population
    for (int i = 0; i < 2 * num_cross; i += 2) {
      auto offspring = blx_cross(pop_temp[i], pop_temp[i+1]);
      pop_temp[i] = offspring.first;
      pop_temp[i+1] = offspring.second;
    }

    // 4. Mutate intermediate population
    set<int> mutated;
    int num_mut = expected_mutations(total_genes);
    for (int i = 0; i < num_mut; i++) {
      int comp;

      // Select (coded) component to mutate, without repetition
      while(mutated.size() == i) {
        comp = random_int(generator);
        mutated.insert(comp);
      }

      int selected = comp / w.size();
      int gene = comp % w.size();

      mutate(pop_temp[selected], gene);

#if DEBUG >= 2
      cout << "[AM-(10, 1.0)] Mutación " << i + 1 << ":"
           << " gen " << comp % w.size() << " del cromosoma "
           << (comp / w.size()) << endl;
#endif

    }

    // 5. Evaluate, replace original population and apply elitism
    for (int i = 0; i < SIZE_AM; i++) {
      if (pop_temp[i].fitness == -1.0) {
        pop_temp[i].fitness = evaluate(training, pop_temp[i].w);
        iter++;
      }
      new_pop.insert(pop_temp[i]);
    }

    auto current_best = new_pop.rbegin();

#if DEBUG >= 2
      cout << "[AM-(10, 1.0)] Mejor fitness intermedio: "
           << current_best->fitness << endl;
#endif

    if (current_best->fitness < best_parent->fitness) {

#if DEBUG >= 2
      cout << "[AM-(10, 1.0)] Reemplazo elitista" << endl;
#endif

      // Replace worst chromosome of intermediate population
      new_pop.erase(new_pop.begin());
      new_pop.insert(*best_parent);
    }

    // 6. Replace previous population entirely
    pop = new_pop;

    // 7. Apply low intensity local search to every chromosome
    if (age % FREQ_BL == 0) {

#if DEBUG >= 1
      cout << "[AM-(10, 1.0)] Aplica BL" << endl;
#endif

      new_pop.clear();
      for (auto it = pop.begin(); it != pop.end(); ++it) {
        Chromosome c = *it;
        iter += low_intensity_local_search(training, c);
        new_pop.insert(c);
      }

      pop = new_pop;
    }

    // 7. New generation
    age++;

#if DEBUG >= 1
    cout << "[AM-(10, 1.0)] Número de iteraciones: " << iter << " -----------" << endl;
#endif

#if TABLE == 2
    cout << iter << " " << pop.rbegin()->fitness << endl;
#endif

  }

  // Choose best chromosome as solution
  w = pop.rbegin()->w;

#if DEBUG >= 1
  cout << "[AM-(10, 1.0)] Fitness solución: " << pop.rbegin()->fitness << endl;
#endif

  return age;
}

// AM-(10, 0.1)
// Apply local search to chromosomes every 10 generations with probability pls
int am_2(const vector<Example> training, vector<double>& w) {
  Population pop;
  Population::reverse_iterator best_parent;  // Elitism
  int iter = 0;
  int age = 1;
  int total_genes = w.size() * SIZE_AM;
  int num_cross = pc * (SIZE_AM / 2);  // Expected crosses
  uniform_int_distribution<int> random_int(0, total_genes - 1);

#if DEBUG >= 1
  cout << "[AM-(10, 0.1)] Genes por cromosoma: " << w.size() << endl;
#endif

  // 1. Build and evaluate initial population
  init_population(pop, SIZE_AM, w.size(), training);
  iter += SIZE_AM;

  while (iter < MAX_ITER) {
    IntermediatePopulation pop_temp;
    Population new_pop;

    best_parent = pop.rbegin();  // Save best parent for elitism

#if DEBUG >= 2
      cout << "[AM-(10, 0.1)] Mejor fitness actual: "
           << best_parent->fitness << endl;
#endif

    // 2. Select intermediate population (already evaluated)
    pop_temp.resize(SIZE_AM);
    for (int i = 0; i < SIZE_AM; i++) {
      pop_temp[i] = selection(pop);
    }

    // 3. Recombine intermediate population
    for (int i = 0; i < 2 * num_cross; i += 2) {
      auto offspring = blx_cross(pop_temp[i], pop_temp[i+1]);
      pop_temp[i] = offspring.first;
      pop_temp[i+1] = offspring.second;
    }

    // 4. Mutate intermediate population
    set<int> mutated;
    int num_mut = expected_mutations(total_genes);
    for (int i = 0; i < num_mut; i++) {
      int comp;

      // Select (coded) component to mutate, without repetition
      while(mutated.size() == i) {
        comp = random_int(generator);
        mutated.insert(comp);
      }

      int selected = comp / w.size();
      int gene = comp % w.size();

      mutate(pop_temp[selected], gene);

#if DEBUG >= 2
      cout << "[AM-(10, 0.1)] Mutación " << i + 1 << ":"
           << " gen " << comp % w.size() << " del cromosoma "
           << (comp / w.size()) << endl;
#endif

    }

    // 5. Evaluate, replace original population and apply elitism
    for (int i = 0; i < SIZE_AM; i++) {
      if (pop_temp[i].fitness == -1.0) {
        pop_temp[i].fitness = evaluate(training, pop_temp[i].w);
        iter++;
      }
      new_pop.insert(pop_temp[i]);
    }

    auto current_best = new_pop.rbegin();

#if DEBUG >= 2
      cout << "[AM-(10, 0.1)] Mejor fitness intermedio: "
           << current_best->fitness << endl;
#endif

    if (current_best->fitness < best_parent->fitness) {

#if DEBUG >= 2
      cout << "[AM-(10, 0.1)] Reemplazo elitista" << endl;
#endif

      // Replace worst chromosome of intermediate population
      new_pop.erase(new_pop.begin());
      new_pop.insert(*best_parent);
    }

    // 6. Replace previous population entirely
    pop = new_pop;

    // 7. Apply low intensity local search to every chromosome
    if (age % FREQ_BL == 0) {

#if DEBUG >= 1
      cout << "[AM-(10, 0.1)] Aplica BL" << endl;
#endif

      // NOTE: Expected applications are pls * SIZE_AM = 0.1 * 10 = 1
      uniform_int_distribution<int> random_int(0, SIZE_AM - 1);
      auto it = pop.begin();
      advance(it, random_int(generator));
      Chromosome c = *it;
      iter += low_intensity_local_search(training, c);
      pop.erase(it);
      pop.insert(c);
    }

    // 7. New generation
    age++;

#if DEBUG >= 1
    cout << "[AM-(10, 0.1)] Número de iteraciones: " << iter << " -----------" << endl;
#endif

#if TABLE == 2
    cout << iter << " " << pop.rbegin()->fitness << endl;
#endif

  }

  // Choose best chromosome as solution
  w = pop.rbegin()->w;

#if DEBUG >= 1
  cout << "[AM-(10, 0.1)] Fitness solución: " << pop.rbegin()->fitness << endl;
#endif

  return age;
}

// AM-(10, 0.1)
// Apply local search to the best 0.1*N chromosomes every 10 generations
int am_3(const vector<Example> training, vector<double>& w) {
  Population pop;
  Population::reverse_iterator best_parent;  // Elitism
  int iter = 0;
  int age = 1;
  int total_genes = w.size() * SIZE_AM;
  int num_cross = pc * (SIZE_AM / 2);  // Expected crosses
  uniform_int_distribution<int> random_int(0, total_genes - 1);

#if DEBUG >= 1
  cout << "[AM-(10, 0.1 mej)] Genes por cromosoma: " << w.size() << endl;
#endif

  // 1. Build and evaluate initial population
  init_population(pop, SIZE_AM, w.size(), training);
  iter += SIZE_AM;

  while (iter < MAX_ITER) {
    IntermediatePopulation pop_temp;
    Population new_pop;

    best_parent = pop.rbegin();  // Save best parent for elitism

#if DEBUG >= 2
      cout << "[AM-(10, 0.1 mej)] Mejor fitness actual: "
           << best_parent->fitness << endl;
#endif

    // 2. Select intermediate population (already evaluated)
    pop_temp.resize(SIZE_AM);
    for (int i = 0; i < SIZE_AM; i++) {
      pop_temp[i] = selection(pop);
    }

    // 3. Recombine intermediate population
    for (int i = 0; i < 2 * num_cross; i += 2) {
      auto offspring = blx_cross(pop_temp[i], pop_temp[i+1]);
      pop_temp[i] = offspring.first;
      pop_temp[i+1] = offspring.second;
    }

    // 4. Mutate intermediate population
    set<int> mutated;
    int num_mut = expected_mutations(total_genes);
    for (int i = 0; i < num_mut; i++) {
      int comp;

      // Select (coded) component to mutate, without repetition
      while(mutated.size() == i) {
        comp = random_int(generator);
        mutated.insert(comp);
      }

      int selected = comp / w.size();
      int gene = comp % w.size();

      mutate(pop_temp[selected], gene);

#if DEBUG >= 2
      cout << "[AM-(10, 0.1 mej)] Mutación " << i + 1 << ":"
           << " gen " << comp % w.size() << " del cromosoma "
           << (comp / w.size()) << endl;
#endif

    }

    // 5. Evaluate, replace original population and apply elitism
    for (int i = 0; i < SIZE_AM; i++) {
      if (pop_temp[i].fitness == -1.0) {
        pop_temp[i].fitness = evaluate(training, pop_temp[i].w);
        iter++;
      }
      new_pop.insert(pop_temp[i]);
    }

    auto current_best = new_pop.rbegin();

#if DEBUG >= 2
      cout << "[AM-(10, 0.1 mej)] Mejor fitness intermedio: "
           << current_best->fitness << endl;
#endif

    if (current_best->fitness < best_parent->fitness) {

#if DEBUG >= 2
      cout << "[AM-(10, 0.1 mej)] Reemplazo elitista" << endl;
#endif

      // Replace worst chromosome of intermediate population
      new_pop.erase(new_pop.begin());
      new_pop.insert(*best_parent);
    }

    // 6. Replace previous population entirely
    pop = new_pop;

    // 7. Apply low intensity local search to every chromosome
    if (age % FREQ_BL == 0) {

#if DEBUG >= 1
      cout << "[AM-(10, 0.1 mej)] Aplica BL" << endl;
#endif

      // NOTE: Expected applications are pls * SIZE_AM = 0.1 * 10 = 1
      auto it = --pop.end();  // Best chromosome
      Chromosome c = *it;
      iter += low_intensity_local_search(training, c);
      pop.erase(it);
      pop.insert(c);
    }

    // 7. New generation
    age++;

#if DEBUG >= 1
    cout << "[AM-(10, 0.1 mej)] Número de iteraciones: " << iter << " -----------" << endl;
#endif

#if TABLE == 2
    cout << iter << " " << pop.rbegin()->fitness << endl;
#endif

  }

  // Choose best chromosome as solution
  w = pop.rbegin()->w;

#if DEBUG >= 1
  cout << "[AM-(10, 0.1 mej)] Fitness solución: " << pop.rbegin()->fitness << endl;
#endif

  return age;
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
void run_p2(const string& filename) {

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
  function<int(const vector<Example>&, vector<double>&)> algorithms[NUM_ALGORITHMS] = {
    agg_blx,
    agg_ca,
    age_blx,
    age_ca,
    am_1,
    am_2,
    am_3
  };

  // Run every algorithm
  for (int p = 0; p < NUM_ALGORITHMS; p++) {

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
      int generations = algorithms[p](training, w);  // Call algorithm
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
      cout << "Generaciones totales: " << generations << endl;
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

  for (int p = 0;  p < NUM_ALGORITHMS; p++) {

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

    generator = default_random_engine(seed);

    for (int i = 2; i < argc; i++)
      run_p2(argv[i]);
  }

  else {
    generator = default_random_engine(seed);

    // Dataset 1: colposcopy
    run_p2("data/colposcopy_normalizados.csv");

    // Dataset 2: ionosphere
    run_p2("data/ionosphere_normalizados.csv");

    // Dataset 3: texture
    run_p2("data/texture_normalizados.csv");
  }
}
